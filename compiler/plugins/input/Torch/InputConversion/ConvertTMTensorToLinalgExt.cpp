// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <numeric>

#include "compiler/plugins/input/Torch/InputConversion/PassDetail.h"
#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"

namespace mlir::iree_compiler::TorchInput {

namespace {

template <typename SrcOpTy, typename TargetOpTy>
struct TMTensorOpConversion : public OpRewritePattern<SrcOpTy> {
  using OpRewritePattern<SrcOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpTy srcOp,
                                PatternRewriter &rewriter) const override {
    OperationState state(srcOp->getLoc(), TargetOpTy::getOperationName(),
                         srcOp->getOperands(), srcOp->getResultTypes(),
                         srcOp->getAttrs(), srcOp->getSuccessors());
    for (Region &srcRegion : srcOp->getRegions()) {
      Region *targetRegion = state.addRegion();
      rewriter.inlineRegionBefore(srcRegion, *targetRegion,
                                  targetRegion->begin());
    }
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }
};

struct ScatterOpConversion
    : public OpRewritePattern<mlir::torch::TMTensor::ScatterOp> {
  using OpRewritePattern<mlir::torch::TMTensor::ScatterOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indicesTy = op.getIndicesType();
    if (!indicesTy.hasRank())
      return failure();

    if (indicesTy.isDynamicDim(indicesTy.getRank() - 1)) {
      return rewriter.notifyMatchFailure(op, "number of indices is unknown");
    }

    auto numIndices = indicesTy.getShape().back();
    llvm::SmallVector<int64_t> dimMap(numIndices);
    for (int i = 0; i < numIndices; i++)
      dimMap[i] = i;

    auto scatterOp = rewriter.create<IREE::LinalgExt::ScatterOp>(
        op.getLoc(), op->getResultTypes(), op.getInputs(), op.getOutputs(),
        dimMap, op.getUniqueIndices());

    rewriter.inlineRegionBefore(op.getRegion(), scatterOp.getRegion(),
                                scatterOp.getRegion().begin());
    rewriter.replaceOp(op, scatterOp->getResults());
    return success();
  }
};
} // namespace

static Value collapseBatches(PatternRewriter &rewriter, Location loc,
                             Value val) {
  auto valSizes = val.getType().cast<RankedTensorType>().getShape();
  int64_t newBatch =
      std::accumulate(valSizes.begin(), valSizes.end() - 2, 1,
                      [](int64_t x, int64_t y) { return x * y; });
  Type elementType = val.getType().cast<RankedTensorType>().getElementType();
  SmallVector<int64_t> newSizes{newBatch};
  newSizes.append(valSizes.end() - 2, valSizes.end());
  Type newType = RankedTensorType::get(newSizes, elementType);

  auto rank = valSizes.size();
  SmallVector<int64_t> collapsed;
  for (auto i = 0; i < rank - 2; i++)
    collapsed.push_back(i);

  SmallVector<ReassociationIndices> reassociation(3);
  reassociation[0].append(collapsed);
  reassociation[1].push_back(rank - 2);
  reassociation[2].push_back(rank - 1);

  return rewriter
      .create<tensor::CollapseShapeOp>(loc, newType, val, reassociation)
      .getResult();
}
static Value expandBatches(PatternRewriter &rewriter, Location loc,
                           SmallVector<int64_t> batchSizes, Value val) {
  auto valSizes = val.getType().cast<RankedTensorType>().getShape();
  Type elementType = val.getType().cast<RankedTensorType>().getElementType();
  SmallVector<int64_t> newSizes(batchSizes);
  newSizes.append(valSizes.end() - 2, valSizes.end());
  auto rank = newSizes.size();
  Type newType = RankedTensorType::get(newSizes, elementType);

  SmallVector<ReassociationIndices> reassociation(3);
  for (auto i = 0; i < batchSizes.size(); i++)
    reassociation[0].push_back(i);
  reassociation[1].push_back(rank - 2);
  reassociation[2].push_back(rank - 1);

  return rewriter
      .create<tensor::ExpandShapeOp>(loc, newType, val, reassociation)
      .getResult();
}
struct AttentionOpConversion
    : public OpRewritePattern<mlir::torch::TMTensor::AttentionOp> {
  using OpRewritePattern<mlir::torch::TMTensor::AttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    auto sizes = query.getType().cast<RankedTensorType>().getShape();
    SmallVector<int64_t> batchSizes(sizes.begin(), sizes.end() - 2);

    if (sizes.size() > 3) {
      query = collapseBatches(rewriter, loc, query);
      key = collapseBatches(rewriter, loc, key);
      value = collapseBatches(rewriter, loc, value);
    }

    SmallVector<int64_t> resultShape(
        op->getResultTypes()[0].cast<RankedTensorType>().getShape());
    SmallVector<int64_t> collapsedResultShape;
    collapsedResultShape.push_back(
        std::accumulate(resultShape.begin(), resultShape.end() - 2, 1,
                        [](int64_t x, int64_t y) { return x * y; }));
    collapsedResultShape.append(resultShape.end() - 2, resultShape.end());
    Type elementType =
        query.getType().cast<RankedTensorType>().getElementType();
    auto collapsedResultType =
        RankedTensorType::get(collapsedResultShape, elementType);
    Value collapsedResult = rewriter.create<tensor::EmptyOp>(
        loc, collapsedResultShape, elementType);

    // TODO: This is a hack. This should be replaced with a simple getScale()
    // when support for scaling is plumbed to TMTensor on the torch-mlir side.
    // Until then, we are using the default value used in scaled dot product
    // attention by PyTorch (most models use the default value because it makes
    // the variance of the result of softmax 1 when the mean of Q, K is 0).
    // We use scale = 1 / sqrt(d), where d is the head dimension.
    // See https://paperswithcode.com/method/scaled for more details.
    //
    // TODO: We are currently assuming that head dimension is dim = -1. Once we
    // have support for batch dims using more general indexing maps, we should
    // change this and rely on more general mechanisms.
    // TODO: We are currently not handeling dynamic shape of head dimensions at
    // all. This is because it messes with dispatch formation. This should be
    // fixed.
    ArrayRef<int64_t> queryShape = op.getQueryType().getShape();
    int64_t headDim = queryShape[queryShape.size() - 1];
    if (headDim <= 0) {
      return op->emitOpError("NYI: Dynamic head dimension");
    }

    // Attention only works for FloatType.
    FloatType targetType = cast<FloatType>(op.getQueryType().getElementType());

    double dk = static_cast<double>(headDim);
    dk = 1.0f / std::sqrt(dk);
    Value scale = rewriter.create<arith::ConstantOp>(
        loc, targetType, rewriter.getFloatAttr(targetType, dk));

    auto attention = rewriter.create<IREE::LinalgExt::AttentionOp>(
        loc, collapsedResultType, SmallVector<Value>{query, key, value, scale},
        collapsedResult);

    if (sizes.size() > 3)
      rewriter.replaceOp(
          op, expandBatches(rewriter, loc, batchSizes, attention.getResult(0)));
    else
      rewriter.replaceOp(op, attention.getResult(0));
    return success();
  }
};

namespace {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertTMTensorToLinalgExtPass
    : public ConvertTMTensorToLinalgExtBase<ConvertTMTensorToLinalgExtPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

#define INSERT_TMTENSOR_CONVERSION_PATTERN(Op)                                 \
  patterns.add<                                                                \
      TMTensorOpConversion<mlir::torch::TMTensor::Op, IREE::LinalgExt::Op>>(   \
      context);

    INSERT_TMTENSOR_CONVERSION_PATTERN(YieldOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(ScanOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(SortOp);

#undef INSERT_TMTENSOR_CONVERSION_PATTERN

    patterns.add<ScatterOpConversion>(context);
    patterns.add<AttentionOpConversion>(context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertTMTensorToLinalgExtPass() {
  return std::make_unique<ConvertTMTensorToLinalgExtPass>();
}

} // namespace mlir::iree_compiler::TorchInput
