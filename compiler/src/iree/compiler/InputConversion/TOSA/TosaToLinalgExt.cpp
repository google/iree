//===- TosaToLinalgExt.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert TOSA operations to their LinalgExt equivalent.
//
//===----------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/InputConversion/TOSA/PassDetail.h"
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tosa;

namespace mlir {
namespace iree_compiler {

// Converts tosa.scatter to the iree_linalg_ext.scatter operation. As the
// LinalgExt version is not batched therefore we materialize the batch index
// for each update.
class ScatterConversion : public OpRewritePattern<tosa::ScatterOp> {
 public:
  using OpRewritePattern<tosa::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ScatterOp op,
                                PatternRewriter &rewriter) const final {
    auto values = op.getValuesIn();
    auto indices = op.getIndices();
    auto updates = op.getInput();
    auto valuesTy = values.getType().dyn_cast<RankedTensorType>();
    auto indicesTy = indices.getType().dyn_cast<RankedTensorType>();
    auto updatesTy = updates.getType().dyn_cast<RankedTensorType>();
    ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (!valuesTy || !indicesTy || !updatesTy)
      return rewriter.notifyMatchFailure(op,
                                         "tosa.gather has unknown input rank");

    // TOSA's scatter does not include a index dimension, instead it implicitly
    // supporteds an index depth of one. We materialize that implicit index of
    // one as follows: [batch, updates] -> [batch, updates, index_depth=1] With
    // a indexing map of [[0], [1, 2]].
    llvm::SmallVector<int64_t> expandIndShape{indicesTy.getDimSize(0),
                                              indicesTy.getDimSize(1), 1};
    SmallVector<ReassociationExprs, 4> expandIndMap;
    expandIndMap.push_back({
        builder.getAffineDimExpr(0),
    });
    expandIndMap.push_back({
        builder.getAffineDimExpr(1),
        builder.getAffineDimExpr(2),
    });

    indices = builder.create<tensor::ExpandShapeOp>(
        indicesTy.clone(expandIndShape), indices, expandIndMap);
    indicesTy = indices.getType().dyn_cast<RankedTensorType>();

    // Materialize the batch indice as LinalgExt scatter is not batched.
    {
      llvm::SmallVector<Value> dynDims;
      for (int i = 0, s = indicesTy.getRank(); i < s; ++i)
        if (indicesTy.isDynamicDim(i))
          dynDims.push_back(builder.create<tensor::DimOp>(indices, i));

      Value empty = builder.create<tensor::EmptyOp>(
          indicesTy.getShape(), indicesTy.getElementType(), dynDims);

      Value batchIdx = nullptr;

      if (indicesTy.getDimSize(0) == 1) {
        Value zero = builder.create<arith::ConstantOp>(
            rewriter.getZeroAttr(indicesTy.getElementType()));
        batchIdx = builder.create<linalg::FillOp>(zero, empty).getResult(0);
      } else {
        SmallVector<utils::IteratorType> iterators(
            indicesTy.getRank(), utils::IteratorType::parallel);
        SmallVector<AffineMap, 3> indexingMaps(
            2, builder.getMultiDimIdentityMap(indicesTy.getRank()));
        batchIdx = builder
                       .create<linalg::GenericOp>(
                           indicesTy, indices, empty, indexingMaps, iterators,
                           [&](OpBuilder &nestedBuilder, Location nestedLoc,
                               ValueRange blockArgs) {
                             ImplicitLocOpBuilder b(op.getLoc(), nestedBuilder);
                             auto index = b.create<linalg::IndexOp>(0);
                             auto cast = b.create<arith::IndexCastOp>(
                                 indicesTy.getElementType(), index);
                             b.create<linalg::YieldOp>(cast.getResult());
                           })
                       .getResult(0);
      }

      indicesTy =
          indicesTy.clone({indicesTy.getDimSize(0), indicesTy.getDimSize(1), 2})
              .cast<RankedTensorType>();
      indices = builder.create<tosa::ConcatOp>(indicesTy,
                                               ValueRange{batchIdx, indices},
                                               rewriter.getI64IntegerAttr(2));
    }

    auto collapseBatch = [](Value value, ImplicitLocOpBuilder &b) -> Value {
      auto valueTy = value.getType().cast<ShapedType>();
      llvm::SmallVector<int64_t> collapseShape(valueTy.getShape().drop_front());
      llvm::SmallVector<ReassociationExprs, 4> collapseMap(valueTy.getRank() -
                                                           1);
      collapseMap.front().push_back(b.getAffineDimExpr(0));
      for (int i = 0, s = collapseMap.size(); i < s; ++i) {
        collapseMap[i].push_back(b.getAffineDimExpr(i + 1));
      }

      int64_t batch = valueTy.getShape().front();
      int64_t rows = collapseShape.front();
      int64_t batchDyn = batch == ShapedType::kDynamic;
      int64_t rowsDyn = rows == ShapedType::kDynamic;
      collapseShape[0] =
          (batchDyn || rowsDyn) ? ShapedType::kDynamic : batch * rows;

      return b.create<tensor::CollapseShapeOp>(valueTy.clone(collapseShape),
                                               value, collapseMap);
    };

    indices = collapseBatch(indices, builder);
    updates = collapseBatch(updates, builder);

    // Create the LinalgExt scatter operation.
    auto scatter = builder.create<IREE::LinalgExt::ScatterOp>(
        TypeRange{values.getType()}, ValueRange{updates, indices},
        ValueRange{values}, builder.getDenseI64ArrayAttr({0, 1}),
        builder.getBoolAttr(true));

    llvm::SmallVector<Type> args(2, valuesTy.getElementType());
    Block *scatterBody =
        builder.createBlock(&scatter.getRegion(), {}, args,
                            llvm::SmallVector<Location>(2, op.getLoc()));
    builder.setInsertionPointToStart(scatterBody);
    builder.create<IREE::LinalgExt::YieldOp>(scatterBody->getArgument(0));
    rewriter.replaceOp(op, scatter.getResult(0));
    return success();
  }
};

struct TosaToLinalgExtPass : public TosaToLinalgExtBase<TosaToLinalgExtPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<tosa::ScatterOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    FunctionOpInterface func = getOperation();
    mlir::iree_compiler::populateTosaToLinalgExtPatterns(&patterns);
    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

void populateTosaToLinalgExtPatterns(RewritePatternSet *patterns) {
  patterns->add<ScatterConversion>(patterns->getContext());
}

std::unique_ptr<OperationPass<func::FuncOp>> createTosaToLinalgExt() {
  return std::make_unique<TosaToLinalgExtPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
