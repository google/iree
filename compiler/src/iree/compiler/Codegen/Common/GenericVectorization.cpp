// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/TileSizeSelection.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-generic-vectorization"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace {

/// Returns the op that contains lowering config. Checks whether the provided op
/// contains the lowering config and returns it. Otherwise, tries to find the
/// lowering config across the function. If there are multiple ops with the same
/// lowering configs, returns the first one found. Returns failure if there are
/// multiple op with different lowering config.
static FailureOr<Operation *> getRootOp(Operation *op) {
  // Check for self first.
  if (iree_compiler::getLoweringConfig(op)) {
    return op;
  }

  // Get the function op.
  auto funcOp = dyn_cast<func::FuncOp>(op);
  if (!funcOp) {
    funcOp = op->getParentOfType<func::FuncOp>();
  }

  assert(funcOp && "Missing funcOp");

  Operation *rootOp = nullptr;
  mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr rootLoweringConfig;
  auto result = funcOp.walk([&](Operation *op) -> WalkResult {
    auto loweringConfig = iree_compiler::getLoweringConfig(op);
    if (!loweringConfig) {
      return WalkResult::advance();
    }
    if (rootLoweringConfig) {
      if (rootLoweringConfig != loweringConfig) {
        return WalkResult::interrupt();
      }
    } else {
      rootOp = op;
      rootLoweringConfig = loweringConfig;
    }
    return WalkResult::advance();
  });

  if (!rootOp || result.wasInterrupted()) {
    return failure();
  }
  return rootOp;
}

/// Tries to infer the vector sizes from an IR using ValueBounds analysis.
/// Returns failure if vector sizes can't be inferred.
static FailureOr<SmallVector<int64_t>>
inferVectorSizesFromIR(linalg::LinalgOp linalgOp) {
  LLVM_DEBUG(VEC_DBGS() << "Inferring vector sizes for:\n" << linalgOp << "\n");

  SmallVector<int64_t> vectorSizes;
  unsigned numDims = linalgOp.getNumLoops();

  for (int dim = 0; dim < numDims; ++dim) {
    // Map dimension `dim` to an operand dimension that we will use to
    // traverse the U-D chain to get `dim` vector size information.
    SmallVector<std::pair<Value, unsigned>> operandDimPairs;
    linalgOp.mapIterationSpaceDimToAllOperandDims(dim, operandDimPairs);
    if (operandDimPairs.empty()) {
      return failure();
    }

    Value firstOperand = operandDimPairs[0].first;
    unsigned firstOperandDim = operandDimPairs[0].second;

    // Trivial case: `dim` size is available in the operand type.
    int64_t dimSize = llvm::cast<ShapedType>(firstOperand.getType())
                          .getShape()[firstOperandDim];
    if (!ShapedType::isDynamic(dimSize)) {
      vectorSizes.push_back(dimSize);
      LLVM_DEBUG(VEC_DBGS() << "Inferred vector size '" << dimSize
                            << "' for dimension '" << dim << "'\n");
      continue;
    }

    // Use ValueBounds analysis to infer `dim` size upper bound.
    FailureOr<int64_t> maybeDimBound;
    for (auto operandDimPair : operandDimPairs) {
      Value operand = operandDimPair.first;
      unsigned operandDim = operandDimPair.second;
      maybeDimBound = ValueBoundsConstraintSet::computeConstantBound(
          presburger::BoundType::UB, operand, operandDim,
          /*stopCondition=*/nullptr, /*closedUB=*/true);

      if (succeeded(maybeDimBound)) {
        break;
      }
    }

    if (failed(maybeDimBound)) {
      return failure();
    }

    dimSize = maybeDimBound.value();
    vectorSizes.push_back(dimSize);
    LLVM_DEBUG(VEC_DBGS() << "Inferred vector size '" << dimSize
                          << "' for dimension '" << dim << "'\n");
  }

  return vectorSizes;
}

// Returns the vector sizes to vectorize a linalg operation. We try to retrieve
// them from its `lowering_config`, if available. Otherwise, we try to infer
// them from the tiled loops in the IR.
static SmallVector<int64_t> getVectorSizes(linalg::LinalgOp linalgOp) {
  auto loweringConfig = iree_compiler::getLoweringConfig(linalgOp);
  // Give priority to the operation's lowering config.
  if (loweringConfig) {
    TilingConfig tilingConfig(loweringConfig);
    SmallVector<int64_t> vectorShape = tilingConfig.getVectorTileSizes();

    // Replace zeros in vector shape to turn it into a valid vector shape.
    std::replace(vectorShape.begin(), vectorShape.end(), 0, 1);

    LLVM_DEBUG(VEC_DBGS() << "Using vector sizes from 'lowering_config'\n");
    return vectorShape;
  }

  // Try to infer the vector sizes from the IR. If it fails, we can't vectorize
  // this op.
  auto inferredVectorSizes = inferVectorSizesFromIR(linalgOp);
  if (succeeded(inferredVectorSizes)) {
    return *inferredVectorSizes;
  }

  // We couldn't infer the vector sizes for this op so we return all the vector
  // sizes set to zero.
  LLVM_DEBUG(VEC_DBGS() << "Couldn't infer vector sizes\n");
  return SmallVector<int64_t>(linalgOp.getNumLoops(), 0);
}

class GenericVectorizationPass
    : public GenericVectorizationBase<GenericVectorizationPass> {
public:
  using GenericVectorizationBase::GenericVectorizationBase;
  GenericVectorizationPass(const GenericVectorizationPassOptions &options) {
    this->enableVectorMasking.setValue(options.enableVectorMasking);
    this->vectorizePadding.setValue(options.vectorizePadding);
    this->vectorizeGatherAccesses.setValue(options.vectorizeGatherAccesses);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void GenericVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  IRRewriter rewriter(context);
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    if (isa<linalg::LinalgOp>(op))
      candidates.push_back(op);
    if (vectorizePadding && enableVectorMasking && isa<tensor::PadOp>(op))
      candidates.push_back(op);
  });
  for (auto op : candidates) {
    SmallVector<int64_t> vectorSizes;
    if (enableVectorMasking) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        vectorSizes = getVectorSizes(linalgOp);
      } else if (auto padOp = dyn_cast<tensor::PadOp>(op)) {
        auto ty = padOp.getResultType();
        // TODO(hanchung): Infer the vector sizes for pad op after
        // maskedVectorize method allows dynamic result shapes.
        if (!ty.hasStaticShape())
          continue;
        vectorSizes.append(ty.getShape().begin(), ty.getShape().end());
      }
    }
    SmallVector<bool> scalableVecDims(vectorSizes.size(), false);
    (void)linalg::vectorize(rewriter, op, vectorSizes, scalableVecDims,
                            vectorizeGatherAccesses);
  };

  // TODO: Move this down the pipeline once we have the ODM-based masking
  // representation.
  RewritePatternSet vectorizationPatterns(funcOp.getContext());
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
      vectorizationPatterns);
  vector::populateVectorTransferPermutationMapLoweringPatterns(
      vectorizationPatterns);
  vector::populateVectorReductionToContractPatterns(vectorizationPatterns);
  vector::populateFoldArithExtensionPatterns(vectorizationPatterns);
  vectorizationPatterns.add<linalg::LinalgCopyVTRForwardingPattern,
                            linalg::LinalgCopyVTWForwardingPattern>(
      funcOp.getContext(), /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                      funcOp.getContext());
  vector::TransferWriteOp::getCanonicalizationPatterns(vectorizationPatterns,
                                                       funcOp.getContext());
  populateVectorTransferTensorSliceTransforms(vectorizationPatterns);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  // Apply the pad tensor op vectorization separately to avoid running the
  // GenericPadOpVectorizationPattern too early.
  // TODO: Improve once we have better infrastructure to control pattern
  // application.
  if (vectorizePadding) {
    RewritePatternSet patterns(funcOp.getContext());
    linalg::populatePadOpVectorizationPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
}
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGenericVectorizationPass() {
  return std::make_unique<GenericVectorizationPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>>
createGenericVectorizationPass(const GenericVectorizationPassOptions &options) {
  return std::make_unique<GenericVectorizationPass>(options);
}
} // namespace iree_compiler
} // namespace mlir
