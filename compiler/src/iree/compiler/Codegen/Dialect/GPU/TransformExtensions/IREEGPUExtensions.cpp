// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensions.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE {

transform_dialect::IREEGPUExtensions::IREEGPUExtensions() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensionsOps.cpp.inc"
      >();
}

//===---------------------------------------------------------------------===//
// ApplyUnrollMultiMmaOp
//===---------------------------------------------------------------------===//

static bool isReductionIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::reduction;
}
static bool isParallelIterator(Attribute attr) {
  return cast<IREE::GPU::IteratorTypeAttr>(attr).getValue() ==
         utils::IteratorType::parallel;
}

/// Pick an unrolling order that reuses the LHS register.
static std::optional<SmallVector<int64_t>>
gpuMultiMmaUnrollOrder(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : mmaOp.getIndexingMapsArray()[0].getResults()) {
    dims.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(mmaOp.getIteratorTypes())) {
    if (isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

static std::optional<SmallVector<int64_t>> getMultiMmaUnitShape(Operation *op) {
  IREE::GPU::MultiMmaOp mmaOp = dyn_cast<IREE::GPU::MultiMmaOp>(op);
  if (!mmaOp) {
    return std::nullopt;
  }
  SmallVector<int64_t> targetOuterShape(mmaOp.getIteratorTypes().size(), 1);
  return targetOuterShape;
}

void transform_dialect::ApplyUnrollMultiMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  GPU::populateIREEGPUVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(getMultiMmaUnitShape)
                    .setUnrollTraversalOrderFn(gpuMultiMmaUnrollOrder));
}

//===---------------------------------------------------------------------===//
// ApplyVectorizeMultiMmaOp
//===---------------------------------------------------------------------===//

void transform_dialect::ApplyVectorizeMultiMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  IREE::GPU::populateIREEGPUVectorizationPatterns(patterns);
}

} // namespace mlir::iree_compiler::IREE

void mlir::iree_compiler::registerTransformDialectIREEGPUExtension(
    DialectRegistry &registry) {
  registry.addExtensions<
      mlir::iree_compiler::IREE::transform_dialect::IREEGPUExtensions>();
}

#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensionsOps.cpp.inc"
