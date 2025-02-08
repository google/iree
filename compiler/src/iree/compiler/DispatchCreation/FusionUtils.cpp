// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.h --- Implementation of fusion utility functions -----===//
//===----------------------------------------------------------------------===//

#include "compiler/src/iree/compiler/DispatchCreation/FusionUtils.h"
#include "compiler/src/iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::iree_compiler::DispatchCreation {

bool areFusableAsElementwiseOps(MLIRContext *context, OpOperand *fusedOperand,
                                ElementwiseOpsFusabilityOptions options) {
  Operation *producerOp = fusedOperand->get().getDefiningOp();
  Operation *consumerOp = fusedOperand->getOwner();
  if (!producerOp)
    return false;

  // Check for i1 return types, if so aggressively fuse to avoid `i1` buffers.
  if (llvm::all_of(producerOp->getResultTypes(), [](Type t) {
        if (t.isInteger(1))
          return true;
        if (auto shapedType = llvm::dyn_cast<ShapedType>(t)) {
          if (shapedType.getElementType().isInteger(1))
            return true;
        }
        return false;
      })) {
    return true;
  }

  // Don't fuse if all of the consumer maps aren't projected permutations.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {
    if (!llvm::all_of(
            linalgConsumerOp.getIndexingMapsArray(),
            [](AffineMap map) { return map.isProjectedPermutation(); })) {
      return false;
    }
  }

  // If the generic op is "just" copy, then fuse always.
  Block &body = producerOp->getRegion(0).front();
  if (std::begin(body)->hasTrait<OpTrait::IsTerminator>())
    return true;
  if (llvm::all_of(body.getArguments(),
                   [](BlockArgument arg) { return arg.use_empty(); })) {
    // The operands aren't used, its just an `linalg.index` op.
    return true;
  }

  // If producer does not have a single user, dont fuse.
  if (!producerOp->hasOneUse()) {
    return false;
  }

  // Do no fuse bitextend-like operations with producers. Such ops are cloned
  // into all their use dispatches. So fusing producer with consumer here would
  // then result in producer also getting cloned into many dispatches which is
  // against the thumb rule of fusion to not introduce additional computation
  // (except for bit-extend ops). If the consumer has only one use, then this
  // fusion is fine since cloning wont result in redundant computation of the
  // producer. (Also note that the producer is always an elementwise operation).
  if (IREE::LinalgExt::isBitExtendOp(consumerOp) && !consumerOp->hasOneUse()) {
    return false;
  }

  auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp);
  if (!linalgConsumerOp) {
    return false;
  }

  // Do not fuse with bit-truncate-like operations with their consumers, unless
  // the consumer has only one ins operand and is an elementwise operation. The
  // elementwise oepration implies that the `outs` operand is not real usage
  // (and is typically a `tensor.empty`), so the core condition is that there is
  // only one "real" operand of the consumer.
  if (!options.fuseTruncateOps &&
      IREE::LinalgExt::isBitTruncateOp(producerOp) &&
      !(linalgConsumerOp.getNumLoops() ==
            linalgConsumerOp.getNumParallelLoops() &&
        linalgConsumerOp.getNumDpsInputs() == 1)) {
    return false;
  }

  // If the producer has a single use (this op), only fuse if
  // - 1) The consumer op is all parallel loops. The parallelism of the consumer
  //      can be used as a way to amortize cost of redundant computation
  // - 2) If consumer op is a reduction, only fuse if the indexing map in the
  //      consumer for the producer result is a permutation. If it is a
  //      broadcast this ends up redundantly computing operations without more
  //      parallelism.
  if (linalgConsumerOp.getNumParallelLoops() !=
      linalgConsumerOp.getNumLoops()) {
    if (!linalgConsumerOp.getMatchingIndexingMap(fusedOperand)
             .isPermutation()) {
      return false;
    }
    if (!options.fuseMultiReduction &&
        linalgConsumerOp.getNumReductionLoops() != 1) {
      return false;
    }
    if (linalg::isaContractionOpInterface(linalgConsumerOp) ||
        linalg::isaConvolutionOpInterface(linalgConsumerOp)) {
      return false;
    }
  }
  return true;
}

} // namespace mlir::iree_compiler::DispatchCreation
