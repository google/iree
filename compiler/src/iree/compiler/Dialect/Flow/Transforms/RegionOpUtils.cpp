// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

/// Return `true` if the given type is a ShapedType and has at least one
/// dynamic dimension.
static bool hasDynamicShape(Type t) {
  auto shapedType = t.dyn_cast<ShapedType>();
  if (!shapedType) return false;
  return !shapedType.hasStaticShape();
}

/// Reify the dynamic dimensions of the given value.
static LogicalResult reifyDynamicResultDims(OpBuilder &b, Value value,
                                            SmallVector<Value> &dynamicDims) {
  OpBuilder::InsertionGuard guard(b);

  // Case 1: No dynamic result dims.
  if (!hasDynamicShape(value.getType())) return success();

  // There is at least one dynamic dimension, continue...
  ShapedType shapedType = value.getType().cast<ShapedType>();

  // Case 2: Value is a block argument.
  if (auto bbArg = value.dyn_cast<BlockArgument>()) {
    b.setInsertionPointToStart(bbArg.getOwner());
    for (int64_t i = 0; i < shapedType.getRank(); ++i) {
      if (shapedType.isDynamicDim(i)) {
        Value dim = b.create<tensor::DimOp>(bbArg.getLoc(), bbArg, i);
        dynamicDims.push_back(dim);
      }
    }
    return success();
  }

  // Value is an OpResult.
  Operation *op = value.getDefiningOp();
  OpResult opResult = value.cast<OpResult>();
  b.setInsertionPoint(op);

  // Case 3: Value is tied. Reify the dimensions of the tied operand.
  auto tiedOp = dyn_cast<IREE::Util::TiedOpInterface>(op);
  if (tiedOp) {
    Value tiedOperand = tiedOp.getTiedResultOperand(value);
    if (tiedOperand && tiedOperand.getType() == value.getType())
      return reifyDynamicResultDims(b, tiedOperand, dynamicDims);
  }

  // Case 4: Query ShapeAwareOpInterface.
  auto shapeAwareOp = dyn_cast<IREE::Util::ShapeAwareOpInterface>(op);
  if (shapeAwareOp) {
    ValueRange dims =
        shapeAwareOp.getResultDynamicDims(opResult.getResultNumber());
    dynamicDims.append(dims.begin(), dims.end());
    return success();
  }

  // Case 5: Query ReifyRankedShapedTypeOpInterface.
  auto reifyShapeOp = dyn_cast<ReifyRankedShapedTypeOpInterface>(op);
  if (reifyShapeOp) {
    ReifiedRankedShapedTypeDims dims;
    if (failed(reifyShapeOp.reifyResultShapes(b, dims))) return failure();
    for (int64_t i = 0; i < shapedType.getRank(); ++i)
      if (shapedType.isDynamicDim(i))
        dynamicDims.push_back(dims[opResult.getResultNumber()][i]);
    return success();
  }

  return failure();
}

// Append a result to the given DispatchRegionOp. The newly created
// DispatchRegionOp is returned.
FailureOr<Flow::DispatchRegionOp> appendDispatchRegionResult(
    RewriterBase &rewriter, Flow::DispatchRegionOp regionOp, Value result) {
  OpBuilder::InsertionGuard guard(rewriter);

  // Determine dynamic result dims.
  rewriter.setInsertionPoint(regionOp);
  SmallVector<Value> dynamicDims(regionOp.getResultDims().begin(),
                                 regionOp.getResultDims().end());
  if (failed(reifyDynamicResultDims(rewriter, result, dynamicDims)))
    return failure();

  // Determine result types of new RegionOp.
  SmallVector<Type> resultTypes(regionOp.getResultTypes().begin(),
                                regionOp.getResultTypes().end());
  resultTypes.push_back(result.getType());

  // Create new DispatchRegionOp and move over the body.
  auto newRegionOp = rewriter.create<Flow::DispatchRegionOp>(
      regionOp->getLoc(), resultTypes, dynamicDims);
  newRegionOp.getBody().takeBody(regionOp.getBody());
  rewriter.replaceOp(
      regionOp, newRegionOp.getResults().take_front(regionOp->getNumResults()));

  // Update terminator.
  Flow::ReturnOp returnOp =
      cast<Flow::ReturnOp>(newRegionOp.getBody().front().getTerminator());
  SmallVector<Value> returnedValues(returnOp.getOperands().begin(),
                                    returnOp.getOperands().end());
  returnedValues.push_back(result);
  returnOp.operandsMutable().assign(returnedValues);

  return newRegionOp;
}

Flow::DispatchRegionOp makeEmptyDispatchRegion(OpBuilder &builder,
                                               Location loc) {
  OpBuilder::InsertionGuard guard(builder);

  // Create RegionOp.
  auto regionOp = builder.create<Flow::DispatchRegionOp>(
      loc, /*resultTypes=*/TypeRange(), /*dynamicDims=*/ValueRange());
  Block &body = regionOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&body);
  builder.create<Flow::ReturnOp>(loc, ValueRange());

  return regionOp;
}

// Clone a `target` op that is preceding the given dispatch region op into the
// dispatch region.
LogicalResult clonePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target,
    Flow::DispatchRegionOp regionOp) {
  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesInsideOfRegion;
  for (OpOperand &use : target->getUses()) {
    if (regionOp->isProperAncestor(use.getOwner()))
      usesInsideOfRegion.push_back(&use);
  }

  // Clone op into dispatch region.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&body);
  Operation *newTargetOp = rewriter.clone(*target);

  // Replace all uses in the dispatch region.
  for (OpOperand *use : usesInsideOfRegion) {
    rewriter.updateRootInPlace(use->getOwner(), [&]() {
      use->set(newTargetOp->getResult(
          use->get().cast<OpResult>().getResultNumber()));
    });
  }

  return success();
}

// Move a `target` op that is preceding the given dispatch region op into the
// dispatch region.
FailureOr<Flow::DispatchRegionOp> movePrecedingOpIntoDispatchRegion(
    RewriterBase &rewriter, Operation *target,
    Flow::DispatchRegionOp regionOp) {
#ifndef NDEBUG
  DominanceInfo domInfo;
  for (OpOperand &use : target->getUses()) {
    if (regionOp->isProperAncestor(use.getOwner())) continue;
    assert(domInfo.properlyDominates(regionOp, use.getOwner()) &&
           "found use that does not post-dominate target");
  }
#endif  // NDEBUG

  Block &body = regionOp.getBody().front();

  // Gather all uses of `target`.
  SmallVector<OpOperand *> usesOutsideOfRegion;
  for (OpOperand &use : target->getUses())
    if (!regionOp->isProperAncestor(use.getOwner()))
      usesOutsideOfRegion.push_back(&use);

  // Move op into dispatch region.
  target->moveBefore(&body.front());

  // Replace all uses outside of the dispatch region.
  if (!usesOutsideOfRegion.empty()) {
    unsigned previousNumResults = regionOp->getNumResults();

    // Note: Appending results one-by-one here so that this can be extended to
    // specific results in the future. Many ops have just one result, so this
    // should not be a large overhead.
    for (Value v : target->getResults()) {
      auto newRegionOp = appendDispatchRegionResult(rewriter, regionOp, v);
      if (failed(newRegionOp)) return failure();
      regionOp = *newRegionOp;
    }

    // Replace uses of `target` after the dispatch region.
    for (OpOperand *use : usesOutsideOfRegion) {
      rewriter.updateRootInPlace(use->getOwner(), [&]() {
        use->set(
            regionOp->getResult(previousNumResults +
                                use->get().cast<OpResult>().getResultNumber()));
      });
    }
  }

  return regionOp;
}

FailureOr<Flow::DispatchRegionOp> wrapOpInDispatchRegion(RewriterBase &rewriter,
                                                         Operation *op) {
  // Make an empty dispatch region right before the op.
  rewriter.setInsertionPointAfter(op);
  Flow::DispatchRegionOp regionOp =
      Flow::makeEmptyDispatchRegion(rewriter, op->getLoc());

  // Move the op into the dispatch region.
  auto newRegionOp = movePrecedingOpIntoDispatchRegion(rewriter, op, regionOp);
  return newRegionOp;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
