// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_pydm;
namespace pydm_d = mlir::iree_pydm;

using llvm::dbgs;
#define DEBUG_TYPE "pydm_opt"

namespace {

struct BlockAccessInfo {
  // Tracks any variable, value mapping that has been hoisted to the block
  // arguments.
  DenseMap<Value, Value> blockArgVariableValueMap;

  // Map of variable alloc value to most terminal value of the variable
  // within the block.
  DenseMap<Value, Value> variableValueMap;

  // Set of any loads that are live.
  DenseSet<Operation *> liveLoads;
};

struct VariablesToSSAPass : public VariablesToSSABase<VariablesToSSAPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Verify that the structure we need is valid.
    for (auto &block : getOperation().getBody().getBlocks()) {
      if (failed(verifyBlockIsLegal(block))) {
        return signalPassFailure();
      }
    }

    // Canonicalize and accumulate information about per-block accesses.
    DenseMap<Block *, BlockAccessInfo> blockAccessInfos;
    bool changed = false;
    for (int i = 0; i < 100; ++i) {
      LLVM_DEBUG(dbgs() << "--- Iteration on all blocks\n");
      changed = false;
      for (auto &block : getOperation().getBody().getBlocks()) {
        auto &info = blockAccessInfos[&block];
        if (canonicalizeBlockVariableAccess(block, info)) changed = true;
        hoistLoadsFromBlock(block, info);

        // Invalidate internal value map and re-initialize from block arg
        // carried values.
        info.variableValueMap.clear();
        info.variableValueMap = info.blockArgVariableValueMap;
      }

      if (!changed) break;
    }

    // We should now have eliminated as many loads as possible, so we can
    // DCE any free variable stores (since free variables do not escape, we
    // can just eliminate them with some simple checks).
    elideDeadFreeVarStores();
  }

  void elideDeadFreeVarStores() {
    getOperation().walk([](AllocFreeVarOp allocOp) {
      bool canElide = true;
      SmallVector<Operation *> storeOps;
      for (auto &use : allocOp.getResult().getUses()) {
        if (llvm::isa<LoadVarOp>(use.getOwner())) {
          canElide = false;
        } else if (llvm::isa<StoreVarOp>(use.getOwner())) {
          storeOps.push_back(use.getOwner());
        } else {
          canElide = false;
        }
      }
      if (canElide) {
        for (auto *storeOp : storeOps) {
          storeOp->erase();
        }
        allocOp->erase();
      }
    });
  }

  // This pass must operate before any CFG operations have been performed
  // which may cause variables to be sunk into block arguments.
  LogicalResult verifyBlockIsLegal(Block &block) {
    for (BlockArgument arg : block.getArguments()) {
      if (arg.getType().isa<FreeVarRefType>()) {
        return emitError(getOperation().getLoc())
               << "cannot convert variables to SSA on a function which carries "
                  "variable references across block boundaries";
      }
    }
    Operation *terminator = block.getTerminator();
    if (!terminator ||
        !llvm::isa<BranchOp, CondBranchOp, iree_pydm::ReturnOp>(terminator)) {
      return emitError(terminator->getLoc())
             << "unsupported terminator for block";
    }
    return success();
  }

  // Canonicalizes variable accesses within a block such that:
  //   - Redundant loads are eliminated.
  bool canonicalizeBlockVariableAccess(Block &block, BlockAccessInfo &info) {
    bool changed = false;
    SmallVector<Operation *> elidedOps;
    for (Operation &op : block) {
      if (auto storeOp = llvm::dyn_cast<StoreVarOp>(op)) {
        Value &currentValue = info.variableValueMap[storeOp.var()];
        currentValue = storeOp.value();
        LLVM_DEBUG(dbgs() << "Initialize store: " << currentValue << "\n");
      } else if (auto loadOp = llvm::dyn_cast<LoadVarOp>(op)) {
        Value &currentValue = info.variableValueMap[loadOp.var()];
        if (currentValue) {
          LLVM_DEBUG(dbgs() << "Forward load from: " << currentValue << "\n");
          Value replacementValue = currentValue;
          if (loadOp.getResult().getType() != currentValue.getType()) {
            OpBuilder builder(loadOp);
            replacementValue = builder.create<StaticInfoCastOp>(
                loadOp.getLoc(), loadOp.getResult().getType(), currentValue);
          }
          loadOp.getResult().replaceAllUsesWith(replacementValue);
          elidedOps.push_back(loadOp);
          changed = true;
        } else {
          LLVM_DEBUG(dbgs() << "Initialize load: " << loadOp << "\n");
          currentValue = loadOp.getResult();
          info.liveLoads.insert(loadOp);
        }
      }
    }

    for (auto *op : elidedOps) {
      op->erase();
    }
    return changed;
  }

  // Lifts any live loads into the block's phi arguments and move the
  // load up to the predecessors. This assumes that the function is in a
  // legal form where all allocs are done in the entry block.
  void hoistLoadsFromBlock(Block &block, BlockAccessInfo &info) {
    SmallVector<std::tuple<Location, Value, Type>> loadVarTypes;
    // Redirect each live load to a block argument.
    for (Operation *genericLoadOp : info.liveLoads) {
      auto loadOp = llvm::cast<LoadVarOp>(genericLoadOp);
      loadVarTypes.emplace_back(loadOp.getLoc(), loadOp.var(),
                                loadOp.getResult().getType());
      Value newArg = block.addArgument(loadOp.getResult().getType());
      info.blockArgVariableValueMap[loadOp.var()] = newArg;
      loadOp.getResult().replaceAllUsesWith(newArg);
      loadOp->erase();
    }

    // In each predecessor, rematerialize the load.
    for (Block *pred : block.getPredecessors()) {
      Operation *terminator = pred->getTerminator();
      OpBuilder builder(terminator);
      SmallVector<Value> newLoadValues;
      for (auto &it : loadVarTypes) {
        Location loc = std::get<0>(it);
        Value varValue = std::get<1>(it);
        Type loadType = std::get<2>(it);
        newLoadValues.push_back(
            builder.create<LoadVarOp>(loc, loadType, varValue));
      }

      if (auto branchOp = llvm::dyn_cast<BranchOp>(terminator)) {
        branchOp.destOperandsMutable().append(newLoadValues);
      } else if (auto condBranchOp = llvm::dyn_cast<CondBranchOp>(terminator)) {
        if (condBranchOp.trueDest() == &block) {
          condBranchOp.trueDestOperandsMutable().append(newLoadValues);
        } else if (condBranchOp.falseDest() == &block) {
          condBranchOp.falseDestOperandsMutable().append(newLoadValues);
        }
      }
    }

    info.liveLoads.clear();
  }
};

}  // namespace

std::unique_ptr<OperationPass<pydm_d::FuncOp>>
mlir::iree_pydm::createVariablesToSSAPass() {
  return std::make_unique<VariablesToSSAPass>();
}
