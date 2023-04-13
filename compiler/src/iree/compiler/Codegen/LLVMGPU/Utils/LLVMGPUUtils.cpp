// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;

#define DEBUG_TYPE "llvm-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
namespace iree_compiler {

static bool isContiguousStore(Operation* write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0) || transferWrite.getMask()) {
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  return false;
}

static bool isContiguousRead(Operation* read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  return false;
}

static Value getMemrefOperand(Operation* op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getSource();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getSource();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

static Value getMask(Operation* op) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(op);
  if (!transferRead || !transferRead.getMask()) return Value();
  auto maskOp = transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  return maskOp.getOperand(0);
}

static Value getValueStored(Operation* writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

static Operation::operand_range getIndices(Operation* op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

void createAsyncGroups(RewriterBase& rewriter, func::FuncOp funcOp,
                       bool useMMASync) {
  LLVM_DEBUG(DBGS() << "Start asyncGroups: useMMASync=" << useMMASync << "\n");
  llvm::SmallSetVector<Operation*, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation* writeOp) {
    if (!isContiguousStore(writeOp)) {
      return WalkResult::advance();
    }
    LLVM_DEBUG(DBGS() << "--candidate writeOp: " << writeOp << "\n");
    Value vectorVal = getValueStored(writeOp);
    if (vectorVal.getType().cast<VectorType>().getRank() != 1) {
      LLVM_DEBUG(
          DBGS()
          << "----writeOp is not an inbounds 1-D minor identity -> Skip \n");
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    if (!hasSharedMemoryAddressSpace(
            memrefOperand.getType().cast<MemRefType>())) {
      LLVM_DEBUG(DBGS() << "----address space is not workgroup -> Skip \n");
      return WalkResult::advance();
    }
    Operation* readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LLVM_DEBUG(DBGS() << "----no readOp defining the writeOp -> Skip \n");
      return WalkResult::advance();
    }

    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      if (transferRead.getMask()) {
        auto paddingCst =
            transferRead.getPadding().getDefiningOp<arith::ConstantFloatOp>();
        if (!paddingCst || !paddingCst.value().isZero()) {
          LLVM_DEBUG(DBGS() << "----read padding value is not 0.f -> Skip \n");
          return WalkResult::advance();
        }
        auto maskOp =
            transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
        if (!maskOp) {
          LLVM_DEBUG(
              DBGS()
              << "----read mask is not a vector.create_mask op -> Skip \n");
          return WalkResult::advance();
        }
      }
    }

    VectorType vecType = vectorVal.getType().cast<VectorType>();
    if (!((vecType.getElementType().isF32() && vecType.getNumElements() <= 4) ||
          (vecType.getElementType().isF16() &&
           vecType.getNumElements() <= 8))) {
      LLVM_DEBUG(
          DBGS() << "----readOp is not (<=4)xf32 or (<=8)xf16 -> Skip \n");
      return WalkResult::advance();
    }

    LLVM_DEBUG(DBGS() << "--writeOp can be made async -> SUCCESS\n");
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation*> group;
    Operation* writeOp = *copyToSharedMem.begin();
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation* nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation* readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        if (!hasSharedMemoryAddressSpace(
                memrefOperand.getType().cast<MemRefType>())) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    for (Operation* writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      Value vectorVal = getValueStored(writeOp);
      Operation* readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value mask = getMask(readOp);
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          rewriter.getIndexAttr(
              vectorVal.getType().cast<VectorType>().getNumElements()),
          mask,
          /*bypassL1=*/useMMASync ? rewriter.getUnitAttr() : UnitAttr());
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                              nullptr);
    // Clean up old stores.
    for (Operation* writeOp : group) rewriter.eraseOp(writeOp);
  }
}

void reorderTranspose(IRRewriter& rewriter, func::FuncOp funcOp) {
  SmallVector<vector::TransposeOp> transposeOps;
  funcOp.walk([&](Operation* op) {
    if (auto transposeOp = dyn_cast<vector::TransposeOp>(op)) {
      Operation* definingOp = transposeOp.getVector().getDefiningOp();
      if (OpTrait::hasElementwiseMappableTraits(definingOp)) {
        transposeOps.push_back(transposeOp);
      }
    }
    return WalkResult::advance();
  });

  for (auto transposeOp : transposeOps) {
    OpBuilder::InsertionGuard g(rewriter);
    Operation* op = transposeOp.getVector().getDefiningOp();
    rewriter.setInsertionPoint(op);
    SmallVector<int64_t> perm;
    transposeOp.getTransp(perm);
    SmallVector<Value> transposedOperands;
    for (auto operand : op->getOperands()) {
      Value transposed =
          rewriter.create<vector::TransposeOp>(op->getLoc(), operand, perm);
      transposedOperands.push_back(transposed);
    }
    SmallVector<Type> resultTypes{transposedOperands.front().getType()};
    Operation* newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        transposedOperands, resultTypes, op->getAttrs());
    rewriter.replaceAllUsesWith(transposeOp.getResult(), newOp->getResult(0));
  }
}

}  // namespace iree_compiler
}  // namespace mlir
