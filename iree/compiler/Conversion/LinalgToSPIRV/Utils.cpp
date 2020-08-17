// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- Utils.cpp - Utility functions used in Linalg to SPIR-V lowering ----===//
//
// Implementaiton of utility functions used while lowering from Linalg to SPIRV.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"

#include "iree/compiler/Conversion/LinalgToSPIRV/MarkerUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/MemorySpace.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

LogicalResult updateWorkGroupSize(FuncOp funcOp,
                                  ArrayRef<int64_t> workGroupSize) {
  // Need to update both the surrounding FuncOp that has the spv.entry_point_abi
  // attribute, and the hal.executable.
  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body))
    return funcOp.emitError("unhandled dispatch function with multiple blocks");

  if (workGroupSize.size() != 3)
    return funcOp.emitError("expected workgroup size to have three entries");
  SmallVector<int32_t, 3> workGroupSizeVec = llvm::to_vector<3>(llvm::map_range(
      workGroupSize, [](int64_t v) { return static_cast<int32_t>(v); }));

  funcOp.setAttr(
      spirv::getEntryPointABIAttrName(),
      spirv::getEntryPointABIAttr(workGroupSizeVec, funcOp.getContext()));
  return success();
}

LogicalResult copyToWorkgroupMemory(OpBuilder &b, Value src, Value dst) {
  auto copyOp = b.create<linalg::CopyOp>(src.getLoc(), src, dst);
  setMarker(copyOp, getCopyToWorkgroupMemoryMarker());
  return success();
}

Optional<Value> allocateWorkgroupMemory(OpBuilder &b, SubViewOp subview,
                                        ArrayRef<Value> boundingSubViewSize,
                                        OperationFolder *folder) {
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape = llvm::to_vector<2>(
      llvm::map_range(boundingSubViewSize, [](Value v) -> int64_t {
        APInt value;
        if (matchPattern(v, m_ConstantInt(&value))) return value.getSExtValue();
        return -1;
      }));
  if (llvm::any_of(shape, [](int64_t v) { return v == -1; })) return {};
  MemRefType allocType = MemRefType::get(
      shape, subview.getType().getElementType(), {}, getWorkgroupMemorySpace());
  Value buffer = b.create<AllocOp>(subview.getLoc(), allocType);
  return buffer;
}

LogicalResult deallocateWorkgroupMemory(OpBuilder &b, Value buffer) {
  auto allocOp = buffer.getDefiningOp<AllocOp>();
  b.create<DeallocOp>(allocOp.getLoc(), buffer);
  return success();
}

template <typename GPUIdOp, typename GPUCountOp>
static linalg::ProcInfo getGPUProcessorIdAndCountImpl(OpBuilder &builder,
                                                      Location loc,
                                                      StringRef dim) {
  Type indexType = builder.getIndexType();
  return {
      builder.create<GPUIdOp>(loc, indexType, builder.getStringAttr(dim)),
      builder.create<GPUCountOp>(loc, indexType, builder.getStringAttr(dim))};
}

template <>
linalg::ProcInfo getGPUProcessorIdAndCountImpl<GPUGlobalId, GPUGlobalCount>(
    OpBuilder &builder, Location loc, StringRef dim) {
  Type indexType = builder.getIndexType();
  Value gridDim = builder.create<gpu::GridDimOp>(loc, indexType,
                                                 builder.getStringAttr(dim));
  Value blockId = builder.create<gpu::BlockIdOp>(loc, indexType,
                                                 builder.getStringAttr(dim));
  Value blockDim = builder.create<gpu::BlockDimOp>(loc, indexType,
                                                   builder.getStringAttr(dim));
  Value threadId = builder.create<gpu::ThreadIdOp>(loc, indexType,
                                                   builder.getStringAttr(dim));
  return {builder.create<AddIOp>(
              loc, builder.create<MulIOp>(loc, blockId, blockDim), threadId),
          builder.create<MulIOp>(loc, blockDim, gridDim)};
}

template <typename GPUIdOp, typename GPUCountOp>
static SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCountsImpl(
    OpBuilder &builder, Location loc, unsigned numDims) {
  SmallVector<linalg::ProcInfo, 2> procInfo(numDims);
  std::array<StringRef, kNumDims> dims{"x", "y", "z"};
  for (unsigned i = 0; i < numDims; ++i) {
    procInfo[numDims - 1 - i] =
        getGPUProcessorIdAndCountImpl<GPUIdOp, GPUCountOp>(builder, loc,
                                                           dims[i]);
  }
  return procInfo;
}

#define DEFINE_PROCINFO_CALLBACK_FNS(IdOp, CountOp)                            \
  template <>                                                                  \
  SmallVector<linalg::ProcInfo, 2> getGPUProcessorIdsAndCounts<IdOp, CountOp>( \
      OpBuilder & builder, Location loc, unsigned numDims) {                   \
    return getGPUProcessorIdsAndCountsImpl<IdOp, CountOp>(builder, loc,        \
                                                          numDims);            \
  }

// clang-format off
DEFINE_PROCINFO_CALLBACK_FNS(gpu::BlockIdOp, gpu::GridDimOp)
DEFINE_PROCINFO_CALLBACK_FNS(gpu::ThreadIdOp, gpu::BlockDimOp)
DEFINE_PROCINFO_CALLBACK_FNS(GPUGlobalId, GPUGlobalCount)
// clang-format on

#undef DEFINE_PROCINFO_CALLBACK_FNS
}  // namespace iree_compiler
}  // namespace mlir
