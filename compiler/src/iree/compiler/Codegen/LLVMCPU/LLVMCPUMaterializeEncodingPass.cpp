// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Codegen/Common/EncodingInfo.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

using namespace IREE::LinalgExt;
using IREE::HAL::ExecutableTargetAttr;

namespace {

static MatmulTileParams chooseMatmulTileParamsGeneric() { return {8, 4, 8}; }

static MatmulTileParams
chooseMatmulTileParamsAArch64(MatmulType type, ExecutableTargetAttr target) {
  switch (type) {
  case MatmulType::F32F32F32:
  case MatmulType::F16F16F32:
  case MatmulType::F16F16F16:
  case MatmulType::BF16BF16F32:
  case MatmulType::BF16BF16BF16:
    // Note: 16-bit floating point types currently use the same tile size as
    // f32. This makes sense when either (1) the accumulator is f32, or (2)
    // the arithmetic will have to expand f16 to f32 in registers. We may
    // reconsider when taking advantage of native f16/bf16 arithmetic when the
    // accumulator itself is f16/bf16.
    return {8, 1, 8};
  case MatmulType::I8I8I32:
    if (hasFeature(target, "+i8mm")) {
      // Aim to use SMMLA.
      return {8, 8, 8};
    }
    if (hasFeature(target, "+dotprod")) {
      // Aim to use SDOT.
      return {8, 4, 8};
    }
    return {8, 1, 8};
  default:
    assert(false);
    return {};
  }
}

static MatmulTileParams
chooseMatmulTileParamsX86_64(MatmulType type, ExecutableTargetAttr target) {
  switch (type) {
  case MatmulType::F32F32F32:
  case MatmulType::F16F16F32:
  case MatmulType::F16F16F16:
  case MatmulType::BF16BF16F32:
  case MatmulType::BF16BF16BF16:
    // Note: 16-bit floating point types currently use the same tile size as
    // f32. This makes sense when either (1) the accumulator is f32, or (2)
    // the arithmetic will have to expand f16 to f32 in registers. We may
    // reconsider when taking advantage of native f16/bf16 arithmetic when the
    // accumulator itself is f16/bf16.
    if (hasFeature(target, "+avx512f")) {
      return {16, 1, 16};
    }
    if (hasFeature(target, "+avx")) {
      // Note: for good performance, most +avx users will also want to add
      // +fma, but that's a local instruction selection detail and the tile
      // layout is unaffected, as there are enough registers even with the
      // need for intermediate product registers when +fma is not used.
      return {8, 1, 8};
    }
    // SSE fallback.
    return {8, 1, 4};
  case MatmulType::I8I8I32:
    if (hasFeature(target, "+avx512vnni")) {
      // Aim to use VPDPWSSD. This is the same tile size as with VPMADDWD
      // as the only difference is that VPDPWSSD accumulates. VPDPBUSD would
      // call for {16, 4, 16} but we can't use it because of its unsigned LHS.
      return {16, 2, 16};
    }
    if (hasFeature(target, "+avx512bw")) {
      // Aim to use VPMADDWD (zmm).
      return {16, 2, 16};
    }
    if (hasFeature(target, "+avx2")) {
      // Aim to use VPMADDWD (ymm).
      return {8, 2, 8};
    }
    // SSE fallback. Aim to use PMADDWD (xmm).
    return {8, 2, 4};
  default:
    assert(false);
    return {};
  }
}

static MatmulTileParams chooseMatmulTileParams(MatmulType type,
                                               ExecutableTargetAttr target) {
  if (isAArch64(target)) {
    return chooseMatmulTileParamsAArch64(type, target);
  }
  if (isX86_64(target)) {
    return chooseMatmulTileParamsX86_64(type, target);
  }
  return chooseMatmulTileParamsGeneric();
}

struct LLVMCPUMaterializeEncodingPass
    : public LLVMCPUMaterializeEncodingBase<LLVMCPUMaterializeEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, affine::AffineDialect,
                IREE::Flow::FlowDialect, IREE::LinalgExt::IREELinalgExtDialect,
                IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override;
};

} // namespace

void LLVMCPUMaterializeEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();
  RewritePatternSet materializeEncodingPattern(context);
  auto targetAttr = ExecutableTargetAttr::lookup(operation);
  MaterializeEncodingTypeConverter typeConverter(
      [targetAttr](
          RankedTensorType tensorType) -> FailureOr<MaterializeEncodingInfo> {
        std::optional<TensorEncoding> encoding = getEncoding(tensorType);
        if (!encoding)
          return failure();

        auto matmulType = getMatmulType(*encoding);
        auto matmulOperandRole = getMatmulOperandRole(*encoding);
        if (!matmulType || !matmulOperandRole) {
          return failure();
        }
        MatmulTileParams tileParams =
            chooseMatmulTileParams(*matmulType, targetAttr);
        auto encodingInfo = chooseEncodingInfoForMatmul(
            *matmulType, *matmulOperandRole, tileParams);
        adjustTileSizesToNarrowStaticShape(encodingInfo, tensorType.getShape());
        return encodingInfo;
      });
  MaterializeEncodingConversionTarget target(*context);
  auto materializeEncodingValueFn = getMaterializeEncodingValueFn(targetAttr);
  populateMaterializeEncodingIntoPackUnPackPatterns(materializeEncodingPattern,
                                                    target, typeConverter,
                                                    materializeEncodingValueFn);

  if (failed(applyPartialConversion(operation, target,
                                    std::move(materializeEncodingPattern)))) {
    operation.emitOpError("materialization failed");
    return signalPassFailure();
  }

  // Add patterns to fold pack/unpack ops with pad/extract_slice ops and resolve
  // dims ops.
  {
    RewritePatternSet patterns(context);
    tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(operation, std::move(patterns)))) {
      operation.emitOpError("folding patterns failed");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUMaterializeEncodingPass() {
  return std::make_unique<LLVMCPUMaterializeEncodingPass>();
}

} // namespace iree_compiler
} // namespace mlir
