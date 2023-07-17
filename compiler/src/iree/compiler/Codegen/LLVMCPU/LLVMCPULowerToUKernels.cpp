// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPULowerToUKernelsPass
    : LLVMCPULowerToUKernelsBase<LLVMCPULowerToUKernelsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override;
};
} // namespace

/// Returns `true` if an `outsOperand` value is initialized to zero.
static bool isInitializedToZero(Value outsOperand) {
  auto fillOp = outsOperand.getDefiningOp<linalg::FillOp>();
  if (!fillOp)
    return false;
  Value fillVal = fillOp.getDpsInputOperand(0)->get();
  return matchPattern(fillVal, m_Zero()) ||
         matchPattern(fillVal, m_AnyZeroFloat());
}

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *ukernelName, RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  if (isVMVXBackend(targetAttr)) {
    result.name = std::string("vmvx.") + ukernelName;
    // TODO(#12327): Based on description in the issue, add an attribute
    // `vm.import.module` and set it to `vmvx`. This only works on `vmvx`
    // backend (obviously), but is enough to unblock while the proper fix
    // lands. For now there are a bunch of attributes set on the function, but
    // this should be made more controllable based on the backend.
    result.defAttrs.emplace_back(rewriter.getStringAttr("vm.import.module"),
                                 rewriter.getStringAttr("vmvx"));
  } else {
    result.name = std::string("iree_uk_") + ukernelName;
    result.defAttrs.emplace_back(
        rewriter.getStringAttr("hal.import.fields"),
        rewriter.getArrayAttr({rewriter.getStringAttr("processor_data")}));
    result.defAttrs.emplace_back(rewriter.getStringAttr("hal.import.bitcode"),
                                 rewriter.getBoolAttr(true));
    result.defAttrs.emplace_back(
        rewriter.getStringAttr("hal.import.cconv"),
        IREE::HAL::CallingConventionAttr::get(
            rewriter.getContext(),
            IREE::HAL::CallingConvention::ParameterStruct));
  }
  return result;
}

/// Matches an (linalg.fill -> )? linalg.mmt4d operation sequence and converts
/// it into a iree_codegen.ukernel.mmt4d operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, linalg::Mmt4DOp op) {
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto lhsType = llvm::cast<ShapedType>(lhs.getType());
  auto rhsType = llvm::cast<ShapedType>(rhs.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_I8I8I32;
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F32F32F32;
  } else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F16F16F32;
  } else if (lhsElemType.isF16() && rhsElemType.isF16() &&
             outElemType.isF16()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F16F16F16;
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_BF16BF16F32;
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isBF16()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_BF16BF16BF16;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  // Check if the accumulator is zero-filled.
  if (isInitializedToZero(out)) {
    // Not setting flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE, so the mmt4d op won't
    // read the existing accumulator, so its defining op can be discarded.
    if (auto fillOp = out.getDefiningOp<linalg::FillOp>()) {
      out = fillOp.getDpsInitOperand(0)->get();
    }
  } else {
    // Tell the mmt4d op to read the existing accumulator.
    flags |= IREE_UK_FLAG_MMT4D_ACCUMULATE;
  }
  Location loc = op.getLoc();
  Value m = rewriter.create<tensor::DimOp>(loc, lhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, rhs, 0);
  Value k = rewriter.create<tensor::DimOp>(loc, rhs, 1);

  auto getDimAsI32 = [](RewriterBase &rewriter, Location loc, Value value,
                        int dim) -> Value {
    return rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(),
        rewriter.create<tensor::DimOp>(loc, value, dim));
  };
  Value m0 = getDimAsI32(rewriter, loc, lhs, 2);
  Value n0 = getDimAsI32(rewriter, loc, rhs, 2);
  Value k0 = getDimAsI32(rewriter, loc, rhs, 3);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("mmt4d", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out,
      ValueRange{m, n, k, m0, n0, k0, flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, tensor::PackOp op) {
  Value in = op.getSource();
  Value out = op.getDest();
  auto inType = llvm::cast<ShapedType>(in.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (inElemType.isSignlessInteger(8) && outElemType.isSignlessInteger(8)) {
    flags = IREE_UK_FLAG_PACK_TYPE_I8I8;
  } else if (inElemType.isSignlessInteger(32) &&
             outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_PACK_TYPE_I32I32;
  } else if (inElemType.isF32() && outElemType.isF32()) {
    flags = IREE_UK_FLAG_PACK_TYPE_F32F32;
  } else if (inElemType.isF16() && outElemType.isF16()) {
    flags = IREE_UK_FLAG_PACK_TYPE_F16F16;
  } else if (inElemType.isBF16() && outElemType.isBF16()) {
    flags = IREE_UK_FLAG_PACK_TYPE_BF16BF16;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  if (inType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "expected input to be 2D");
  }

  if (outType.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected output to be 4D");
  }

  int64_t innerDimsPos[2] = {0, 1};
  ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
  if (!innerDimsPosArr.empty()) {
    innerDimsPos[0] = innerDimsPosArr[0];
    innerDimsPos[1] = innerDimsPosArr[1];
  }

  int64_t outerDimsPerm[2] = {0, 1};
  ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
  if (!outerDimsPosArr.empty()) {
    outerDimsPerm[0] = outerDimsPosArr[0];
    outerDimsPerm[1] = outerDimsPosArr[1];
  }

  if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
    // nothing to do
  } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_INNER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
  }

  if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
    // nothing to do
  } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
    flags |= IREE_UK_FLAG_PACK_TRANSPOSE_OUTER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
  }

  Location loc = op.getLoc();
  Type i64 = rewriter.getI64Type();

  // The ukernel requires a padding value of type i64. When the element type is
  // a narrower N-bit type, only the least significant N bits of the i64 padding
  // value are used.
  Value paddingVal = op.getPaddingValue();
  // If the pack op didn't have a padding_value attribute, default to 0.
  if (!paddingVal) {
    paddingVal =
        rewriter.create<arith::ConstantOp>(loc, i64, rewriter.getZeroAttr(i64));
  }
  int paddingValBitWidth = paddingVal.getType().getIntOrFloatBitWidth();
  // Non-integer element types get bitcast to integer of same bit width.
  if (!paddingVal.getType().isSignlessInteger()) {
    Type sameWidthIntType = rewriter.getIntegerType(paddingValBitWidth);
    if (!sameWidthIntType) {
      return rewriter.notifyMatchFailure(op, "no integer type with this width");
    }
    paddingVal =
        rewriter.create<arith::BitcastOp>(loc, sameWidthIntType, paddingVal);
  }
  // Element types > 64bits could be supported, when the padding value is a
  // repeating 64-bit pattern. For now, we leave this as not-yet-implemented.
  if (paddingValBitWidth > 64) {
    return rewriter.notifyMatchFailure(op,
                                       "unsupported padding_value bit width");
  }
  // Integers narrower than 64 bit get extended to 64 bits, it doesn't matter
  // how, as the high bits are unused.
  if (paddingValBitWidth < 64) {
    paddingVal = rewriter.create<arith::ExtUIOp>(loc, i64, paddingVal);
  }
  Value in_size0 = rewriter.create<tensor::DimOp>(loc, in, 0);
  Value in_size1 = rewriter.create<tensor::DimOp>(loc, in, 1);
  Value out_size0 = rewriter.create<tensor::DimOp>(loc, out, 0);
  Value out_size1 = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value out_size2 = rewriter.create<tensor::DimOp>(loc, out, 2);
  Value out_size3 = rewriter.create<tensor::DimOp>(loc, out, 3);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("pack", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, in, out,
      ValueRange{in_size0, in_size1, out_size0, out_size1, out_size2, out_size3,
                 paddingVal, flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, tensor::UnPackOp op) {
  Value in = op.getSource();
  Value out = op.getDest();
  auto inType = llvm::cast<ShapedType>(in.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (inElemType.isSignlessInteger(32) && outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_I32I32;
  } else if (inElemType.isF32() && outElemType.isF32()) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_F32F32;
  } else if (inElemType.isF16() && outElemType.isF16()) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_F16F16;
  } else if (inElemType.isBF16() && outElemType.isBF16()) {
    flags = IREE_UK_FLAG_UNPACK_TYPE_BF16BF16;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }

  if (inType.getRank() != 4) {
    return rewriter.notifyMatchFailure(op, "expected input to be 4D");
  }

  if (outType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "expected output to be 2D");
  }

  int64_t innerDimsPos[2] = {0, 1};
  ArrayRef<int64_t> innerDimsPosArr = op.getInnerDimsPos();
  if (!innerDimsPosArr.empty()) {
    innerDimsPos[0] = innerDimsPosArr[0];
    innerDimsPos[1] = innerDimsPosArr[1];
  }

  int64_t outerDimsPerm[2] = {0, 1};
  ArrayRef<int64_t> outerDimsPosArr = op.getOuterDimsPerm();
  if (!outerDimsPosArr.empty()) {
    outerDimsPerm[0] = outerDimsPosArr[0];
    outerDimsPerm[1] = outerDimsPosArr[1];
  }

  if (innerDimsPos[0] == 0 && innerDimsPos[1] == 1) {
    // nothing to do
  } else if (innerDimsPos[0] == 1 && innerDimsPos[1] == 0) {
    flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported inner_dims_pos");
  }

  if (outerDimsPerm[0] == 0 && outerDimsPerm[1] == 1) {
    // nothing to do
  } else if (outerDimsPerm[0] == 1 && outerDimsPerm[1] == 0) {
    flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER;
  } else {
    return rewriter.notifyMatchFailure(op, "unsupported outer_dims_perm");
  }

  Location loc = op.getLoc();
  Value in_size0 = rewriter.create<tensor::DimOp>(loc, in, 0);
  Value in_size1 = rewriter.create<tensor::DimOp>(loc, in, 1);
  Value in_size2 = rewriter.create<tensor::DimOp>(loc, in, 2);
  Value in_size3 = rewriter.create<tensor::DimOp>(loc, in, 3);
  Value out_size0 = rewriter.create<tensor::DimOp>(loc, out, 0);
  Value out_size1 = rewriter.create<tensor::DimOp>(loc, out, 1);
  Value flagsVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32IntegerAttr(flags));
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("unpack", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, in, out,
      ValueRange{in_size0, in_size1, in_size2, in_size3, out_size0, out_size1,
                 flagsVal},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static uint32_t flagForUser(IREE::LinalgExt::EncodingUser user) {
  switch (user) {
  case IREE::LinalgExt::EncodingUser::MATMUL_F32F32F32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32;
  case IREE::LinalgExt::EncodingUser::MATMUL_I8I8I32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32;
  case IREE::LinalgExt::EncodingUser::MATMUL_F16F16F32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F32;
  case IREE::LinalgExt::EncodingUser::MATMUL_F16F16F16:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F16F16F16;
  case IREE::LinalgExt::EncodingUser::MATMUL_BF16BF16F32:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16F32;
  case IREE::LinalgExt::EncodingUser::MATMUL_BF16BF16BF16:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_BF16BF16BF16;
  }
}

static uint32_t flagForRole(IREE::LinalgExt::EncodingRole role) {
  switch (role) {
  case IREE::LinalgExt::EncodingRole::LHS:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_LHS;
  case IREE::LinalgExt::EncodingRole::RHS:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RHS;
  case IREE::LinalgExt::EncodingRole::RESULT:
    return IREE_UK_FLAG_QUERY_TILE_SIZES_OPERAND_ROLE_RESULT;
  }
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, IREE::Codegen::QueryTileSizesOp op) {
  auto tensorType = op.getTensorType().dyn_cast<RankedTensorType>();
  if (!tensorType) {
    return rewriter.notifyMatchFailure(op,
                                       "need a ranked tensor type attribute");
  }
  if (tensorType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "only the 2D case is implemented");
  }
  auto encoding = tensorType.getEncoding()
                      .dyn_cast_or_null<IREE::LinalgExt::EncodingAttr>();
  if (!encoding) {
    return rewriter.notifyMatchFailure(op, "no encoding attribute");
  }
  SmallVector<Type> resultTypes(tensorType.getRank(), rewriter.getIndexType());
  SmallVector<Value> inputValues;
  Location loc = op.getLoc();
  for (int64_t i : tensorType.getShape()) {
    inputValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, i));
  }
  inputValues.push_back(rewriter.create<arith::ConstantIntOp>(
      loc,
      flagForUser(encoding.getUser().getValue()) |
          flagForRole(encoding.getRole().getValue()),
      32));
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("query_tile_sizes.2d", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, resultTypes, fn.name, inputValues, /*outs=*/ValueRange{},
      /*other_operands=*/ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/IntegerAttr{});
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

template <typename OpType>
struct LowerToUKernelPattern : OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }
};

} // namespace

void LLVMCPULowerToUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<LowerToUKernelPattern<linalg::Mmt4DOp>,
                  LowerToUKernelPattern<tensor::PackOp>,
                  LowerToUKernelPattern<tensor::UnPackOp>,
                  LowerToUKernelPattern<IREE::Codegen::QueryTileSizesOp>>(
      context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>> createLLVMCPULowerToUKernelsPass() {
  return std::make_unique<LLVMCPULowerToUKernelsPass>();
}

} // namespace iree_compiler
} // namespace mlir
