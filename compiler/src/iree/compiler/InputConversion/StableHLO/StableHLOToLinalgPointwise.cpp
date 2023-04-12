// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO/CHLO pointwise ops to Linalg dialect.
// These patterns are separated out to their own file to save on the compilation
// times, given that we instantiate a large number of class templates here.

#include "iree/compiler/InputConversion/StableHLO/LegalizeToLinalgUtils.h"
#include "iree/compiler/InputConversion/StableHLO/MapStableHLOToScalarOp.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "iree/compiler/InputConversion/StableHLO/TypeConversion.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {
namespace {
namespace stablehlo = mlir::stablehlo;

int64_t getRank(Value v) { return cast<ShapedType>(v.getType()).getRank(); }

int64_t getMaxRank(ValueRange operands) {
  int64_t maxRank = 0;
  for (Value operand : operands) {
    maxRank = std::max(maxRank, getRank(operand));
  }
  return maxRank;
}

bool isScalar(Value v) { return getRank(v) == 0; }

/// Inserts block arguments in places where scalar inputs have a nullptr.
SmallVector<Value> interleaveScalarAndBlockArgs(ValueRange scalarInputs,
                                                ValueRange blockArgs) {
  SmallVector<Value> result;
  auto argsIter = blockArgs.begin();
  for (Value scalarInput : scalarInputs) {
    if (scalarInput) {
      result.push_back(scalarInput);
    } else {
      result.push_back(*argsIter);
      ++argsIter;
    }
  }
  return result;
}

struct PointwiseConversionInfo {
  int64_t maxOperandRank = 0;
  ShapedType resultType;
};

/// Checks the preconditions for conversion of pointwise HLO ops to linalg.
/// Returns the max operand rank and the result type on success.
FailureOr<PointwiseConversionInfo> checkOperandsAndResults(
    Operation* op, ValueRange operands, TypeConverter& typeConverter,
    ConversionPatternRewriter& rewriter) {
  int64_t maxRank = getMaxRank(operands);

  // Apply only if all operands are scalar or have the same rank. Some ops,
  // like `mhlo.select`, support implicit broadcasting of scalars.
  if (!llvm::all_of(operands, [&](Value v) {
        int64_t r = getRank(v);
        return r == 0 || r == maxRank;
      })) {
    return rewriter.notifyMatchFailure(
        op, "Operands must be of same rank or scalar.");
  }

  // Find result type, if on tensors.
  auto resultTy = dyn_cast_or_null<ShapedType>(
      typeConverter.convertType(op->getResultTypes().front()));

  // Check result type compatibility.
  if (!resultTy || !resultTy.hasRank() || resultTy.getRank() != maxRank ||
      !(resultTy.getElementType().isSignlessIntOrFloat() ||
        isa<ComplexType>(resultTy.getElementType()))) {
    return rewriter.notifyMatchFailure(
        op, "mismatched operand/result types or iterator count");
  }

  // All-scalar pointwise ops inside of linalg ops are processes by
  // ScalarHloToArithmeticPattern.
  if (maxRank == 0 && isInBodyOfLinalgOps(op)) return failure();

  return PointwiseConversionInfo{maxRank, resultTy};
}

/// Converts a HLO operation to a linalg.map op that contains the corresponding
/// scalar operations.
template <typename OpTy>
struct PointwiseToLinalgMapConverter final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto conversionInfo = checkOperandsAndResults(
        op, adaptor.getOperands(), *this->typeConverter, rewriter);
    if (failed(conversionInfo)) {
      return failure();
    }

    int64_t maxRank = conversionInfo->maxOperandRank;
    ShapedType resultTy = conversionInfo->resultType;
    Location loc = op.getLoc();

    // Find input/output values and types.
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    // Mapped inputs are cast to the same shape as the init tensor.
    // Values from scalar inputs are extracted and used directly in the block.
    SmallVector<Value> mappedInputs;
    SmallVector<Value> scalarInputs;
    for (Value input : adaptor.getOperands()) {
      if (getRank(input) == maxRank) {
        mappedInputs.push_back(coerceTensorShape(
            rewriter, loc, cast<TypedValue<ShapedType>>(input),
            emptyTensor.getType()));
        scalarInputs.push_back(nullptr);
      } else {
        scalarInputs.push_back(rewriter.create<tensor::ExtractOp>(loc, input));
      }
    }

    auto mapOp = rewriter.create<linalg::MapOp>(
        loc, mappedInputs, emptyTensor,
        [&](OpBuilder& b, Location loc, ValueRange args) {
          Value innerResult = stablehlo::StableHloOpToStdScalarOp::mapOp(
              op, getElementTypeOrSelf(emptyTensor),
              interleaveScalarAndBlockArgs(scalarInputs, args), &b);

          b.create<linalg::YieldOp>(loc, innerResult);
        },
        linalg::getPrunedAttributeList(op));

    rewriter.replaceOp(op, mapOp->getResults());
    return success();
  }
};

/// Converts a HLO operation to a linalg.generic op that contains the
/// corresponding scalar operations.
template <typename OpTy>
struct PointwiseToLinalgConverter final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult matchAndRewrite(
      OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto conversionInfo = checkOperandsAndResults(
        op, adaptor.getOperands(), *this->typeConverter, rewriter);
    if (failed(conversionInfo)) {
      return failure();
    }

    int64_t maxRank = conversionInfo->maxOperandRank;
    ShapedType resultTy = conversionInfo->resultType;
    Location loc = op.getLoc();

    // Find input/output values and types.
    ValueRange inputs = adaptor.getOperands();
    Value output =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    // Create indexing maps.
    AffineMap scalarMap = AffineMap::get(maxRank, 0, rewriter.getContext());
    AffineMap idMap = rewriter.getMultiDimIdentityMap(maxRank);
    SmallVector<AffineMap, 4> maps;
    for (Value v : inputs) maps.push_back(isScalar(v) ? scalarMap : idMap);
    maps.push_back(idMap);

    // Build `linalg.generic` op.
    bool failed = false;
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy ? resultTy : TypeRange{}, inputs, output, maps,
        getNParallelLoopsAttrs(maxRank),
        [&](OpBuilder& nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          Type innerResultTy = getElementTypeOrSelf(output);
          auto argvec = llvm::to_vector<2>(args.take_front(inputs.size()));
          Value semiring = preSparsify(op, argvec, innerResultTy, &rewriter);
          Value innerResult = mlir::stablehlo::StableHloOpToStdScalarOp::mapOp(
              op, innerResultTy, argvec, &rewriter);
          if (!innerResult) {
            failed = true;
          } else {
            innerResult = postSparsify(op, semiring, innerResult, &rewriter);
            nestedBuilder.create<linalg::YieldOp>(loc, innerResult);
          }
        },
        linalg::getPrunedAttributeList(op));
    if (failed) return failure();

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};
}  // namespace

namespace detail {
void populatePointwiseStableHloToLinalgConversionPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns, bool enablePrimitiveOps) {
  if (enablePrimitiveOps) {
    patterns
        ->add<PointwiseToLinalgMapConverter<stablehlo::AbsOp>,
              PointwiseToLinalgMapConverter<stablehlo::AddOp>,
              PointwiseToLinalgMapConverter<stablehlo::AndOp>,
              PointwiseToLinalgMapConverter<stablehlo::Atan2Op>,
              PointwiseToLinalgMapConverter<stablehlo::BitcastConvertOp>,
              PointwiseToLinalgMapConverter<stablehlo::CbrtOp>,
              PointwiseToLinalgMapConverter<stablehlo::CeilOp>,
              PointwiseToLinalgMapConverter<stablehlo::ClampOp>,
              PointwiseToLinalgMapConverter<stablehlo::ClzOp>,
              PointwiseToLinalgMapConverter<stablehlo::CompareOp>,
              PointwiseToLinalgMapConverter<stablehlo::ComplexOp>,
              PointwiseToLinalgMapConverter<stablehlo::ConvertOp>,
              PointwiseToLinalgMapConverter<stablehlo::CosineOp>,
              PointwiseToLinalgMapConverter<stablehlo::DivOp>,
              PointwiseToLinalgMapConverter<stablehlo::ExpOp>,
              PointwiseToLinalgMapConverter<stablehlo::Expm1Op>,
              PointwiseToLinalgMapConverter<stablehlo::FloorOp>,
              PointwiseToLinalgMapConverter<stablehlo::ImagOp>,
              PointwiseToLinalgMapConverter<stablehlo::IsFiniteOp>,
              PointwiseToLinalgMapConverter<stablehlo::Log1pOp>,
              PointwiseToLinalgMapConverter<stablehlo::LogOp>,
              PointwiseToLinalgMapConverter<stablehlo::LogisticOp>,
              PointwiseToLinalgMapConverter<stablehlo::MaxOp>,
              PointwiseToLinalgMapConverter<stablehlo::MinOp>,
              PointwiseToLinalgMapConverter<stablehlo::MulOp>,
              PointwiseToLinalgMapConverter<stablehlo::NegOp>,
              PointwiseToLinalgMapConverter<stablehlo::NotOp>,
              PointwiseToLinalgMapConverter<stablehlo::OrOp>,
              PointwiseToLinalgMapConverter<stablehlo::PopulationCountOp>,
              PointwiseToLinalgMapConverter<stablehlo::PowOp>,
              PointwiseToLinalgMapConverter<stablehlo::RealOp>,
              PointwiseToLinalgMapConverter<stablehlo::ReducePrecisionOp>,
              PointwiseToLinalgMapConverter<stablehlo::RemOp>,
              PointwiseToLinalgMapConverter<stablehlo::RoundNearestEvenOp>,
              PointwiseToLinalgMapConverter<stablehlo::RoundOp>,
              PointwiseToLinalgMapConverter<stablehlo::RsqrtOp>,
              PointwiseToLinalgMapConverter<stablehlo::SelectOp>,
              PointwiseToLinalgMapConverter<stablehlo::ShiftLeftOp>,
              PointwiseToLinalgMapConverter<stablehlo::ShiftRightArithmeticOp>,
              PointwiseToLinalgMapConverter<stablehlo::ShiftRightLogicalOp>,
              PointwiseToLinalgMapConverter<stablehlo::SignOp>,
              PointwiseToLinalgMapConverter<stablehlo::SineOp>,
              PointwiseToLinalgMapConverter<stablehlo::SqrtOp>,
              PointwiseToLinalgMapConverter<stablehlo::SubtractOp>,
              PointwiseToLinalgMapConverter<stablehlo::TanhOp>,
              PointwiseToLinalgMapConverter<stablehlo::XorOp>>(typeConverter,
                                                               context);
    return;
  }

  patterns->add<PointwiseToLinalgConverter<stablehlo::AbsOp>,
                PointwiseToLinalgConverter<stablehlo::AddOp>,
                PointwiseToLinalgConverter<stablehlo::AndOp>,
                PointwiseToLinalgConverter<stablehlo::Atan2Op>,
                PointwiseToLinalgConverter<stablehlo::BitcastConvertOp>,
                PointwiseToLinalgConverter<stablehlo::CbrtOp>,
                PointwiseToLinalgConverter<stablehlo::CeilOp>,
                PointwiseToLinalgConverter<stablehlo::ClampOp>,
                PointwiseToLinalgConverter<stablehlo::ClzOp>,
                PointwiseToLinalgConverter<stablehlo::CompareOp>,
                PointwiseToLinalgConverter<stablehlo::ComplexOp>,
                PointwiseToLinalgConverter<stablehlo::ConvertOp>,
                PointwiseToLinalgConverter<stablehlo::CosineOp>,
                PointwiseToLinalgConverter<stablehlo::DivOp>,
                PointwiseToLinalgConverter<stablehlo::ExpOp>,
                PointwiseToLinalgConverter<stablehlo::Expm1Op>,
                PointwiseToLinalgConverter<stablehlo::FloorOp>,
                PointwiseToLinalgConverter<stablehlo::ImagOp>,
                PointwiseToLinalgConverter<stablehlo::IsFiniteOp>,
                PointwiseToLinalgConverter<stablehlo::Log1pOp>,
                PointwiseToLinalgConverter<stablehlo::LogOp>,
                PointwiseToLinalgConverter<stablehlo::LogisticOp>,
                PointwiseToLinalgConverter<stablehlo::MaxOp>,
                PointwiseToLinalgConverter<stablehlo::MinOp>,
                PointwiseToLinalgConverter<stablehlo::MulOp>,
                PointwiseToLinalgConverter<stablehlo::NegOp>,
                PointwiseToLinalgConverter<stablehlo::NotOp>,
                PointwiseToLinalgConverter<stablehlo::OrOp>,
                PointwiseToLinalgConverter<stablehlo::PopulationCountOp>,
                PointwiseToLinalgConverter<stablehlo::PowOp>,
                PointwiseToLinalgConverter<stablehlo::RealOp>,
                PointwiseToLinalgConverter<stablehlo::ReducePrecisionOp>,
                PointwiseToLinalgConverter<stablehlo::RemOp>,
                PointwiseToLinalgConverter<stablehlo::RoundNearestEvenOp>,
                PointwiseToLinalgConverter<stablehlo::RoundOp>,
                PointwiseToLinalgConverter<stablehlo::RsqrtOp>,
                PointwiseToLinalgConverter<stablehlo::SelectOp>,
                PointwiseToLinalgConverter<stablehlo::ShiftLeftOp>,
                PointwiseToLinalgConverter<stablehlo::ShiftRightArithmeticOp>,
                PointwiseToLinalgConverter<stablehlo::ShiftRightLogicalOp>,
                PointwiseToLinalgConverter<stablehlo::SignOp>,
                PointwiseToLinalgConverter<stablehlo::SineOp>,
                PointwiseToLinalgConverter<stablehlo::SqrtOp>,
                PointwiseToLinalgConverter<stablehlo::SubtractOp>,
                PointwiseToLinalgConverter<stablehlo::TanhOp>,
                PointwiseToLinalgConverter<stablehlo::XorOp>>(typeConverter,
                                                              context);
}
}  // namespace detail
}  // namespace mlir::iree_compiler::stablehlo
