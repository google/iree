// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/MemRefToVM/ConvertMemRefToVM.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/VM/Conversion/TargetOptions.h"
#include "iree/compiler/Dialect/VM/Conversion/TypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Returns true if the given `type` is a MemRef of rank 0 or 1.
static bool isRankZeroOrOneMemRef(Type type) {
  if (auto memrefType = type.dyn_cast<MemRefType>()) {
    return memrefType.hasRank() && memrefType.getRank() <= 1;
  }
  return false;
}

// Returns the offset, in bytes, of an index within a linearized dense buffer.
// Expects that the |memrefValue| has been linearized already.
static Value getBufferOffset(Location loc, Value memrefValue,
                             ValueRange indices, Type indexType,
                             ConversionPatternRewriter &rewriter) {
  auto memrefType = memrefValue.getType().cast<ShapedType>();
  if (memrefType.getRank() == 0) {
    // Rank 0 buffers (like memref<i32>) have only a single valid offset at 0.
    return rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
  }
  assert(memrefType.getRank() == 1 && "memrefs should have been flattened");

  // Element type byte length as the base.
  auto elementType = memrefType.getElementType();
  auto scalingExpr = getAffineBinaryOpExpr(
      AffineExprKind::Mul, getAffineSymbolExpr(0, rewriter.getContext()),
      getAffineConstantExpr(IREE::Util::getRoundedElementByteWidth(elementType),
                            rewriter.getContext()));

  // Rank 1 memrefs are just offset by their element width by the offset.
  Value offset = rewriter.createOrFold<AffineApplyOp>(
      loc, scalingExpr, ArrayRef<Value>{indices.front()});
  return rewriter.create<arith::IndexCastOp>(loc, indexType, offset);
}

class ConvertMemRefGlobalOp : public OpConversionPattern<memref::GlobalOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::GlobalOp globalOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(globalOp.getType())) {
      return rewriter.notifyMatchFailure(
          globalOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }

    // For mutable values we'd want to either have a RwdataOp or a global
    // !vm.buffer that we initialized with rodata.
    if (!globalOp.getConstant()) {
      return rewriter.notifyMatchFailure(
          globalOp, "mutable global memrefs not yet implemented");
    }

    auto rodataOp = rewriter.replaceOpWithNewOp<IREE::VM::RodataOp>(
        globalOp, globalOp.getSymName(),
        globalOp.getInitialValueAttr().cast<ElementsAttr>());
    rodataOp.setPrivate();
    return success();
  }
};

class ConvertMemRefGetGlobalOp
    : public OpConversionPattern<memref::GetGlobalOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::GetGlobalOp getOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(getOp.getResult().getType())) {
      return rewriter.notifyMatchFailure(
          getOp, "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    rewriter.replaceOpWithNewOp<IREE::VM::ConstRefRodataOp>(getOp,
                                                            getOp.getName());
    return success();
  }
};

// TODO(#9165): Support alignment for vm.buffer.alloc. So far we ignore the
// alignment attribute when lowering the op to VM dialect.
class ConvertMemRefAllocaOp : public OpConversionPattern<memref::AllocaOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::AllocaOp allocaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto type = allocaOp.getType().cast<ShapedType>();
    if (!type.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          allocaOp, "unable to create buffers for dynamic shapes");
    }

    int64_t length = type.getSizeInBits();
    length = llvm::divideCeil(length, 8);

    auto oldType = allocaOp.getType();
    auto newType = getTypeConverter()->convertType(oldType);
    Value size =
        rewriter.create<IREE::VM::ConstI64Op>(allocaOp.getLoc(), length);
    rewriter.replaceOpWithNewOp<IREE::VM::BufferAllocOp>(allocaOp, newType,
                                                         size);
    return success();
  }
};

class ConvertMemRefLoadOp : public OpConversionPattern<memref::LoadOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::LoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(loadOp.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          loadOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto oldType = loadOp.getResult().getType();
    auto newType = getTypeConverter()->convertType(oldType);
    auto byteOffset =
        getBufferOffset(loadOp.getLoc(), loadOp.getMemref(),
                        loadOp.getIndices(), rewriter.getI64Type(), rewriter);
    if (auto integerType = oldType.dyn_cast<IntegerType>()) {
      if (integerType.isInteger(1) || integerType.isInteger(8)) {
        if (integerType.isSigned() || integerType.isSignless()) {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI8SOp>(
              loadOp, newType, adaptor.getMemref(), byteOffset);
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI8UOp>(
              loadOp, newType, adaptor.getMemref(), byteOffset);
        }
      } else if (integerType.isInteger(16)) {
        if (integerType.isSigned() || integerType.isSignless()) {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI16SOp>(
              loadOp, newType, adaptor.getMemref(), byteOffset);
        } else {
          rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI16UOp>(
              loadOp, newType, adaptor.getMemref(), byteOffset);
        }
      } else if (integerType.isInteger(32)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI32Op>(
            loadOp, newType, adaptor.getMemref(), byteOffset);
      } else if (integerType.isInteger(64)) {
        rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadI64Op>(
            loadOp, newType, adaptor.getMemref(), byteOffset);
      } else {
        return rewriter.notifyMatchFailure(
            loadOp, "invalid integer buffer element type");
      }
    } else if (oldType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadF32Op>(
          loadOp, newType, adaptor.getMemref(), byteOffset);
    } else if (oldType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferLoadF64Op>(
          loadOp, newType, adaptor.getMemref(), byteOffset);
    } else {
      return rewriter.notifyMatchFailure(loadOp,
                                         "invalid float buffer element type");
    }
    return success();
  }
};

class ConvertMemRefStoreOp : public OpConversionPattern<memref::StoreOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      memref::StoreOp storeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isRankZeroOrOneMemRef(storeOp.getMemref().getType())) {
      return rewriter.notifyMatchFailure(
          storeOp,
          "only rank-0 and rank-1 memrefs are supported; flatten first");
    }
    auto oldType = storeOp.getValue().getType();
    auto byteOffset =
        getBufferOffset(storeOp.getLoc(), storeOp.getMemref(),
                        storeOp.getIndices(), rewriter.getI64Type(), rewriter);
    if (oldType.isInteger(1) || oldType.isInteger(8)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI8Op>(
          storeOp, adaptor.getMemref(), byteOffset, adaptor.getValue());
    } else if (oldType.isInteger(16)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI16Op>(
          storeOp, adaptor.getMemref(), byteOffset, adaptor.getValue());
    } else if (oldType.isInteger(32)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI32Op>(
          storeOp, adaptor.getMemref(), byteOffset, adaptor.getValue());
    } else if (oldType.isInteger(64)) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreI64Op>(
          storeOp, adaptor.getMemref(), byteOffset, adaptor.getValue());
    } else if (oldType.isF32()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreF32Op>(
          storeOp, adaptor.getMemref(), byteOffset, adaptor.getValue());
    } else if (oldType.isF64()) {
      rewriter.replaceOpWithNewOp<IREE::VM::BufferStoreF64Op>(
          storeOp, adaptor.getMemref(), byteOffset, adaptor.getValue());
    } else {
      return rewriter.notifyMatchFailure(storeOp,
                                         "invalid buffer element type");
    }
    return success();
  }
};

class ElideMemRefAssumeAlignmentOp
    : public OpConversionPattern<memref::AssumeAlignmentOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      memref::AssumeAlignmentOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

}  // namespace

void populateMemRefToVMPatterns(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter,
                                RewritePatternSet &patterns) {
  conversionTarget.addIllegalDialect<memref::MemRefDialect>();

  typeConverter.addConversion([&](MemRefType type) -> llvm::Optional<Type> {
    if (isRankZeroOrOneMemRef(type)) {
      return IREE::VM::RefType::get(
          IREE::VM::BufferType::get(type.getContext()));
    }
    return llvm::None;
  });

  // Unranked memrefs are emitted for library call integration when we just
  // need void* semantics. An unranked memref is basically just a (pointer,
  // memory-space, element-type).
  typeConverter.addConversion(
      [&](UnrankedMemRefType type) -> llvm::Optional<Type> {
        return IREE::VM::RefType::get(
            IREE::VM::BufferType::get(type.getContext()));
      });

  patterns.insert<FoldAsNoOp<bufferization::ToMemrefOp>,
                  FoldAsNoOp<memref::AssumeAlignmentOp>,
                  FoldAsNoOp<memref::CastOp>>(typeConverter, context);
  patterns.insert<ConvertMemRefGlobalOp, ConvertMemRefGetGlobalOp,
                  ConvertMemRefAllocaOp, ConvertMemRefLoadOp,
                  ConvertMemRefStoreOp, ElideMemRefAssumeAlignmentOp>(
      typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
