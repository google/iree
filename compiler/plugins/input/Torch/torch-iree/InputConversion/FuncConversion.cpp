// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torch-iree/InputConversion/Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "torch-iree/InputConversion/PassDetail.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

namespace mlir::iree_compiler::TorchInput {

namespace {

//===----------------------------------------------------------------------===//
// Func dialect -> Util patterns
//===----------------------------------------------------------------------===//

class FuncFuncOpPattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::FuncOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType srcFuncType = srcOp.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(
        srcOp.getNumArguments());

    // Convert function arguments.
    for (unsigned i = 0, e = srcFuncType.getNumInputs(); i < e; ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(
              i, srcFuncType.getInput(i), signatureConversion))) {
        return rewriter.notifyMatchFailure(srcOp, "argument failed to convert");
      }
    }

    // To support async, we add two fences (wait and signal) to the converted
    // function arguments. Conversion patterns in this file access them with
    // helper functions that know this.
    Type fenceType = rewriter.getType<IREE::HAL::FenceType>();
    signatureConversion.addInputs(ArrayRef<Type>{fenceType, fenceType});

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(getTypeConverter()->convertTypes(srcFuncType.getResults(),
                                                convertedResultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }

    // // Build tied operands index mapping results back to operands.
    // SmallVector<int64_t> tiedOperands;
    // bool anyTiedOperands = false;
    // for (unsigned i = 0; i < srcFuncType.getNumResults(); ++i) {
    //   auto tiedAttr =
    //       srcOp.getResultAttrOfType<IntegerAttr>(i, "iree.abi.tied");
    //   if (tiedAttr) {
    //     tiedOperands.push_back(tiedAttr.getInt());
    //   } else {
    //     tiedOperands.push_back(-1);
    //   }
    // }
    // auto tiedOperandsAttr = anyTiedOperands
    //                             ? rewriter.getIndexArrayAttr(tiedOperands)
    //                             : ArrayAttr{};

    // Create new function with converted argument and result types.
    // Note that attributes are dropped. Consider preserving some if needed.
    auto newFuncType = mlir::FunctionType::get(
        srcOp.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);
    auto newFuncOp = rewriter.create<IREE::Util::FuncOp>(
        srcOp.getLoc(), srcOp.getName(), newFuncType,
        /*tiedOperandsAttr=*/ArrayAttr{});
    newFuncOp.setSymVisibilityAttr(srcOp.getSymVisibilityAttr());
    rewriter.inlineRegionBefore(srcOp.getBody(), newFuncOp.getFunctionBody(),
                                newFuncOp.end());

    // Handle defacto attrs to specialized ones.
    if (srcOp->hasAttr("noinline")) {
      newFuncOp.setInliningPolicyAttr(
          rewriter.getAttr<IREE::Util::InlineNeverAttr>());
    }

    // Allowlist of function attributes to retain when importing funcs.
    constexpr const char *kRetainedAttributes[] = {
        "iree.reflection",
        "vm.fallback",
        "vm.signature",
        "vm.version",
    };
    auto retainedAttributes = ArrayRef<const char *>(
        kRetainedAttributes,
        sizeof(kRetainedAttributes) / sizeof(kRetainedAttributes[0]));
    for (auto retainAttrName : retainedAttributes) {
      StringRef attrName(retainAttrName);
      Attribute attr = srcOp->getAttr(attrName);
      if (attr)
        newFuncOp->setAttr(attrName, attr);
    }

    // Copy all arg/result attrs. We could filter these.
    if (auto argAttrs = srcOp.getAllArgAttrs()) {
      newFuncOp.setAllArgAttrs(argAttrs);
    }
    if (auto resultAttrs = srcOp.getAllResultAttrs()) {
      newFuncOp.setAllResultAttrs(resultAttrs);
    }

    // Tell the rewriter to convert the region signature.
    const TypeConverter &typeConverter = *getTypeConverter();
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           typeConverter,
                                           &signatureConversion))) {
      return failure();
    }

    rewriter.eraseOp(srcOp);
    return success();
  }
};

class FuncCallOpPattern : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern<func::CallOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type, 1> resultTypes;
    if (failed(getTypeConverter()->convertTypes(srcOp.getResultTypes(),
                                                resultTypes))) {
      return rewriter.notifyMatchFailure(srcOp, "results failed to convert");
    }
    auto tiedOperandsAttr =
        srcOp->getAttrOfType<ArrayAttr>("iree.abi.tied_operands");
    rewriter.replaceOpWithNewOp<IREE::Util::CallOp>(
        srcOp, resultTypes, srcOp.getCallee(), adaptor.getOperands(),
        tiedOperandsAttr);
    return success();
  }
};

class FuncReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::Util::ReturnOp>(srcOp,
                                                      adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Immutable tensor type conversion.
//===----------------------------------------------------------------------===//

std::optional<std::pair<Value, Value>> getParentWaitSignalFences(Value value) {
  auto parentFuncOp =
      dyn_cast<IREE::Util::FuncOp>(value.getParentRegion()->getParentOp());
  if (!parentFuncOp) {
    return {};
  }
  Block *entryBlock = &parentFuncOp.front();
  auto numArguments = entryBlock->getNumArguments();
  Value coarseWaitFence = entryBlock->getArgument(numArguments - 2);
  Value coarseSignalFence = entryBlock->getArgument(numArguments - 1);
  return std::make_pair(coarseWaitFence, coarseSignalFence);
}

void setupImmutableTensorConversion(ConversionTarget &target,
                                    RewritePatternSet &patterns,
                                    TypeConverter &typeConverter) {
  // target.addIllegalOp<Torch::CopyToValueTensorOp>();
  //  target.addIllegalOp<Torch::CopyToNonValueTensorOp>();
  //  target.addIllegalOp<Torch::OverwriteTensorContentsOp>();
  //  patterns.insert<CopyEntryArgToValueTensorPattern>(typeConverter,
  //                                                    patterns.getContext());

  typeConverter.addConversion(
      [](Torch::ValueTensorType type) -> std::optional<Type> {
        return IREE::HAL::BufferViewType::get(type.getContext());
      });
  auto sourceMaterialization = [](OpBuilder &builder,
                                  Torch::ValueTensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    Value source = inputs.front();
    auto waitSignalFences = getParentWaitSignalFences(source);
    if (!waitSignalFences)
      return {};
    Value waitFence = waitSignalFences->first;

    TensorType builtinTensorType = type.toBuiltinTensor();
    Value importedTensor = builder.create<IREE::HAL::TensorImportOp>(
        loc, builtinTensorType, source, TypeAttr::get(builtinTensorType),
        waitFence,
        /*name=*/StringAttr());
    return builder.create<TorchConversion::FromBuiltinTensorOp>(loc, type,
                                                                importedTensor);
  };
  auto targetMaterialization = [](OpBuilder &builder,
                                  IREE::HAL::BufferViewType type,
                                  ValueRange inputs, Location loc) -> Value {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    castOp->setAttr("torch.export.immutable_tensor", builder.getUnitAttr());
    return castOp.getResult(0);
  };
  typeConverter.addArgumentMaterialization(sourceMaterialization);
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addTargetMaterialization(targetMaterialization);
}

//===----------------------------------------------------------------------===//
// Mutable tensor type conversion.
// Here we rely on the characteristic that at the torch level, conversion to
// and from the value domain is only legal at certain well defined points in
// the program (currently at graph edges but potentially in the future at
// various control flow points). These conversions are modeled by:
//   * torch.copy.to_vtensor: Copy from a mutable tensor (torch.tensor) to
//     an immutable value (torch.vtensor).
//   * torch.copy.to_tensor: Allocates a new mutable tensor and initializes it
//     with the value of the given immutable tensor.
//   * torch.overwrite.tensor.contents: Updates the contents of a mutable
//     tensor from a given immutable tensor.
//
// Note that when importing from Torch, these ops cannot just be added at will,
// and they are only created as a result of structural conversions. Therefore,
// we can rely on these invariants and assume that usage outside of this is an
// invalid program.
//
// Conversion Mechanic:
// --------------------
//   * torch.copy.to_vtensor is handled directly as a conversion pattern because
//     it is pure (with the only constraint that it has to happen after the
//     function level wait fence).
//   * The mutation ops are not handled during conversion, but we emit an
//     unrealized_conversion_cast from the block-arg !hal.buffer_view to
//     unhandled ops.
//   * Function level post processing is performed to clean up sub-graphs of
//     mutation. The cast op has an attribute `torch.coarse_signal_mutation`
//     added to make it easy to perform post-processing.
//
// Post Processing:
// ----------------
// At the function level, all mutable argument mutations must be coellesced
// into a single barrier to signal the signal fence (as a default, this is
// conservatively safe, but for more complicated programs, we may want to
// enable explicit tieing of the mutation to a signal fence in order to
// enable more pipelining).
//
// We structurally have a very narrow definition of how such mutable arguments
// can be used:
//   * Consumed by a `torch.overwrite.tensor.contents`.
//   * Returned from the function.
//
// This results in a small number of subgraphs that can exist for users of
// a mutable argument. We detect all such subgraphs by walking the module for
// unrealized_conversion_cast operators with the
// `torch.coarse_signal_mutation` attribute, added during type
// materialization for any unrecognized (presumed mutation) ops operating on
// the argument. If no such casts are present, then the function does not
// actually mutate its argument and no action is needed.
//===----------------------------------------------------------------------===//

class CopyEntryArgToValueTensorPattern
    : public OpConversionPattern<Torch::CopyToValueTensorOp> {
  using OpConversionPattern<Torch::CopyToValueTensorOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Torch::CopyToValueTensorOp srcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Access the type converted buffer view.
    auto bufferView = dyn_cast<BlockArgument>(adaptor.getOperand());
    if (!bufferView) {
      return rewriter.notifyMatchFailure(srcOp,
                                         "not produced by a BlockArgument");
    }
    Block *producerBlock = bufferView.getOwner();
    if (!isa<IREE::Util::FuncOp>(producerBlock->getParentOp())) {
      return rewriter.notifyMatchFailure(
          srcOp, "not produced directly by parent function");
    }
    if (!producerBlock->isEntryBlock()) {
      return rewriter.notifyMatchFailure(srcOp, "not produced by entry block");
    }

    auto resultVTensorType =
        cast<Torch::ValueTensorType>(srcOp.getResult().getType());
    auto ireeTensorType = resultVTensorType.toBuiltinTensor();

    // The producer block will always end in {wait_fence, signal_fence}.
    Value waitFence =
        producerBlock->getArgument(producerBlock->getNumArguments() - 2);
    Value imported = rewriter.create<IREE::HAL::TensorImportOp>(
        srcOp.getLoc(), ireeTensorType, bufferView,
        /*target_encoding=*/TypeAttr::get(ireeTensorType),
        /*wait_fence*/ waitFence,
        /*name=*/StringAttr());

    rewriter.replaceOpWithNewOp<TorchConversion::FromBuiltinTensorOp>(
        srcOp, resultVTensorType, imported);
    return success();
  }
};

Value convertToBuiltinTensor(OpBuilder &builder, Value possibleTorchTensor) {
  Type ty = possibleTorchTensor.getType();
  if (isa<TensorType>(ty))
    return possibleTorchTensor;

  Torch::ValueTensorType vtensorType = cast<Torch::ValueTensorType>(ty);
  TensorType builtinTy = vtensorType.toBuiltinTensor();
  return builder.create<TorchConversion::ToBuiltinTensorOp>(
      possibleTorchTensor.getLoc(), builtinTy, possibleTorchTensor);
}

void setupMutableTensorConversion(ConversionTarget &target,
                                  RewritePatternSet &patterns,
                                  TypeConverter &typeConverter) {
  target.addIllegalOp<Torch::CopyToValueTensorOp>();
  // target.addIllegalOp<Torch::CopyToNonValueTensorOp>();
  // target.addIllegalOp<Torch::OverwriteTensorContentsOp>();
  patterns.insert<CopyEntryArgToValueTensorPattern>(typeConverter,
                                                    patterns.getContext());

  typeConverter.addConversion(
      [](Torch::NonValueTensorType type) -> std::optional<Type> {
        return IREE::HAL::BufferViewType::get(type.getContext());
      });
  auto materialization = [](OpBuilder &builder, Torch::NonValueTensorType type,
                            ValueRange inputs, Location loc) -> Value {
    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    castOp->setAttr("torch.coarse_signal_mutation", builder.getUnitAttr());
    return castOp.getResult(0);
  };
  typeConverter.addSourceMaterialization(materialization);
  typeConverter.addArgumentMaterialization(materialization);
}

LogicalResult postProcessFunctionMutation(IREE::Util::FuncOp funcOp) {
  // Walk to find arguments subject to some mutation.
  SmallPtrSet<Value, 4> coarseSignalArgs;
  SmallPtrSet<Value, 4> coarseSignalExportTensors;
  SmallVector<IREE::Util::ReturnOp> returnOps;
  funcOp.walk([&](Operation *op) {
    OpBuilder builder(op);
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (castOp->hasAttr("torch.coarse_signal_mutation")) {
        Value source = castOp.getOperand(0);
        Value target = castOp.getResult(0);
        coarseSignalArgs.insert(source);

        // Makes the IR invalid for any consumers, but that is fine. If we do
        // not transform the consumers, we will fail anyway, and eliminating the
        // cast here makes everything simpler.
        target.replaceAllUsesWith(source);
        castOp->erase();
      } else if (castOp->hasAttr("torch.export.immutable_tensor")) {
        // This cast is placed when going from an immutable vtensor to a
        // BufferView. We do this by exporting and including in the coarse
        // barrier.
        Value source = castOp.getOperand(0);
        Value target = castOp.getResult(0);

        source = convertToBuiltinTensor(builder, source);
        coarseSignalExportTensors.insert(source);

        // Makes the IR invalid for any consumers, but that is fine. If we do
        // not transform the consumers, we will fail anyway, and eliminating the
        // cast here makes everything simpler.
        target.replaceAllUsesWith(source);
        castOp->erase();
      }
    } else if (auto returnOp = dyn_cast<IREE::Util::ReturnOp>(op)) {
      returnOps.push_back(returnOp);
    }
  });

  bool hasCoarseSignalMutation = !coarseSignalArgs.empty();
  Block *entryBlock = &funcOp.front();
  Value coarseSignalFence =
      entryBlock->getArgument(entryBlock->getNumArguments() - 1);

  // Enforce some structural conditions. We presently have no way to generate
  // programs that violate these, but maybe someday. Then a more advanced
  // algorithm will be needed.
  if (hasCoarseSignalMutation && returnOps.size() > 1) {
    auto diag =
        emitError(funcOp.getLoc())
        << "functions with coarse signal mutation must only have a single exit";
    for (auto returnOp : returnOps) {
      diag.attachNote(returnOp.getLoc()) << "illegal multiple return";
    }
    return diag;
  }

  // Assemble the postamble, consisting of a signaling barrier and mutating
  // exports.
  IREE::Util::ReturnOp returnOp = returnOps.front();
  OpBuilder builder(returnOp);
  IRMapping returnMapping;
  SmallVector<Operation *> eraseOps;
  // Tensors that need to be joined in a barrier on the coarse signal fence
  // and exported.
  SmallVector<Value> barrierSources;
  SmallVector<Value> tiedTargetStorages;

  // Process each function arg that participates in coarse mutation signaling.
  for (Value bufferArg : coarseSignalArgs) {
    Value overwriteTensor;
    for (OpOperand &use : bufferArg.getUses()) {
      Operation *useOp = use.getOwner();

      // Legal uses that require no further action.
      if (isa<IREE::HAL::TensorImportOp>(useOp))
        continue;

      // Other uses.
      if (auto overwrite = dyn_cast<Torch::OverwriteTensorContentsOp>(useOp)) {
        if (overwriteTensor) {
          return emitError(useOp->getLoc())
                 << "unsupported multiple updates on coarse signaling mutable "
                    "tensor";
        }
        overwriteTensor = convertToBuiltinTensor(builder, overwrite.getValue());
        barrierSources.push_back(overwriteTensor);
        tiedTargetStorages.push_back(bufferArg);
        eraseOps.push_back(overwrite);
      } else {
        return emitError(useOp->getLoc())
               << "unsupported operation on coarse signaling mutable tensor";
      }
    }
  }

  // Process each unbacked tensor that must be exported and synchronized.
  for (Value unbackedTensor : coarseSignalExportTensors) {
    barrierSources.push_back(unbackedTensor);
    tiedTargetStorages.push_back(nullptr);
  }

  // Generate barriers and exports.
  auto barrierOp = builder.create<IREE::HAL::TensorBarrierOp>(
      funcOp.getLoc(), barrierSources, coarseSignalFence);
  for (auto [sourceTensor, tiedTargetStorage, barrierTensor] : llvm::zip_equal(
           barrierSources, tiedTargetStorages, barrierOp.getResults())) {
    Value exportedValue = builder.create<IREE::HAL::TensorExportOp>(
        sourceTensor.getLoc(), builder.getType<IREE::HAL::BufferViewType>(),
        barrierTensor, TypeAttr::get(sourceTensor.getType()), tiedTargetStorage,
        StringAttr());
    if (tiedTargetStorage) {
      // We must not drop exports of mutation, so hold on to it.
      builder.create<IREE::Util::OptimizationBarrierOp>(sourceTensor.getLoc(),
                                                        exportedValue);
    }
    returnMapping.map(sourceTensor, exportedValue);
  }

  // Update the return op with mapped values.
  for (OpOperand &operand : returnOp->getOpOperands()) {
    operand.assign(returnMapping.lookupOrDefault(operand.get()));
  }

  // Clean up.
  for (Operation *op : eraseOps) {
    op->erase();
  }

  return success();
}

} // namespace

struct FuncConversionPass : public FuncConversionBase<FuncConversionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    registry.insert<IREE::Util::UtilDialect>();
    registry.insert<TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addLegalDialect<IREE::Util::UtilDialect>();
    target.markUnknownOpDynamicallyLegal([&](Operation *op) { return true; });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    // Func conversion.
    target.addDynamicallyLegalDialect<func::FuncDialect>(
        [&](Operation *op) -> std::optional<bool> {
          // Allow the func dialect within nested modules but not in the
          // top-level one that represents the host program.
          return op->getParentOfType<mlir::ModuleOp>() != getOperation();
        });

    patterns.insert<FuncFuncOpPattern>(typeConverter, context);
    patterns.insert<FuncCallOpPattern>(typeConverter, context);
    patterns.insert<FuncReturnOpPattern>(typeConverter, context);

    // Mutable tensor at the graph edges conversion.
    setupImmutableTensorConversion(target, patterns, typeConverter);
    setupMutableTensorConversion(target, patterns, typeConverter);
    // patterns.insert<CopyEntryArgToValueTensorPattern>(typeConverter,
    // context); patterns.insert<OverwriteEntryArgTensorContents>(typeConverter,
    // context);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // Post-process functions.
    for (auto funcOp : module.getOps<IREE::Util::FuncOp>()) {
      if (funcOp.isExternal())
        continue;
      if (failed(postProcessFunctionMutation(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createFuncConversionPass() {
  return std::make_unique<FuncConversionPass>();
}

} // namespace mlir::iree_compiler::TorchInput
