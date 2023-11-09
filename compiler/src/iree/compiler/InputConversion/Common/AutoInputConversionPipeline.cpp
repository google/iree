// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Dialect specific
#ifdef IREE_HAVE_STABLEHLO_INPUT
#include "iree/compiler/InputConversion/StableHLO/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#endif // IREE_HAVE_STABLEHLO_INPUT

namespace mlir::iree_compiler {
namespace {
struct AutoInputConversionPipelinePass final
    : AutoInputConversionPipelineBase<AutoInputConversionPipelinePass> {
  AutoInputConversionPipelinePass(
      const AutoInputConversionPipelineOptions &inputOptions,
      PipelineExtensions *pipelineExtensions)
      : pipelineExtensions(pipelineExtensions) {
    demoteI64ToI32 = inputOptions.demoteI64ToI32;
    demoteF64ToF32 = inputOptions.demoteF64ToF32;
    promoteBF16ToF32 = inputOptions.promoteBF16ToF32;
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;

  PipelineExtensions *pipelineExtensions = nullptr;
};

// All the features seen that should be handled during input conversion.
struct InputFeatures {
  // HLO features.
  bool hasStableHLO = false;
  // - XLA import features.
  bool hasTuples = false;
};

static void populateHloFeatures(Operation *op, InputFeatures &features) {
  if (features.hasTuples) {
    return;
  }

  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
    for (auto t : type.getResults()) {
      if (isa<TupleType>(t)) {
        features.hasTuples = true;
        return;
      }
    }
    for (auto t : type.getInputs()) {
      if (isa<TupleType>(t)) {
        features.hasTuples = true;
        return;
      }
    }
  }

  // Check for tuple operands or results.
  for (auto t : op->getOperandTypes()) {
    if (isa<TupleType>(t)) {
      features.hasTuples = true;
      return;
    }
  }
  for (auto t : op->getResultTypes()) {
    if (isa<TupleType>(t)) {
      features.hasTuples = true;
      return;
    }
  }
}

static void populateFeatures(Operation *op, const Dialect *chloDialect,
                             const Dialect *stablehloDialect,
                             InputFeatures &features) {
  Dialect *d = op->getDialect();
  if (d == stablehloDialect || d == chloDialect) {
    features.hasStableHLO = true;
    return populateHloFeatures(op, features);
  }
}

void AutoInputConversionPipelinePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  // Check if any plugin-provided pipeline extensions can convert dialects in
  // the module first.
  if (pipelineExtensions) {
    llvm::StringSet<> detectedTypeMnemonics;
    pipelineExtensions->populateDetectedCustomInputConversionTypes(
        module, detectedTypeMnemonics);

    if (detectedTypeMnemonics.getNumItems() > 1) {
      // TODO(scotttodd): handle multiple typeMnemonics (use all?)
      auto diag = module.emitError(
          "mixture of input types not yet implemented, set "
          "'--iree-input-type=[type]' explicitly instead of using 'auto' or "
          "audit the input program to understand why dialects are mixed");
      diag << " (detected:";
      for (auto &s : detectedTypeMnemonics) {
        diag << " '" << s.first() << "'";
      }
      diag << ")";
      return signalPassFailure();
    } else if (detectedTypeMnemonics.getNumItems() == 1) {
      auto typeMnemonic = detectedTypeMnemonics.begin()->getKey();
      OpPassManager passManager(module.getOperationName());
      bool foundExtension =
          pipelineExtensions->extendCustomInputConversionPassPipeline(
              passManager, typeMnemonic);
      if (!foundExtension) {
        // We expect that callers properly validate supported extensions and
        // that if a plugin advertises support, it actually provides it.
        module.emitError() << "custom input conversion for extension '"
                           << typeMnemonic << "' not found";
        return signalPassFailure();
      }
      if (failed(runPipeline(passManager, module))) {
        return signalPassFailure();
      }
      return;
    }
  }

  // No plugin-provided pipeline extensions were detected, try the built-in
  // dialect conversions.
  // TODO(scotttodd): Migrate these to compiler plugins?

  InputFeatures features;
  const Dialect *chloDialect = context->getLoadedDialect("chlo");
  const Dialect *stablehloDialect = context->getLoadedDialect("stablehlo");
  if (!chloDialect && !stablehloDialect) {
    return;
  }

  module.walk([&](Operation *op) {
    populateFeatures(op, chloDialect, stablehloDialect, features);
    return WalkResult::advance();
  });
  if (!features.hasStableHLO) {
    return;
  }

  OpPassManager pm(ModuleOp::getOperationName(),
                   OpPassManager::Nesting::Explicit);
#ifdef IREE_HAVE_STABLEHLO_INPUT
  if (features.hasStableHLO) {
    stablehlo::StableHloOptions options;
    options.demoteI64ToI32 = demoteI64ToI32;
    options.demoteF64ToF32 = demoteF64ToF32;
    options.promoteBF16ToF32 = promoteBF16ToF32;
    if (features.hasTuples) {
      stablehlo::buildStableHLOXLAInputConversionPassPipeline(pm, options);
    } else {
      stablehlo::buildStableHLOInputConversionPassPipeline(pm, options);
    }
  }
#endif // IREE_HAVE_STABLEHLO_INPUT

  if (failed(runPipeline(pm, module))) {
    signalPassFailure();
  }
}

void AutoInputConversionPipelinePass::getDependentDialects(
    DialectRegistry &registry) const {
  // Register dialects from all possible pipelines, as we do not statically know
  // which pipeline will be selected, while dialect registration happens before
  // we run any detection on the input.
  //
  // TODO(kuhar): Find a better registration mechanism so that we do not have to
  // build pipelines just to query dialects and discard them immediately after.
  auto appendPipelineDialects =
      [&registry](function_ref<void(OpPassManager &)> buildFn) {
        OpPassManager pm;
        buildFn(pm);
        pm.getDependentDialects(registry);
      };

#ifdef IREE_HAVE_STABLEHLO_INPUT
  auto appendStablehloPipelineDialects =
      [&registry](function_ref<void(OpPassManager &,
                                    const stablehlo::StableHloOptions &options)>
                      buildFn) {
        const stablehlo::StableHloOptions options;
        OpPassManager pm;
        buildFn(pm, options);
        pm.getDependentDialects(registry);
      };

  appendStablehloPipelineDialects(
      stablehlo::buildStableHLOInputConversionPassPipeline);
  appendStablehloPipelineDialects(
      stablehlo::buildStableHLOXLAInputConversionPassPipeline);
#endif // IREE_HAVE_STABLEHLO_INPUT

  if (pipelineExtensions) {
    pipelineExtensions->registerDialects(registry);
  }

  if (pipelineExtensions) {
    pipelineExtensions->registerDialects(registry);
  }

  (void)appendPipelineDialects;
}
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createAutoInputConversionPipelinePass() {
  AutoInputConversionPipelineOptions options;
  return std::make_unique<AutoInputConversionPipelinePass>(options, nullptr);
}

std::unique_ptr<OperationPass<ModuleOp>> createAutoInputConversionPipelinePass(
    const AutoInputConversionPipelineOptions &options,
    PipelineExtensions *pipelineExtensions) {
  return std::make_unique<AutoInputConversionPipelinePass>(options,
                                                           pipelineExtensions);
}

} // namespace mlir::iree_compiler
