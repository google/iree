// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Utils/TransformDialectUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-user-configs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

llvm::cl::opt<std::string> clCodegenTransformDialectLibraryFileName(
    "iree-codegen-transform-dialect-library",
    llvm::cl::desc(
        "File path to a module containing a library of transform dialect"
        "strategies. Can be suffixed with the name of a transform sequence"
        "within the library to run as preprocessing per executable variant."
        "This is specified as <file-path>@<sequence-name>. If not specified,"
        "this will default to `__kernel_config`."),
    llvm::cl::init(""));

namespace {

static const char kTranslationInfoAttrName[] = "translation_info";

struct MaterializeUserConfigsPass
    : public MaterializeUserConfigsBase<MaterializeUserConfigsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {

      // Parse the file path and kernel config strategy from flags. There are
      // two possible usage flows for transform dialect libraries.
      //   1. Use `__kernel_config` to match and annotate variants with the
      //      strategy to use. This could either be a transform dialect strategy
      //      or any other IREE codegen pipeline.
      //
      //   2. Use the configuration strategy to do codegen directly. At the end
      //   of
      //      the strategy, the variant needs to be annotated with
      //      "translation_info" = #iree_codegen.translation_info<None>

      auto parsedFilePath = parseTransformLibraryFileNameAndEntrySequence(
          clCodegenTransformDialectLibraryFileName);
      if (failed(parsedFilePath)) {
        funcOp.emitError() << "Could not parse transform dialect library";
        return signalPassFailure();
      }
      std::optional<std::string> libraryFileName = parsedFilePath->first;
      std::string entrySequenceName = parsedFilePath->second.has_value()
                                          ? parsedFilePath->second.value()
                                          : "__kernel_config";

      LDBG("MaterializeUserConfigsPass on function: " << funcOp);
      std::optional<ModuleOp> transformLibrary = std::nullopt;
      if (libraryFileName.has_value()) {
        auto dialect =
            context->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
        auto maybeTransformLibrary =
            dialect->getOrLoadTransformLibraryModule(libraryFileName.value());
        if (failed(maybeTransformLibrary)) {
          funcOp.emitError() << "failed to load transform library module: "
                             << libraryFileName.value();
          return signalPassFailure();
        }
        transformLibrary = *maybeTransformLibrary;
        LDBG("--found transform library @" << libraryFileName.value());

        auto runResult = runTransformConfigurationStrategy(
            funcOp, entrySequenceName, *transformLibrary);
        if (runResult == StrategyRunResult::NotFound) {
          funcOp.emitError() << "transform kernel config strategy `"
                             << entrySequenceName << " not found";
          return signalPassFailure();
        } else if (runResult == StrategyRunResult::Failed) {
          funcOp.emitError() << "transform kernel config strategy `"
                             << entrySequenceName << "` failed to apply";
          return signalPassFailure();
        }
      }

      /// Nothing to do if the export already has a config.
      IREE::Codegen::TranslationInfoAttr translationInfo =
          getTranslationInfo(funcOp);
      if (translationInfo) {
        return;
      }

      /// First, apply all user configs.
      auto res = funcOp.walk([&](Operation *op) {
        if (auto compilationInfo = getCompilationInfo(op)) {
          if (failed(setUserConfig(funcOp, op, compilationInfo))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      if (res.wasInterrupted()) {
        funcOp.emitOpError("error in setting user configuration");
        return signalPassFailure();
      }

      translationInfo = getTranslationInfo(funcOp);
      LDBG("--guaranteed unique translationInfo: " << translationInfo);
      /// We only need to resolve symbols for transform dialect based
      /// strategies.
      if (!translationInfo ||
          translationInfo.getDispatchLoweringPassPipeline() !=
              IREE::Codegen::DispatchLoweringPassPipeline::
                  TransformDialectCodegen) {
        return;
      }

      std::optional<SymbolRefAttr> strategyName =
          translationInfo.getCodegenSpec();
      if (!strategyName || *strategyName == SymbolRefAttr()) {
        return;
      }

      /// If we have a symbol, verify the existence of the symbol within the
      /// transform library.
      StringRef entryPoint = strategyName->getLeafReference();
      if (!transformLibrary || !(*transformLibrary) ||
          !transform::detail::findTransformEntryPoint(funcOp, *transformLibrary,
                                                      entryPoint)) {
        funcOp.emitOpError("failed to find transform strategy symbol");
      }
    }
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeUserConfigsPass() {
  return std::make_unique<MaterializeUserConfigsPass>();
}

} // namespace mlir::iree_compiler
