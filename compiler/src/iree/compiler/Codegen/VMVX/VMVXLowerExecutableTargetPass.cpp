// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/VMVX/PassDetail.h"
#include "iree/compiler/Codegen/VMVX/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir {
namespace iree_compiler {

namespace {

/// Lowers an hal.executable.variant operation to scalar/native-vector code.
class VMVXLowerExecutableTargetPass
    : public VMVXLowerExecutableTargetBase<VMVXLowerExecutableTargetPass> {
public:
  VMVXLowerExecutableTargetPass() = default;
  VMVXLowerExecutableTargetPass(const VMVXLowerExecutableTargetPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<IREE::HAL::HALDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    bufferization::BufferizationDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  void runOnOperation() override;
};
} // namespace

void VMVXLowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();
  if (!moduleOp) {
    getOperation()->emitError(
        "Expected a variantOp root with an inner ModuleOp");
    return signalPassFailure();
  }

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  // There might be multiple entry points in the module. Currently, all of
  // them need to have the same translation info. This should already be
  // verified when the strategies are set, but we still need to retrieve the
  // correct translation info.
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
  for (auto &it : exportOps) {
    auto exportOp = it.second;
    if (IREE::Codegen::TranslationInfoAttr currTranslationInfo =
            getTranslationInfo(exportOp)) {
      if (translationInfo) {
        if (currTranslationInfo != translationInfo.value()) {
          moduleOp.emitOpError(
              "unhandled compilation of entry point functions with different "
              "translation info");
          return signalPassFailure();
        }
      } else {
        translationInfo = currTranslationInfo;
      }
    }
  }

  if (translationInfo.has_value()) {
    auto target = variantOp.getTarget();
    bool enableMicrokernels = hasMicrokernels(target);
    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
    // No pipleline specified, nothing to do.
    case IREE::Codegen::DispatchLoweringPassPipeline::None:
      return;
    case IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault:
      addVMVXDefaultPassPipeline(executableLoweringPipeline,
                                 enableMicrokernels);
      break;
    default:
      moduleOp.emitOpError("Unsupported pipeline on CPU target.");
      return signalPassFailure();
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createVMVXLowerExecutableTargetPass() {
  return std::make_unique<VMVXLowerExecutableTargetPass>();
}

} // namespace iree_compiler
} // namespace mlir
