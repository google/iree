// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/Analysis/DeviceAnalysis.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::GlobalOptimization {

// TODO: Remove the flag once the codegen can handle the late materialization
// path. This is mainly for testing.
static llvm::cl::opt<bool> clEnableExperimentalRocmDataTiling(
    "iree-global-opt-experimental-rocm-data-tiling",
    llvm::cl::desc("Enables data-tiling materialization for rocm backends "
                   "(experimental)."),
    llvm::cl::init(false));

#define GEN_PASS_DEF_MATERIALIZEHOMOGENEOUSENCODINGSPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using FunctionLikeNest =
    MultiOpNest<IREE::Util::InitializerOp, IREE::Util::FuncOp>;

namespace {
struct MaterializeHomogeneousEncodingsPass final
    : impl::MaterializeHomogeneousEncodingsPassBase<
          MaterializeHomogeneousEncodingsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect, tensor::TensorDialect,
                    IREE::Codegen::IREECodegenDialect>();
  }

  void addNopPipeline(OpPassManager &passManager) {
    FunctionLikeNest(passManager).addPass(createMaterializeEncodingIntoNopPass);
    FunctionLikeNest(passManager).addPass(createCanonicalizerPass);
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    IREE::HAL::DeviceAnalysis deviceAnalysis(moduleOp);
    if (failed(deviceAnalysis.run()))
      return signalPassFailure();

    SetVector<IREE::HAL::ExecutableTargetAttr> executableTargets;
    deviceAnalysis.gatherAllExecutableTargets(executableTargets);
    OpPassManager passManager(moduleOp.getOperationName());
    if (executableTargets.size() != 1) {
      addNopPipeline(passManager);
      if (failed(runPipeline(passManager, moduleOp))) {
        return signalPassFailure();
      }
      return;
    }

    // TODO: vmvx has its own logic about supporting dynamic tile
    // sizes. It is not fully integrated into the pipeline, so we remain the
    // materialization to the end.
    IREE::HAL::ExecutableTargetAttr executableTarget = executableTargets[0];
    if (executableTarget.getBackend() == "vmvx") {
      return;
    }

    // Only llvm-cpu and rocm backends handle encodings for now, others just go
    // with nop.
    if (executableTarget.getBackend() == "llvm-cpu") {
      passManager.addPass(createMaterializeHostEncodingPass());
    } else if (clEnableExperimentalRocmDataTiling &&
               executableTarget.getBackend() == "rocm") {
      passManager.addPass(createMaterializeHostEncodingPass());
      FunctionLikeNest(passManager).addPass([&]() {
        return createDecomposePackUnPackOpsPass(
            DecomposePackUnPackOpsPassOptions{/*tileOuterToOne=*/false,
                                              /*useOnlyReshapes=*/true});
      });
    } else {
      addNopPipeline(passManager);
    }
    if (failed(runPipeline(passManager, moduleOp))) {
      return signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
