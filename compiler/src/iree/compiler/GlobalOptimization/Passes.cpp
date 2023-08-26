// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions) {
  // Preprocessing passes to get the program into a canonical state.
  FunctionLikeNest(mainPassManager)
      .addPass(IREE::Flow::createRemoveZeroExtentTensorsPass)
      .addPass(IREE::Flow::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(IREE::Flow::createConvert1X1FilterConv2DToMatmulPass);
  mainPassManager.addPass(IREE::Flow::createEraseUnusedLinalgOperands());

  // Expand tensor shapes into SSA values and optimize the whole program.
  // The more we are able to equate shape dimensions at this level the
  // better our fusions will be.
  FunctionLikeNest(mainPassManager)
      .addPass(IREE::Flow::createTopLevelSCFToCFGPass);
  mainPassManager.addPass(IREE::Flow::createExpandTensorShapesPass());

  FunctionLikeNest(mainPassManager)
      // Preprocess the input to a form more amenable for fusion
      // - Convert all elementwise ops to Linalg
      // - Remove unit-extent dimensions.
      .addPass(mlir::createConvertElementwiseToLinalgPass)
      .addPass(IREE::Flow::createGeneralizeLinalgNamedOpsPass)
      .addPass(IREE::Flow::createFuseDequantizationMatmulPass)
      .addPass(IREE::Flow::createFoldUnitExtentDimsPass)
      .addPass(IREE::Flow::createRaiseSpecialOps)
      .addPass(IREE::Flow::createInterchangeGenericOpsPass);

  OpPassManager pipeline(ModuleOp::getOperationName());
  FunctionLikeNest(pipeline)
      // Simplify util.global accesses early on; this can help with dispatch
      // region formation as redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Module level cleanup and canonicalization of util.global (and other util
  // ops).
  pipeline.addPass(IREE::Util::createApplyPatternsPass());
  pipeline.addPass(IREE::Util::createFoldGlobalsPass());
  pipeline.addPass(IREE::Util::createIPOPass());

  if (transformOptions.constExprHoisting) {
    pipeline.addPass(IREE::Util::createHoistIntoGlobalsPass());
  }

  if (transformOptions.buildConstEvalPassPipeline) {
    transformOptions.buildConstEvalPassPipeline(pipeline);
  }

  if (transformOptions.numericPrecisionReduction) {
    pipeline.addPass(IREE::Flow::createInferNumericNarrowingPass());
    pipeline.addPass(IREE::Flow::createOptimizeNumericsPass());
    pipeline.addPass(IREE::Flow::createCleanupNumericNarrowingPass());
  }

  FunctionLikeNest(pipeline)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Add the whole fixed point iterator.
  mainPassManager.addPass(
      IREE::Util::createFixedPointIteratorPass(std::move(pipeline)));
}

void registerGlobalOptimizationPipeline() {
  PassPipelineRegistration<TransformOptions>
      globalOptimizationTransformPassPipeline(
          "iree-global-optimization-transformation-pipeline",
          "Runs the IREE global optimization transformation pipeline",
          [](OpPassManager &passManager,
             const TransformOptions &transformOptions) {
            buildGlobalOptimizationPassPipeline(passManager, transformOptions);
          });
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
