// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

std::unique_ptr<OperationPass<void>> createApplyPatternsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createCombineInitializersPass();
std::unique_ptr<OperationPass<void>> createDropCompilerHintsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createHoistIntoGlobalsPass();
std::unique_ptr<OperationPass<void>> createSimplifyGlobalAccessesPass();

// Test passes.
std::unique_ptr<OperationPass<void>> createTestFloatRangeAnalysis();

// Register all Passes
inline void registerTransformPasses() {
  createApplyPatternsPass();
  createCombineInitializersPass();
  createDropCompilerHintsPass();
  createFoldGlobalsPass();
  createFuseGlobalsPass();
  createHoistIntoGlobalsPass();
  createSimplifyGlobalAccessesPass();
  createTestFloatRangeAnalysis();
}

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
