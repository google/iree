// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_GPUTRANSFORMDIALECT_STRATEGIES_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace iree_compiler {

LogicalResult matchAndSetReductionTransformStrategy(func::FuncOp entryPoint,
                                                    linalg::LinalgOp op);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_LLVMGPU_GPUTRANSFORMDIALECT_STRATEGIES_H_
