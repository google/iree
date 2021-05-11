// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"
#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCTypeConverter.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

using RegisterAllocationCache = DenseMap<Operation *, RegisterAllocation>;
using ValueLivenessCache = DenseMap<Operation *, ValueLiveness>;

void populateVMToEmitCPatterns(MLIRContext *context,
                               IREE::VM::EmitCTypeConverter &typeConverter,
                               OwningRewritePatternList &patterns,
                               RegisterAllocationCache &registerAllocationCache,
                               ValueLivenessCache &valueLivenessCache);

namespace IREE {
namespace VM {

std::unique_ptr<OperationPass<IREE::VM::ModuleOp>> createConvertVMToEmitCPass();

}  // namespace VM
}  // namespace IREE

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_VMTOEMITC_CONVERTVMTOEMITC_H_
