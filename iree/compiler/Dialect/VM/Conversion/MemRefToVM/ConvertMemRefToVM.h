// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_CONVERSION_MEMREFTOVM_CONVERTMEMREFTOVM_H_
#define IREE_COMPILER_DIALECT_VM_CONVERSION_MEMREFTOVM_CONVERTMEMREFTOVM_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

// Appends memref dialect to vm dialect patterns to the given pattern list.
void populateMemRefToVMPatterns(MLIRContext *context,
                                ConversionTarget &conversionTarget,
                                TypeConverter &typeConverter,
                                RewritePatternSet &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_VM_CONVERSION_MEMREFTOVM_CONVERTMEMREFTOVM_H_
