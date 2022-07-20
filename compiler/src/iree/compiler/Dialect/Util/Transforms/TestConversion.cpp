// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Conversion/ConversionPatterns.h"
#include "iree/compiler/Dialect/Util/Conversion/MemRefToUtil/ConvertMemRefToUtil.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {

class TestConversionPass
    : public PassWrapper<TestConversionPass, OperationPass<void>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConversionPass)

  StringRef getArgument() const override { return "iree-util-test-conversion"; }

  StringRef getDescription() const override {
    return "Tests util dialect conversion patterns";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect, func::FuncDialect,
                    mlir::arith::ArithmeticDialect, math::MathDialect,
                    mlir::AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    auto *context = &getContext();

    ConversionTarget conversionTarget(*context);
    conversionTarget.addLegalDialect<arith::ArithmeticDialect>();
    conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateUtilConversionPatterns(context, conversionTarget, typeConverter,
                                   patterns);
    populateGenericStructuralConversionPatterns(context, conversionTarget,
                                                typeConverter, patterns);
    populateMemRefToUtilPatterns(context, conversionTarget, typeConverter,
                                 patterns);

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(patterns)))) {
      getOperation()->emitError() << "conversion to util failed";
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<void>> createTestConversionPass() {
  return std::make_unique<TestConversionPass>();
}

static PassRegistration<TestConversionPass> pass;

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
