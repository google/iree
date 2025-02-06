// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Math/Transforms/Approximation.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

/// Command line to use native hardware operations instead of polynomial
/// approximation.
static llvm::cl::opt<bool> clNativeMathPrecision(
    "iree-codegen-gpu-native-math-precision",
    llvm::cl::desc("Deprecated! This flag had buggy/unintentional semantics. "
                   "Its original description said: \"Skip polynomial lowering "
                   "for math op natively available on GPU.\""),
    llvm::cl::init(false));

#define GEN_PASS_DEF_POLYNOMIALAPPROXIMATIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

#if 0
static void populateErfPattern(RewritePatternSet &patterns) {
  if (clNativeMathPrecision) {
    patterns.add<math::ErfPolynomialApproximation>(patterns.getContext());
  } else {
    populateExpandExp2FPattern(patterns);
    populateMathPolynomialApproximationPatterns(patterns);
    populateExpandRoundEvenPattern(patterns);
  }
}
#endif

static void populateMathFunctionsExpandPatterns(
    RewritePatternSet &patterns,
    const std::function<bool(StringRef)> &predicate) {
  if (predicate("tan")) {
    populateExpandTanPattern(patterns);
  }
  if (predicate("sinh")) {
    populateExpandSinhPattern(patterns);
  }
  if (predicate("cosh")) {
    populateExpandCoshPattern(patterns);
  }
  if (predicate("asinh")) {
    populateExpandAsinhPattern(patterns);
  }
  if (predicate("acosh")) {
    populateExpandAcoshPattern(patterns);
  }
  if (predicate("atanh")) {
    populateExpandAtanhPattern(patterns);
  }
  if (predicate("powf")) {
    populateExpandPowFPattern(patterns);
  }
  if (predicate("fpowi")) {
    populateExpandFPowIPattern(patterns);
  }
  if (predicate("exp2")) {
    populateExpandExp2FPattern(patterns);
  }
  if (predicate("roundeven")) {
    populateExpandRoundEvenPattern(patterns);
  }
}

/// math dialect elementry functions -> polynomial form.
class PolynomialApproximationPass final
    : public impl::PolynomialApproximationPassBase<
          PolynomialApproximationPass> {
public:
  using Base::Base;

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    llvm::dbgs() << "XXXXXXXXXXXXXXXXXXXXXXXX initializeOptions sees "
                 << expandOps.ValueStr << "\n";
    llvm::dbgs() << "XXXXXXXXXXXXXXXXXXXXXXXX initializeOptions gets options: "
                 << options << "\n";

    LogicalResult result = Pass::initializeOptions(options, errorHandler);
    llvm::dbgs() << "XXXXXXXXXXXXXXXXXXXXXXXX initializeOptions sees "
                 << expandOps.ValueStr << "\n";

    return result;
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    llvm::dbgs() << "XXXXXXXXXXXXXXXXXXXXXXXX" << expandOps.ValueStr << "\n";

    auto mathFunctionExpansion = [=](StringRef name) {
      // TODO(bjacob): this weird `if` statement is for compatibility only.
      // Remove when legacy `clNativeMathPrecision` is dropped.
      if (clNativeMathPrecision) {
        if (name == "exp2" || name == "roundeven") {
          return false;
        }
      }
      return llvm::is_contained(expandOps, name);
    };
    populateMathFunctionsExpandPatterns(patterns, mathFunctionExpansion);

    auto predicateF32Expansion = [=](StringRef name) {
      // TODO(bjacob): this weird `if` statement is for compatibility only.
      // Remove when legacy `clNativeMathPrecision` is dropped.
      if (clNativeMathPrecision) {
        return false;
      }
      return llvm::is_contained(f32ExpandOps, name);
      /*
      name == "atan" || name == "atan2" || name == "tanh" ||
          name == "log" || name == "log2" || name == "log1p" ||
          name == "erf" || name == "exp" || name == "expm1" ||
          name == "cbrt" || name == "sin" || name == "cos"; */
    };
    populateMathF32ExpansionPatterns(patterns, predicateF32Expansion);

    auto predicateApprox = [=](StringRef name) {
      return llvm::is_contained(approxOps, name);
      /*
          return name == "atan" || name == "atan2" || name == "tanh" ||
           name == "log" || name == "log2" || name == "log1p" ||
           name == "erf" || name == "asin" || name == "acos" || name == "exp" ||
           name == "expm1" || name == "cbrt" || name == "sin" || name == "cos";
       */
    };
    populateMathPolynomialApproximationPatterns(patterns, predicateApprox);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
