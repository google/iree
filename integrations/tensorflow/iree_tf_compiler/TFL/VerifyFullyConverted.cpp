// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_tf_compiler/TFL/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace iree_integrations {
namespace TFL {

static bool isTFLOp(Operation *op) {
  if (!op || !op->getDialect()) return false;
  StringRef opNamespace = op->getDialect()->getNamespace();
  return opNamespace == mlir::TFL::TensorFlowLiteDialect::getDialectNamespace();
}

class VerifyFullyConvertedPass
    : public PassWrapper<VerifyFullyConvertedPass, FunctionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  }

  StringRef getArgument() const override {
    return "iree-tflite-verify-fully-converted";
  }

  StringRef getDescription() const override {
    return "Verifies that all TFLite frontend ops were converted and none "
           "remain";
  }

  // Validates that no TFLite frontends ops are in the function.
  void runOnFunction() override {
    DenseSet<Operation *> illegalOps;
    getFunction().walk([&](Operation *op) {
      if (isTFLOp(op)) illegalOps.insert(op);
    });
    if (!illegalOps.empty()) {
      emitLegalizationErrors(getFunction().getLoc(), illegalOps);
      return signalPassFailure();
    }
  }

  // Emits debug information which includes the number of ops of each type which
  // failed to legalize.
  void emitLegalizationErrors(Location loc,
                              const DenseSet<Operation *> &nonlegalizedOps) {
    // Print op errors for each of the TFLite ops that still remain.
    std::map<StringRef, int> opNameCounts;
    for (Operation *nonlegalizedOp : nonlegalizedOps) {
      StringRef opName = nonlegalizedOp->getName().getStringRef();
      opNameCounts[opName]++;
      nonlegalizedOp->emitOpError() << ": unlegalized TFLite op still exists";
    }

    std::vector<std::string> errorMessages;
    errorMessages.reserve(opNameCounts.size());
    for (const auto &opInfo : opNameCounts) {
      errorMessages.push_back(
          llvm::formatv("\t{0} (count: {1})", opInfo.first, opInfo.second));
    }
    emitError(loc) << "The following TFLite operations still remain: \n"
                   << llvm::join(errorMessages, "\n") << "\n";
  }
};

static PassRegistration<VerifyFullyConvertedPass> pass;

std::unique_ptr<OperationPass<FuncOp>> createVerifyFullyConvertedPass() {
  return std::make_unique<VerifyFullyConvertedPass>();
}

}  // namespace TFL
}  // namespace iree_integrations
}  // namespace mlir
