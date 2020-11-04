// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Flow/Analysis/ExecutableHashAnalysis.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

class ComputeExecutableHashesPass
    : public PassWrapper<ComputeExecutableHashesPass,
                         OperationPass<ExecutableOp>> {
 public:
  void runOnOperation() override {
    getAnalysis<ExecutableHashAnalysis>();
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<OperationPass<ExecutableOp>>
createComputeExecutableHashesPass() {
  return std::make_unique<ComputeExecutableHashesPass>();
}

static PassRegistration<ComputeExecutableHashesPass> pass(
    "iree-flow-compute-executable-hashes",
    "Computes and caches hashes of executable ops");

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
