// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

struct AbstractGemmLikeStrategy {
  AbstractGemmLikeStrategy() {}

  virtual ~AbstractGemmLikeStrategy();

  //===--------------------------------------------------------------------===//
  // Parameters that control the tiling and mapping.
  //===--------------------------------------------------------------------===//
  struct MappingInfo {
    SmallVector<int64_t> numThreads;
    // Explicitly computing the tileSizes is only needed until masked
    // vectorization properly computes the bounds automatically.
    SmallVector<int64_t> tileSizes;
    SmallVector<Attribute> threadMapping;
    void print(llvm::raw_ostream &os) const;
    LLVM_DUMP_METHOD void dump() const;
  };

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies. The initial values are set by initDefaultValues();
  SmallVector<int64_t> blockTileSizes;
  int64_t reductionTileSize;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> numWarps;
  virtual int64_t blockTileM() const = 0;
  virtual int64_t blockTileN() const = 0;

  virtual int64_t numWarpsM() const = 0;
  virtual int64_t numWarpsN() const = 0;

  virtual MappingInfo getBlockMapping() const = 0;

  /// Common values based on derived quantities.
  int64_t totalNumThreads() const {
    int64_t res = 1;
    for (auto v : numThreads) res *= v;
    return res;
  }

  int64_t totalNumWarps() const {
    int64_t res = 1;
    for (auto v : numWarps) res *= v;
    return res;
  }

  //===--------------------------------------------------------------------===//
  // Parameters that control copy/padding transfers from global to shared.
  //===--------------------------------------------------------------------===//
  SmallVector<float> paddingValues;
  SmallVector<int64_t> paddingDimensions;
  SmallVector<int64_t> packingDimensions;

  // Copy vector sizes based on innermost K/N dims.
  // TODO: These are now hardcoded for f32 but are element-type dependent.
  int64_t lhsCopyVectorSize() const {
    if (k() % 4 == 0) return 4;
    if (k() % 2 == 0) return 2;
    return 1;
  }
  int64_t rhsCopyVectorSize() const {
    if (n() % 4 == 0) return 4;
    if (n() % 2 == 0) return 2;
    return 1;
  }
  int64_t resCopyVectorSize() const { return rhsCopyVectorSize(); }

  bool alignedLhs() const {
    return m() % blockTileM() == 0 && k() % reductionTileSize == 0;
  }
  bool alignedRhs() const {
    return n() % blockTileN() == 0 && k() % reductionTileSize == 0;
  }
  bool alignedRes() const {
    return m() % blockTileM() == 0 && n() % blockTileN() == 0;
  }

  virtual MappingInfo lhsCopyMapping() const = 0;
  virtual LogicalResult validateLhsCopyMapping() const = 0;
  virtual MappingInfo rhsCopyMapping() const = 0;
  virtual LogicalResult validateRhsCopyMapping() const = 0;
  virtual MappingInfo resCopyMapping() const = 0;
  virtual LogicalResult validateResCopyMapping() const = 0;

  //===--------------------------------------------------------------------===//
  // Parameters that control compute mapping decisions.
  //===--------------------------------------------------------------------===//
  bool useAsyncCopies;
  bool useMmaSync;
  int64_t pipelineDepth;
  virtual MappingInfo computeMapping() const = 0;

  virtual LogicalResult validate() const = 0;

  //===--------------------------------------------------------------------===//
  // Problem-related quantities.
  //===--------------------------------------------------------------------===//
  virtual int64_t m() const = 0;
  virtual int64_t n() const = 0;
  virtual int64_t k() const = 0;

  virtual void print(llvm::raw_ostream &os) const = 0;
  virtual LLVM_DUMP_METHOD void dump() const = 0;

  //===--------------------------------------------------------------------===//
  // Preconditions of internal transforms lifted to the top-level for more
  // actionnable error messages. In the fullness of time, transforms should
  // expose preconditions and we should aggregate them automatically.
  //===--------------------------------------------------------------------===//

  // TODO: To handle different element types efficiently, it would be much
  // better to expose the unrolling to native size explicitly to the transforms
  // rather than hide it behind an opaque transform.

  // wmma preconditions that we want to lift out in an actionnable top-level
  // error message instead of failing late in the transformation schedule.
  // TODO: These are now hardcoded for f32 but are element-type dependent.
  // Precondition: the pipeline transformation for wmma requires at least 2
  // k-groups.
  constexpr static int64_t kMinWmmaMinM = 16;
  constexpr static int64_t kMinWmmaMinN = 16;
  constexpr static int64_t kMinWmmaMinK = 8;

  // mma.sync preconditions that we want to lift out in an actionnable top-level
  // error message instead of failing late in the transformation schedule.
  // TODO: These are now hardcoded for f32 but are element-type dependent.
  // Precondition: the pipeline transformation for mma.sync requires at least 2
  // k-groups.
  constexpr static int64_t kMinMmaSyncGroups = 2;
  // Precondition: the pipeline transformation for mma.sync requires at least a
  // pipeline depth of 3.
  constexpr static int64_t kMinMmaSyncPipelineDepth = 3;
  // Precondition: if mma.sync is used, the tile sizes must be at least 8x8x4.
  constexpr static int64_t kMinMmaSyncMinM = 8;
  constexpr static int64_t kMinMmaSyncMinN = 8;
  constexpr static int64_t kMinMmaSyncMinK = 4;
};

}  // namespace gpu
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_ABSTRACT_GEMM_LIKE_STRATEGY_H_
