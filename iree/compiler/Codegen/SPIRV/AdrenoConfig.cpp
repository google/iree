// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AdrenoConfig.h - Adreno CodeGen Configurations ---------------------===//
//
// This file contains CodeGen configurations for Adreno GPUs.
//
//===----------------------------------------------------------------------===//

#include <array>

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

namespace mlir {
namespace iree_compiler {
namespace detail {

//===----------------------------------------------------------------------===//
// Convolution
//===----------------------------------------------------------------------===//

/// Sets CodeGen configurations via attributes to the given convolution
/// `linalgOp` by trying to achieve the given `bestTilingFactor`, which is how
/// many scalar elements each thread should handle.
static LogicalResult setOpConfig(linalg::LinalgOp linalgOp,
                                 int64_t bestTilingFactor) {
  ArrayRef<int64_t> inputShape = getUntiledShape(linalgOp.inputs()[0]);
  ArrayRef<int64_t> outputShape = getUntiledResultShape(linalgOp, 0);
  if (llvm::any_of(inputShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(outputShape, ShapedType::isDynamic)) return success();

  int64_t ic = inputShape[3];
  int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // The conversion pipeline requires the input channel dimension to be some
  // multipler of four, or less than four.
  if (!(ic % 4 == 0 || ic < 4)) return success();

  // The core idea is to distribute the convolution OH/OW/OC dimension to the
  // workgroup Z/Y/X dimension, with each thread in a workgroup handling
  // multiple vector elements. We try to 1) utilize all threads in a subgroup,
  // and 2) handle an optimal tile size along each dimension.

  int64_t residualThreads = 64;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1);        // (X, Y, Z)
  SmallVector<int64_t, 4> workgroupTileSizes(4, 0);   // (N, OH, OW, OC)
  SmallVector<int64_t, 4> invocationTileSizes(4, 0);  // (N, OH, OW, OC)

  // Deduce the configuration for the OC dimension.
  for (int64_t x = residualThreads; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = 4;
    if (oc % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[3] = x * chosenTileSize;
      invocationTileSizes[3] = chosenTileSize;
      residualThreads /= x;
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[3] == 0) return success();

  // Deduce the configruation for the OW and OH dimension. Try to make them even
  // if possible given we typically have images with the same height and width.
  if (ow == oh && residualThreads != 1 &&
      llvm::Log2_64(residualThreads) % 2 == 0) {
    int64_t yz = 1 << (llvm::Log2_64(residualThreads) / 2);
    int64_t chosenTileSize = 1 << (llvm::Log2_64(residualTilingFactor) / 2);
    while (ow % chosenTileSize != 0) chosenTileSize >>= 1;
    workgroupSize[1] = workgroupSize[2] = yz;
    workgroupTileSizes[2] = workgroupTileSizes[1] = yz * chosenTileSize;
    invocationTileSizes[2] = invocationTileSizes[1] = chosenTileSize;
  } else {
    auto decideOneDim = [&](int64_t inputDim, int64_t &wgDimSize,
                            int64_t &wgTileSize, int64_t &invoTileSize) {
      for (int64_t dim = residualThreads; dim >= 1; dim >>= 1) {
        int64_t chosenTileSize = 0;
        for (int64_t t = residualTilingFactor; t >= 1; t >>= 1) {
          if (inputDim % (dim * t) == 0) {
            chosenTileSize = t;
            break;
          }
        }
        if (chosenTileSize) {
          wgDimSize = dim;
          wgTileSize = dim * chosenTileSize;
          invoTileSize = chosenTileSize;
          residualThreads /= dim;
          residualTilingFactor /= chosenTileSize;
          return true;
        }
      }
      return false;
    };

    if (!decideOneDim(ow, workgroupSize[1], workgroupTileSizes[2],
                      invocationTileSizes[2]) ||
        !decideOneDim(oh, workgroupSize[2], workgroupTileSizes[1],
                      invocationTileSizes[1])) {
      return success();
    }
  }

  auto pipeline = IREE::HAL::DispatchLoweringPassPipeline::SPIRVVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.emplace_back();  // Subgroup level
  tileSizes.push_back(invocationTileSizes);

  auto funcOp = linalgOp->getParentOfType<FuncOp>();
  if (failed(setOpConfigAndEntryPointFnTranslation(
          funcOp, linalgOp, tileSizes, {}, pipeline, workgroupSize))) {
    return failure();
  }
  return defineConvWorkgroupCountRegion(
      linalgOp, llvm::makeArrayRef(outputShape).drop_front(),
      llvm::makeArrayRef(workgroupTileSizes).drop_front());
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult setAdrenoCodeGenConfig(const spirv::TargetEnv &,
                                     Operation *rootOp) {
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([](auto op) {
        std::array<int64_t, 2> workgroupXY = {32, 2};
        std::array<int64_t, 3> threadMNK = {8, 8, 4};
        return setMatmulOpConfig(op, workgroupXY, threadMNK);
      })
      .Case<linalg::Conv2DNhwcHwcfOp>(
          [](auto op) { return setOpConfig(op, 32); })
      .Case<linalg::DepthwiseConv2DNhwOp>(
          [](auto op) { return setOpConfig(op, 16); })
      .Default([](Operation *) { return success(); });
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
