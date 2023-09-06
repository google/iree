// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_SOFTMAX_INTERNAL_H_
#define IREE_BUILTINS_UKERNEL_SOFTMAX_INTERNAL_H_

#include "iree/builtins/ukernel/softmax.h"

typedef void (*iree_uk_softmax_tile_func_t)(
    const void* IREE_UK_RESTRICT src_buffer,
    void* IREE_UK_RESTRICT out_buffer,
    iree_uk_int32_t M,
    iree_uk_int32_t N);

iree_uk_softmax_tile_func_t iree_uk_softmax_select_tile_func(const iree_uk_softmax_params_t *params);

iree_uk_softmax_tile_func_t iree_uk_softmax_select_tile_func_arch(const iree_uk_softmax_params_t *params);

// Tile kernel declarations. Prototype matches iree_uk_softmax_tile_func_t.
#define IREE_UK_SOFTMAX_TILE_FUNC_DECL(NAME)                             \
  void NAME(const void* IREE_UK_RESTRICT src_buffer,                      \
            void* IREE_UK_RESTRICT out_buffer,                 \
            iree_uk_int32_t M, iree_uk_int32_t N);

#endif // IREE_BUILTINS_UKERNEL_SOFTMAX_INTERNAL_H_
