// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_MMT4D_H_
#define IREE_BUILTINS_UKERNEL_MMT4D_H_

#include "iree/builtins/ukernel/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TODO(benoitjacob): move to internal, user specifies type in flags.
typedef enum iree_uk_mmt4d_type_t {
  iree_uk_mmt4d_type_f32f32f32 =
      IREE_UK_TIE_3_TYPES_LITERAL(FLOAT_32, FLOAT_32, FLOAT_32),
  iree_uk_mmt4d_type_i8i8i32 =
      IREE_UK_TIE_3_TYPES_LITERAL(INT_8, INT_8, INT_32),
} iree_uk_mmt4d_type_t;

typedef struct iree_uk_mmt4d_params_t {
  iree_uk_mmt4d_type_t type;
  iree_uk_uint32_t flags;
  iree_uk_ssize_t lhs_stride;
  iree_uk_ssize_t rhs_stride;
  iree_uk_ssize_t out_stride;
  iree_uk_ssize_t M;
  iree_uk_ssize_t N;
  iree_uk_ssize_t K;
  iree_uk_int32_t M0;
  iree_uk_int32_t N0;
  iree_uk_int32_t K0;
  const void* lhs_buffer;
  const void* rhs_buffer;
  void* out_buffer;
  const iree_uk_uint64_t* cpu_data;
} iree_uk_mmt4d_params_t;

IREE_UK_EXPORT void iree_uk_mmt4d(const iree_uk_mmt4d_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BUILTINS_UKERNEL_MMT4D_H_
