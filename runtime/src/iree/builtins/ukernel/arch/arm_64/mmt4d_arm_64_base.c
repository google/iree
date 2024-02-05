// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//#define IREE_DEVICE_STANDALONE
#ifndef IREE_DEVICE_STANDALONE
#include <stdio.h>

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c %c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0')

#define WORD_TO_BINARY_PATTERN "%c%c%c%c %c%c%c%c %c%c%c%c %c%c%c%c"
#define WORD_TO_BINARY(word)  \
  ((word) & 0x8000 ? '1' : '0'), \
  ((word) & 0x4000 ? '1' : '0'), \
  ((word) & 0x2000 ? '1' : '0'), \
  ((word) & 0x1000 ? '1' : '0'), \
  ((word) & 0x0800 ? '1' : '0'), \
  ((word) & 0x0400 ? '1' : '0'), \
  ((word) & 0x0200 ? '1' : '0'), \
  ((word) & 0x0100 ? '1' : '0'), \
  ((word) & 0x0080 ? '1' : '0'), \
  ((word) & 0x0040 ? '1' : '0'), \
  ((word) & 0x0020 ? '1' : '0'), \
  ((word) & 0x0010 ? '1' : '0'), \
  ((word) & 0x0008 ? '1' : '0'), \
  ((word) & 0x0004 ? '1' : '0'), \
  ((word) & 0x0002 ? '1' : '0'), \
  ((word) & 0x0001 ? '1' : '0')

#endif

#include "iree/builtins/ukernel/arch/arm_64/common_arm_64.h"
#include "iree/builtins/ukernel/arch/arm_64/mmt4d_arm_64_internal.h"


static inline void iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;
  float32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vld1q_f32(out_ptr + 4 * i);
    }
  } else {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vdupq_n_f32(0);
    }
  }
  for (int k = 0; k < params->K; ++k) {
    float32x4_t rhs[2];
    for (int i = 0; i < 2; ++i) {
      rhs[i] = vld1q_f32(rhs_ptr + 4 * i);
    }
    rhs_ptr += 8;

    if (M0 == 1) {
      float lhs = *lhs_ptr++;
      acc[0] = vfmaq_n_f32(acc[0], rhs[0], lhs);
      acc[1] = vfmaq_n_f32(acc[1], rhs[1], lhs);
    } else if (M0 == 2) {
      float32x2_t lhs = vld1_f32(lhs_ptr);
      lhs_ptr += 2;
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], lhs, 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], lhs, 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], lhs, 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], lhs, 1);
    } else {
      float32x4_t lhs[2];
      for (int i = 0; i < M0 / 4; ++i) {
        lhs[i] = vld1q_f32(lhs_ptr + 4 * i);
      }
      lhs_ptr += M0;
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], vget_low_f32(lhs[0]), 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], vget_low_f32(lhs[0]), 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], vget_low_f32(lhs[0]), 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], vget_low_f32(lhs[0]), 1);
      acc[4] = vfmaq_lane_f32(acc[4], rhs[0], vget_high_f32(lhs[0]), 0);
      acc[5] = vfmaq_lane_f32(acc[5], rhs[1], vget_high_f32(lhs[0]), 0);
      acc[6] = vfmaq_lane_f32(acc[6], rhs[0], vget_high_f32(lhs[0]), 1);
      acc[7] = vfmaq_lane_f32(acc[7], rhs[1], vget_high_f32(lhs[0]), 1);
      if (M0 == 8) {
        acc[8] = vfmaq_lane_f32(acc[8], rhs[0], vget_low_f32(lhs[1]), 0);
        acc[9] = vfmaq_lane_f32(acc[9], rhs[1], vget_low_f32(lhs[1]), 0);
        acc[10] = vfmaq_lane_f32(acc[10], rhs[0], vget_low_f32(lhs[1]), 1);
        acc[11] = vfmaq_lane_f32(acc[11], rhs[1], vget_low_f32(lhs[1]), 1);
        acc[12] = vfmaq_lane_f32(acc[12], rhs[0], vget_high_f32(lhs[1]), 0);
        acc[13] = vfmaq_lane_f32(acc[13], rhs[1], vget_high_f32(lhs[1]), 0);
        acc[14] = vfmaq_lane_f32(acc[14], rhs[0], vget_high_f32(lhs[1]), 1);
        acc[15] = vfmaq_lane_f32(acc[15], rhs[1], vget_high_f32(lhs[1]), 1);
      }
    }
  }
  for (int i = 0; i < 2 * M0; ++i) {
    vst1q_f32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_2x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_4x8x1_arm_64,
    iree_uk_mmt4d_tile_f32f32f32_8x8x1_arm_64)

// Shared implementation for f16f16f16 and f16f16f32.
// In the f16f16f16 case, intermediate roundings are skipped. This function
// should only be used if IREE_UK_FLAG_MMT4D_SKIP_INTERMEDIATE_ROUNDINGS is set.
static inline void iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, iree_uk_type_t acc_type, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const float16_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float16_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    if (acc_type == IREE_UK_TYPE_FLOAT_32) {
      float* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < 2 * M0; ++i) {
        acc[i] = vld1q_f32(out_ptr + 4 * i);
      }
    } else {
      float16_t* IREE_UK_RESTRICT out_ptr = out_tile;
      for (int i = 0; i < 2 * M0; ++i) {
        acc[i] = vcvt_f32_f16(vld1_f16(out_ptr + 4 * i));
      }
    }
  } else {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vdupq_n_f32(0);
    }
  }
  for (int k = 0; k < params->K; ++k) {
    float32x4_t rhs[2];
    for (int i = 0; i < 2; ++i) {
      rhs[i] = vcvt_f32_f16(vld1_f16(rhs_ptr + 4 * i));
    }
    rhs_ptr += 8;

    if (M0 == 1) {
      float lhs = (float)*lhs_ptr++;
      acc[0] = vfmaq_n_f32(acc[0], rhs[0], lhs);
      acc[1] = vfmaq_n_f32(acc[1], rhs[1], lhs);
    } else if (M0 == 2) {
      float16x4_t lhs_f16 = vld1_dup_f16(lhs_ptr);
      lhs_f16 = vld1_lane_f16(lhs_ptr + 1, lhs_f16, 1);
      lhs_ptr += 2;
      float32x2_t lhs = vget_low_f32(vcvt_f32_f16(lhs_f16));
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], lhs, 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], lhs, 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], lhs, 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], lhs, 1);
    } else {
      float32x4_t lhs[2];
      for (int i = 0; i < M0 / 4; ++i) {
        lhs[i] = vcvt_f32_f16(vld1_f16(lhs_ptr + 4 * i));
      }
      lhs_ptr += M0;
      acc[0] = vfmaq_lane_f32(acc[0], rhs[0], vget_low_f32(lhs[0]), 0);
      acc[1] = vfmaq_lane_f32(acc[1], rhs[1], vget_low_f32(lhs[0]), 0);
      acc[2] = vfmaq_lane_f32(acc[2], rhs[0], vget_low_f32(lhs[0]), 1);
      acc[3] = vfmaq_lane_f32(acc[3], rhs[1], vget_low_f32(lhs[0]), 1);
      acc[4] = vfmaq_lane_f32(acc[4], rhs[0], vget_high_f32(lhs[0]), 0);
      acc[5] = vfmaq_lane_f32(acc[5], rhs[1], vget_high_f32(lhs[0]), 0);
      acc[6] = vfmaq_lane_f32(acc[6], rhs[0], vget_high_f32(lhs[0]), 1);
      acc[7] = vfmaq_lane_f32(acc[7], rhs[1], vget_high_f32(lhs[0]), 1);
      if (M0 == 8) {
        acc[8] = vfmaq_lane_f32(acc[8], rhs[0], vget_low_f32(lhs[1]), 0);
        acc[9] = vfmaq_lane_f32(acc[9], rhs[1], vget_low_f32(lhs[1]), 0);
        acc[10] = vfmaq_lane_f32(acc[10], rhs[0], vget_low_f32(lhs[1]), 1);
        acc[11] = vfmaq_lane_f32(acc[11], rhs[1], vget_low_f32(lhs[1]), 1);
        acc[12] = vfmaq_lane_f32(acc[12], rhs[0], vget_high_f32(lhs[1]), 0);
        acc[13] = vfmaq_lane_f32(acc[13], rhs[1], vget_high_f32(lhs[1]), 0);
        acc[14] = vfmaq_lane_f32(acc[14], rhs[0], vget_high_f32(lhs[1]), 1);
        acc[15] = vfmaq_lane_f32(acc[15], rhs[1], vget_high_f32(lhs[1]), 1);
      }
    }
  }
  if (acc_type == IREE_UK_TYPE_FLOAT_32) {
    float* IREE_UK_RESTRICT out_ptr = out_tile;
    for (int i = 0; i < 2 * M0; ++i) {
      vst1q_f32(out_ptr + 4 * i, acc[i]);
    }
  } else {
    float16_t* IREE_UK_RESTRICT out_ptr = out_tile;
    for (int i = 0; i < 2 * M0; ++i) {
      vst1_f16(out_ptr + 4 * i, vcvt_f16_f32(acc[i]));
    }
  }
}

static inline void iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_arm_64(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_16, M0);
}

static inline void iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  iree_uk_mmt4d_tile_f16f16fXX_1x8x1_to_8x8x1_arm_64(
      out_tile, lhs_panel, rhs_panel, params, IREE_UK_TYPE_FLOAT_32, M0);
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_1x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_2x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_4x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f32_8x8x1_arm_64)

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_1x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_2x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_4x8x1_arm_64,
    iree_uk_mmt4d_tile_f16f16f16_8x8x1_arm_64)

static inline void iree_uk_mmt4d_tile_s8s8s32_1x8x1_to_8x8x1_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  int32x4_t acc[16];
  if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vld1q_s32(out_ptr + 4 * i);
    }
  } else {
    for (int i = 0; i < 2 * M0; ++i) {
      acc[i] = vdupq_n_s32(0);
    }
  }
  for (int k = 0; k < params->K; ++k) {
    int16x8_t rhs = vmovl_s8(vld1_s8(rhs_ptr));
    rhs_ptr += 8;
    if (M0 <= 4) {
      for (int i = 0; i < M0; ++i) {
        int16_t lhs = *lhs_ptr++;
        acc[2 * i + 0] = vmlal_n_s16(acc[2 * i + 0], vget_low_s16(rhs), lhs);
        acc[2 * i + 1] = vmlal_n_s16(acc[2 * i + 1], vget_high_s16(rhs), lhs);
      }
    } else {
      int16x8_t lhs = vmovl_s8(vld1_s8(lhs_ptr));
      lhs_ptr += 8;
      acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(rhs), vget_low_s16(lhs), 0);
      acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(rhs), vget_low_s16(lhs), 0);
      acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(rhs), vget_low_s16(lhs), 1);
      acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(rhs), vget_low_s16(lhs), 1);
      acc[4] = vmlal_lane_s16(acc[4], vget_low_s16(rhs), vget_low_s16(lhs), 2);
      acc[5] = vmlal_lane_s16(acc[5], vget_high_s16(rhs), vget_low_s16(lhs), 2);
      acc[6] = vmlal_lane_s16(acc[6], vget_low_s16(rhs), vget_low_s16(lhs), 3);
      acc[7] = vmlal_lane_s16(acc[7], vget_high_s16(rhs), vget_low_s16(lhs), 3);
      acc[8] = vmlal_lane_s16(acc[8], vget_low_s16(rhs), vget_high_s16(lhs), 0);
      acc[9] =
          vmlal_lane_s16(acc[9], vget_high_s16(rhs), vget_high_s16(lhs), 0);
      acc[10] =
          vmlal_lane_s16(acc[10], vget_low_s16(rhs), vget_high_s16(lhs), 1);
      acc[11] =
          vmlal_lane_s16(acc[11], vget_high_s16(rhs), vget_high_s16(lhs), 1);
      acc[12] =
          vmlal_lane_s16(acc[12], vget_low_s16(rhs), vget_high_s16(lhs), 2);
      acc[13] =
          vmlal_lane_s16(acc[13], vget_high_s16(rhs), vget_high_s16(lhs), 2);
      acc[14] =
          vmlal_lane_s16(acc[14], vget_low_s16(rhs), vget_high_s16(lhs), 3);
      acc[15] =
          vmlal_lane_s16(acc[15], vget_high_s16(rhs), vget_high_s16(lhs), 3);
    }
  }
  for (int i = 0; i < 2 * M0; ++i) {
    vst1q_s32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4_8(
    iree_uk_mmt4d_tile_s8s8s32_1x8x1_to_8x8x1_arm_64,
    iree_uk_mmt4d_tile_s8s8s32_1x8x1_arm_64,
    iree_uk_mmt4d_tile_s8s8s32_2x8x1_arm_64,
    iree_uk_mmt4d_tile_s8s8s32_4x8x1_arm_64,
    iree_uk_mmt4d_tile_s8s8s32_8x8x1_arm_64)


static inline void iree_uk_mmt4d_tile_s8s4s32_1x16x2_to_4x16x2_arm_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 4 && iree_uk_is_po2_u32(M0));
  IREE_UK_ASSERT(!(params->K0 % 2));
  const iree_uk_int8_t* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const iree_uk_int8_t* IREE_UK_RESTRICT rhs_ptr = rhs_panel;

#ifndef IREE_DEVICE_STANDALONE
  printf("----- Start optimized M: %lld, N: %lld, K: %lld, M0: %d, N0: %d, K0: %d ------\n", params->M, params->N, params->K, M0, params->N0, params->K0);
#endif

  iree_uk_int32_t* IREE_UK_RESTRICT out_ptr = out_tile;
  int32x4_t acc[16];
  // We start with zero accumulators and add the value of *out_ptr later.
  for (int i = 0; i < 4 * M0; ++i) {
    acc[i] = vdupq_n_s32(0);
  }

#ifndef IREE_DEVICE_STANDALONE
  for (int i = 0; i < 4 * M0; ++i) {
    for (int j = 0; j < 4; j++) {
      int32_t val = acc[i][j];
      printf("init acc[%d][%d]: %d\n", i, j, val);
    }
  }
#endif

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));

  for (int k = 0; k < params->K; ++k) {
    // Handle 1x16x2.
    int16x8_t lhs = vmovl_s8(vld1_s8(lhs_ptr));
    lhs_ptr += 2 * M0;

    int8x8_t rhs_0 = vld1_s8(rhs_ptr);
    rhs_ptr += 8;
    int8x8_t rhs_1 = vld1_s8(rhs_ptr);
    rhs_ptr += 8;

    int8x8_t rhs_0_low = vshl_n_s8(rhs_0, 4);
    int8x8_t rhs_0_high = vand_s8(rhs_0, vmask);

    int8x8_t rhs_1_low = vshl_n_s8(rhs_1, 4);
    int8x8_t rhs_1_high = vand_s8(rhs_1, vmask);

    int16x8_t rhs_a = vmovl_s8(rhs_0_low);
    int16x8_t rhs_b = vmovl_s8(rhs_0_high);
    int16x8_t rhs_c = vmovl_s8(rhs_1_low);
    int16x8_t rhs_d = vmovl_s8(rhs_1_high);

#ifndef IREE_DEVICE_STANDALONE
    printf("lhs_0: %d\n", lhs[0]);
    printf("lhs_1: %d\n", lhs[1]);

    for (int m = 0; m < 8; m++) {
      int8_t rhs_byte = rhs_0[m];
      printf("rhs_0[%d]: rhs_byte "BYTE_TO_BINARY_PATTERN"\n", m, BYTE_TO_BINARY(rhs_byte));
    }
//    for (int m = 0; m < 8; m++) {
//      int16_t val = rhs_0_low[m];
//      printf("rhs_0_low[%d]: "BYTE_TO_BINARY_PATTERN"\n", m, BYTE_TO_BINARY(val));
//    }
//    for (int m = 0; m < 8; m++) {
//      int16_t val = rhs_0_high[m];
//      printf("rhs_0_high[%d]: "BYTE_TO_BINARY_PATTERN"\n", m, BYTE_TO_BINARY(val));
//    }
    for (int m = 0; m < 8; m++) {
      int16_t val = rhs_a[m];
      printf("rhs_a[%d]: "WORD_TO_BINARY_PATTERN"\n", m, WORD_TO_BINARY(val));
      printf("rhs_a[%d]: %d\n", m, val);
    }
    for (int m = 0; m < 8; m++) {
      int16_t val = rhs_b[m];
      printf("rhs_b[%d]: "WORD_TO_BINARY_PATTERN"\n", m, WORD_TO_BINARY(val));
      printf("rhs_b[%d]: %d\n", m, val);
    }

    for (int m = 0; m < 8; m++) {
      int8_t rhs_byte = rhs_1[m];
      printf("rhs_1[%d]: rhs_byte %hhx\n", m, rhs_byte);
    }
    for (int m = 0; m < 8; m++) {
      int16_t val = rhs_c[m];
      printf("rhs_c[%d]: %d\n", m, val);
    }
    for (int m = 0; m < 8; m++) {
      int16_t val = rhs_d[m];
      printf("rhs_d[%d]: %d\n", m, val);
    }
#endif

    acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(rhs_a), vget_low_s16(lhs), 0);
#ifndef IREE_DEVICE_STANDALONE
    int16_t lhs_0_test = lhs[0];
    for (int i = 0; i < 4; i++) {
      int16_t rhs_0_test = rhs_a[i];
      int32_t val = acc[0][i];
      printf("first acc[0][%d]: %d. lhs: %d, rhs: %d\n", i, val, lhs_0_test, rhs_0_test);
    }
#endif

    acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(rhs_a), vget_low_s16(lhs), 0);
    acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(rhs_c), vget_low_s16(lhs), 0);
    acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(rhs_c), vget_low_s16(lhs), 0);

    acc[0] = vmlal_lane_s16(acc[0], vget_low_s16(rhs_b), vget_low_s16(lhs), 1);
#ifndef IREE_DEVICE_STANDALONE
    int16_t lhs_1_test = lhs[1];
    for (int i = 0; i < 4; i++) {
      int16_t rhs_1_test = rhs_b[i];
      int32_t val = acc[0][i];
      printf("second acc[0][%d]: %d. lhs: %d, rhs: %d\n", i, val, lhs_1_test, rhs_1_test);
    }
#endif

    acc[1] = vmlal_lane_s16(acc[1], vget_high_s16(rhs_b), vget_low_s16(lhs), 1);
    acc[2] = vmlal_lane_s16(acc[2], vget_low_s16(rhs_d), vget_low_s16(lhs), 1);
    acc[3] = vmlal_lane_s16(acc[3], vget_high_s16(rhs_d), vget_low_s16(lhs), 1);

    if (M0 >= 2) {
      // Handle 2x16x2.
      acc[4] =
          vmlal_lane_s16(acc[4], vget_low_s16(rhs_a), vget_low_s16(lhs), 2);
      acc[5] =
          vmlal_lane_s16(acc[5], vget_high_s16(rhs_a), vget_low_s16(lhs), 2);
      acc[6] =
          vmlal_lane_s16(acc[6], vget_low_s16(rhs_c), vget_low_s16(lhs), 2);
      acc[7] =
          vmlal_lane_s16(acc[7], vget_high_s16(rhs_c), vget_low_s16(lhs), 2);

      acc[4] =
          vmlal_lane_s16(acc[4], vget_low_s16(rhs_b), vget_low_s16(lhs), 3);
      acc[5] =
          vmlal_lane_s16(acc[5], vget_high_s16(rhs_b), vget_low_s16(lhs), 3);
      acc[6] =
          vmlal_lane_s16(acc[6], vget_low_s16(rhs_d), vget_low_s16(lhs), 3);
      acc[7] =
          vmlal_lane_s16(acc[7], vget_high_s16(rhs_d), vget_low_s16(lhs), 3);

      if (M0 == 4) {
        // Handle 4x16x2.
        acc[8] =
            vmlal_lane_s16(acc[8], vget_low_s16(rhs_a), vget_high_s16(lhs), 0);
        acc[9] =
            vmlal_lane_s16(acc[9], vget_high_s16(rhs_a), vget_high_s16(lhs), 0);
        acc[10] =
            vmlal_lane_s16(acc[10], vget_low_s16(rhs_c), vget_high_s16(lhs), 0);
        acc[11] = vmlal_lane_s16(acc[11], vget_high_s16(rhs_c),
                                 vget_high_s16(lhs), 0);

        acc[8] =
            vmlal_lane_s16(acc[8], vget_low_s16(rhs_b), vget_high_s16(lhs), 1);
        acc[9] =
            vmlal_lane_s16(acc[9], vget_high_s16(rhs_b), vget_high_s16(lhs), 1);
        acc[10] =
            vmlal_lane_s16(acc[10], vget_low_s16(rhs_d), vget_high_s16(lhs), 1);
        acc[11] = vmlal_lane_s16(acc[11], vget_high_s16(rhs_d),
                                 vget_high_s16(lhs), 1);

        acc[12] =
            vmlal_lane_s16(acc[12], vget_low_s16(rhs_a), vget_high_s16(lhs), 2);
        acc[13] = vmlal_lane_s16(acc[13], vget_high_s16(rhs_a),
                                 vget_high_s16(lhs), 2);
        acc[14] =
            vmlal_lane_s16(acc[14], vget_low_s16(rhs_c), vget_high_s16(lhs), 2);
        acc[15] = vmlal_lane_s16(acc[15], vget_high_s16(rhs_c),
                                 vget_high_s16(lhs), 2);

        acc[12] =
            vmlal_lane_s16(acc[12], vget_low_s16(rhs_b), vget_high_s16(lhs), 3);
        acc[13] = vmlal_lane_s16(acc[13], vget_high_s16(rhs_b),
                                 vget_high_s16(lhs), 3);
        acc[14] =
            vmlal_lane_s16(acc[14], vget_low_s16(rhs_d), vget_high_s16(lhs), 3);
        acc[15] = vmlal_lane_s16(acc[15], vget_high_s16(rhs_d),
                                 vget_high_s16(lhs), 3);
      }
    }
  }

  // Divide by 16.
//  for (int i = 0; i < 4 * M0; ++i) {
//#ifndef IREE_DEVICE_STANDALONE
////    for (int j = 0; j < 4; j++) {
////      int32_t val = acc[i][j];
////      printf("before acc[%d][%d]: %d\n", i, j, val);
////    }
//#endif
//    acc[i] = vshrq_n_s32(acc[i], 4);
//#ifndef IREE_DEVICE_STANDALONE
//    for (int j = 0; j < 4; j++) {
//      int32_t val = acc[i][j];
//      printf("after acc[%d][%d]: %d\n", i, j, val);
//    }
//#endif
//  }

  for (int i = 0; i < 4 * M0; ++i) {
    acc[i] = vshrq_n_s32(acc[i], 4);

    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      int32x4_t existing_acc = vld1q_s32(out_ptr + 4 * i);
      acc[i] = vaddq_s32(acc[i], existing_acc);
    }

#ifndef IREE_DEVICE_STANDALONE
    for (int j = 0; j < 4; j++) {
      int32_t val = acc[i][j];
      printf("after acc[%d][%d]: %d\n", i, j, val);
    }
#endif
    vst1q_s32(out_ptr + 4 * i, acc[i]);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0_1_2_4(
    iree_uk_mmt4d_tile_s8s4s32_1x16x2_to_4x16x2_arm_64,
    iree_uk_mmt4d_tile_s8s4s32_1x16x2_arm_64,
    iree_uk_mmt4d_tile_s8s4s32_2x16x2_arm_64,
    iree_uk_mmt4d_tile_s8s4s32_4x16x2_arm_64)
