// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"
#include "iree/builtins/ukernel/api.h"
#include "iree/builtins/ukernel/tools/test.h"
#include "iree/builtins/ukernel/tools/util.h"

static void iree_unpack_reference(const iree_uk_unpack_params_t* params) {
  // For now, the input and output element types are always the same.
  iree_uk_type_t elem_type = iree_uk_unpack_in_type(params->type);
  iree_uk_ssize_t elem_size = iree_uk_type_size(elem_type);
  iree_uk_ssize_t outer_size0 = params->in_size0;
  iree_uk_ssize_t outer_size1 = params->in_size1;
  iree_uk_ssize_t tile_size0 = params->in_size2;
  iree_uk_ssize_t tile_size1 = params->in_size3;
  iree_uk_ssize_t in_stride_outer0 = params->in_stride0;
  iree_uk_ssize_t in_stride_outer1 = params->in_size3 * params->in_size2;
  iree_uk_ssize_t in_stride_tile0 = params->in_size3;
  iree_uk_ssize_t in_stride_tile1 = 1;
  if (params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER) {
    iree_uk_ssize_swap(&outer_size0, &outer_size1);
    iree_uk_ssize_swap(&in_stride_outer0, &in_stride_outer1);
  }
  if (params->flags & IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER) {
    iree_uk_ssize_swap(&tile_size0, &tile_size1);
    iree_uk_ssize_swap(&in_stride_tile0, &in_stride_tile1);
  }
  for (iree_uk_ssize_t outer_i0 = 0; outer_i0 < outer_size0; ++outer_i0) {
    for (iree_uk_ssize_t outer_i1 = 0; outer_i1 < outer_size1; ++outer_i1) {
      for (iree_uk_ssize_t tile_i0 = 0; tile_i0 < tile_size0; ++tile_i0) {
        for (iree_uk_ssize_t tile_i1 = 0; tile_i1 < tile_size1; ++tile_i1) {
          iree_uk_ssize_t in_offset =
              outer_i0 * in_stride_outer0 + tile_i0 * in_stride_tile0 +
              outer_i1 * in_stride_outer1 + tile_i1 * in_stride_tile1;
          iree_uk_ssize_t i0 = outer_i0 * tile_size0 + tile_i0;
          iree_uk_ssize_t i1 = outer_i1 * tile_size1 + tile_i1;
          if (!(i0 >= params->out_size0 || i1 >= params->out_size1)) {
            iree_uk_ssize_t out_offset = i1 + i0 * params->out_stride0;
            const char* in_ptr =
                ((char*)params->in_buffer) + in_offset * elem_size;
            char* out_ptr =
                ((char*)params->out_buffer) + out_offset * elem_size;
            iree_uk_memcpy(out_ptr, in_ptr, elem_size);
          }
        }
      }
    }
  }
}

static void iree_unpack_actual(const iree_uk_unpack_params_t* params,
                               bool typed_entry_point) {
  if (!typed_entry_point) {
    iree_uk_unpack(params);
  } else if (params->type == iree_uk_unpack_type_f32f32) {
    iree_uk_unpack_f32f32_params_t p = {
        .in_buffer_base = ((const float*)params->in_buffer) - 1,
        .in_buffer_offset = 1,
        .out_buffer_base = ((float*)params->out_buffer) - 2,
        .out_buffer_offset = 2,
        .in_stride0 = params->in_stride0,
        .out_stride0 = params->out_stride0,
        .in_size0 = params->in_size0,
        .in_size1 = params->in_size1,
        .in_size2 = params->in_size2,
        .in_size3 = params->in_size3,
        .out_size0 = params->out_size0,
        .out_size1 = params->out_size1,
        .flags = params->flags,
        .cpu_data = params->cpu_data};
    iree_uk_unpack_f32f32(&p);
  } else if (params->type == iree_uk_unpack_type_i32i32) {
    iree_uk_unpack_i32i32_params_t p = {
        .in_buffer_base = ((const iree_uk_int32_t*)params->in_buffer) - 1,
        .in_buffer_offset = 1,
        .out_buffer_base = ((iree_uk_int32_t*)params->out_buffer) - 2,
        .out_buffer_offset = 2,
        .in_stride0 = params->in_stride0,
        .out_stride0 = params->out_stride0,
        .in_size0 = params->in_size0,
        .in_size1 = params->in_size1,
        .in_size2 = params->in_size2,
        .in_size3 = params->in_size3,
        .out_size0 = params->out_size0,
        .out_size1 = params->out_size1,
        .flags = params->flags,
        .cpu_data = params->cpu_data};
    iree_uk_unpack_i32i32(&p);
  }
}

static void iree_uk_test_unpack_for_shape_params(
    iree_uk_test_t* test, const iree_uk_unpack_params_t* src_params) {
  iree_uk_unpack_params_t params;
  memcpy(&params, src_params, sizeof params);
  // Populate strides first - we need them below to compute buffer lengths.
  // Randomly make strides either tight or not to exercise all cases.
  iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
  params.out_stride0 = params.out_size1 + iree_uk_random_engine_get_0_1(engine);
  params.in_stride0 = params.in_size1 * params.in_size2 * params.in_size3 +
                      iree_uk_random_engine_get_0_1(engine);
  iree_uk_type_t in_type = iree_uk_unpack_in_type(params.type);
  iree_uk_ssize_t in_buffer_size =
      iree_uk_2d_buffer_length(in_type, params.in_size0, params.in_stride0);
  void* in_buffer = malloc(in_buffer_size);
  iree_uk_write_random_buffer(in_buffer, in_buffer_size, in_type, engine);
  params.in_buffer = in_buffer;

  iree_uk_unpack_params_t reference_params;
  memcpy(&reference_params, &params, sizeof reference_params);
  iree_uk_type_t out_type = iree_uk_unpack_out_type(params.type);
  iree_uk_ssize_t out_buffer_size =
      iree_uk_2d_buffer_length(out_type, params.out_size0, params.out_stride0);
  reference_params.out_buffer = malloc(out_buffer_size);
  iree_uk_write_random_buffer(reference_params.out_buffer, out_buffer_size,
                              out_type, engine);

  iree_uk_unpack_params_t actual_params;
  memcpy(&actual_params, &params, sizeof actual_params);
  actual_params.out_buffer = malloc(out_buffer_size);
  iree_uk_write_random_buffer(actual_params.out_buffer, out_buffer_size,
                              out_type, engine);

  iree_unpack_reference(&reference_params);
  iree_unpack_actual(&actual_params, iree_uk_random_engine_get_0_1(engine));

  if (!iree_uk_2d_buffers_equal(
          actual_params.out_buffer, reference_params.out_buffer, out_type,
          params.out_size0, params.out_size1, params.out_stride0)) {
    IREE_UK_TEST_FAIL(test);
  }

  free(reference_params.out_buffer);
  free(actual_params.out_buffer);
  free(in_buffer);
}

static void iree_uk_test_unpack_for_tile_params(iree_uk_test_t* test,
                                                const void* src_params) {
  typedef struct outer_shape_t {
    int size0, size1;
  } outer_shape_t;
  const outer_shape_t outer_shapes[] = {
      // Degenerate cases. Vacuous.
      {0, 1},
      {1, 0},
      // Non-degenerate cases.
      {1, 1},
      {3, 2},
      {9, 33},
  };
  typedef enum {
    pad_none,
    pad_one_incomplete_tile,
    pad_a_lot,
    pad_enum_end
  } pad_t;
  for (int i = 0; i < IREE_ARRAYSIZE(outer_shapes); ++i) {
    for (int transpose_inner = 0; transpose_inner <= 1; ++transpose_inner) {
      for (int transpose_outer = 0; transpose_outer <= 1; ++transpose_outer) {
        for (pad_t pad = 0; pad < pad_enum_end; ++pad) {
          iree_uk_unpack_params_t params;
          memcpy(&params, src_params, sizeof params);
          params.cpu_data = iree_uk_test_cpu_data(test);
          outer_shape_t outer_shape = outer_shapes[i];
          iree_uk_ssize_t in_size0 = outer_shape.size0;
          iree_uk_ssize_t in_size1 = outer_shape.size1;
          params.in_size0 = in_size0;
          params.in_size1 = in_size1;
          if (pad == pad_a_lot) {
            params.in_size0 += 16;
            params.in_size1 += 16;
          }
          iree_uk_ssize_t tile_size0 = params.in_size2;
          iree_uk_ssize_t tile_size1 = params.in_size3;
          params.flags = 0;
          if (transpose_outer) {
            params.flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_OUTER;
            iree_uk_ssize_swap(&in_size0, &in_size1);
          }
          if (transpose_inner) {
            params.flags |= IREE_UK_FLAG_UNPACK_TRANSPOSE_INNER;
            iree_uk_ssize_swap(&tile_size0, &tile_size1);
          }
          params.out_size0 = in_size0 * tile_size0;
          params.out_size1 = in_size1 * tile_size1;
          if (pad == pad_one_incomplete_tile) {
            iree_uk_random_engine_t* engine = iree_uk_test_random_engine(test);
            iree_uk_ssize_t pad_size0 =
                iree_uk_random_engine_get_0_65535(engine) % tile_size0;
            iree_uk_ssize_t pad_size1 =
                iree_uk_random_engine_get_0_65535(engine) % tile_size1;
            params.out_size0 = params.out_size0 - pad_size0;
            if (params.out_size0 < 0) params.out_size0 = 0;
            params.out_size1 = params.out_size1 - pad_size1;
            if (params.out_size1 < 0) params.out_size1 = 0;
          }
          iree_uk_test_unpack_for_shape_params(test, &params);
        }
      }
    }
  }
}

static void iree_uk_test_unpack(iree_uk_unpack_type_t type, int tile_size0,
                                int tile_size1, const char* cpu_features) {
  iree_uk_unpack_params_t params = {
      .type = type, .in_size2 = tile_size0, .in_size3 = tile_size1};
  char types_str[32];
  iree_uk_type_pair_str(types_str, sizeof types_str, type);
  char test_label_str[256];
  snprintf(test_label_str, sizeof test_label_str, "types:%s tile:%dx%d",
           types_str, tile_size0, tile_size1);
  iree_uk_test(test_label_str, iree_uk_test_unpack_for_tile_params, &params,
               cpu_features);
}

int main(int argc, char** argv) {
  // Generic tests, not matching any particular CPU feature. This is the place
  // to test weird tile shapes to ensure e.g. that we haven't unwittingly baked
  // in a power-of-two assumption
  iree_uk_test_unpack(iree_uk_unpack_type_f32f32, 3, 5, NULL);
  iree_uk_test_unpack(iree_uk_unpack_type_i32i32, 3, 4, NULL);

#if defined(IREE_UK_ARCH_ARM_64)
  iree_uk_test_unpack(iree_uk_unpack_type_f32f32, 8, 8, NULL);
  iree_uk_test_unpack(iree_uk_unpack_type_i32i32, 8, 8, NULL);
#elif defined(IREE_UK_ARCH_X86_64)
  iree_uk_test_unpack(iree_uk_unpack_type_f32f32, 8, 8, "avx2_fma");
  iree_uk_test_unpack(iree_uk_unpack_type_i32i32, 8, 8, "avx2_fma");
  iree_uk_test_unpack(iree_uk_unpack_type_f32f32, 16, 16, "avx512_base");
  iree_uk_test_unpack(iree_uk_unpack_type_i32i32, 16, 16, "avx512_base");
#endif  // defined(IREE_UK_ARCH_ARM_64)
}
