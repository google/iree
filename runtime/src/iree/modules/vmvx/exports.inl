// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
//         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
//         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
//         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
//          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
//
//===----------------------------------------------------------------------===//
//
// This file matches the vmvx.imports.mlir in the compiler. It'd be nice to
// autogenerate this as the order of these functions must be sorted ascending by
// name in a way compatible with iree_string_view_compare.
//
// Users are meant to `#define EXPORT_FN` to be able to access the information.
// #define EXPORT_FN(name, target_fn, arg_struct, arg_type, ret_type)

// clang-format off

EXPORT_FN("absf.2d.x32", iree_ukernel_x32u_absf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("addf.2d.x32", iree_ukernel_x32b_addf_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("addi.2d.x32", iree_ukernel_x32b_addi_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("andi.2d.x32", iree_ukernel_x32b_andi_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("ceilf.2d.x32", iree_ukernel_x32u_ceilf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x16", iree_vmvx_copy2d_x16, unary2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x32", iree_vmvx_copy2d_x32, unary2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x64", iree_vmvx_copy2d_x64, unary2d, rIIIrIIIII, v)
EXPORT_FN("copy.2d.x8", iree_vmvx_copy2d_x8, unary2d, rIIIrIIIII, v)
EXPORT_FN("ctlz.2d.x32", iree_ukernel_x32u_ctlz_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("divf.2d.x32", iree_ukernel_x32b_divf_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("divsi.2d.x32", iree_ukernel_x32b_divsi_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("divui.2d.x32", iree_ukernel_x32b_divui_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("expf.2d.x32", iree_ukernel_x32u_expf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("fill.2d.x32", iree_vmvx_fill2d_x32, fill2d_x32, irIIII, v)
EXPORT_FN("floorf.2d.x32", iree_ukernel_x32u_floorf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("logf.2d.x32", iree_ukernel_x32u_logf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("matmul.f32f32f32", iree_vmvx_matmul_f32f32f32, matmul_f32, rIIrIIrIIIIIffi, v)
// NOTE: must still be in alphabetical order with all other exports.
#if defined(IREE_HAVE_MMT4D_BUILTINS)
EXPORT_FN("mmt4d.f32f32f32", iree_vmvx_mmt4d_f32f32f32, mmt4d_f32, rIIrIIrIIIIIffi, v)
#endif  // IREE_HAVE_MMT4D_BUILTINS
EXPORT_FN("mulf.2d.x32", iree_ukernel_x32b_mulf_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("muli.2d.x32", iree_ukernel_x32b_muli_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("negf.2d.x32", iree_ukernel_x32u_negf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("ori.2d.x32", iree_ukernel_x32b_ori_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("rsqrtf.2d.x32", iree_ukernel_x32u_rsqrtf_2d, ukernel_x32u_2d, rIIIrIIIII, v)
EXPORT_FN("shli.2d.x32", iree_ukernel_x32b_shli_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("shrsi.2d.x32", iree_ukernel_x32b_shrsi_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("shrui.2d.x32", iree_ukernel_x32b_shrui_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("subf.2d.x32", iree_ukernel_x32b_subf_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("subi.2d.x32", iree_ukernel_x32b_subi_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)
EXPORT_FN("xori.2d.x32", iree_ukernel_x32b_xori_2d, ukernel_x32b_2d, rIIIrIIIrIIIII, v)

// clang-format on
