// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/common.h"

IREE_UKERNEL_EXPORT const char* iree_ukernel_status_message(
    iree_ukernel_status_t status) {
  switch (status) {
    case iree_ukernel_status_bad_flags:
      return "bad mmt4d flags";
    case iree_ukernel_status_bad_type:
      return "bad mmt4d type enum";
    case iree_ukernel_status_unsupported_huge_or_negative_dimension:
      return "unsupported huge or negative size in mmt4d";
    case iree_ukernel_status_unsupported_generic_tile_size:
      return "tile size too large for the generic tile implementation";
    default:
      return "unknown";
  }
}
