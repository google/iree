// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/status_cc.h"

#include <cstddef>
#include <cstdlib>

#include "iree/base/attributes.h"

namespace iree {

// static
IREE_MUST_USE_RESULT std::string Status::ToString(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return "OK";
  }
  iree_host_size_t buffer_length = 0;
  if (IREE_UNLIKELY(!iree_status_format(status, /*buffer_capacity=*/0,
                                        /*buffer=*/NULL, &buffer_length))) {
    return "<!>";
  }
  std::string result(buffer_length, '\0');
  if (IREE_UNLIKELY(!iree_status_format(status, result.size() + 1,
                                        const_cast<char*>(result.data()),
                                        &buffer_length))) {
    return "<!>";
  }
  return result;
}

namespace status_impl {

void Helper::HandleInvalidStatusCtorArg(Status* status) {
  fprintf(stderr,
          "An OK status is not a valid constructor argument to StatusOr<T>\n");
  abort();
}

void Helper::Crash(const Status& status) {
  std::string message = status.ToString();
  fprintf(stderr, "Attempting to fetch value instead of handling error:\n%s\n",
          message.c_str());
  abort();
}

}  // namespace status_impl

}  // namespace iree
