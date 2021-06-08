#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# For use within a IREE bazel-swiftshader docker image on a Kokoro VM.
# Log some information about the environment, initialize the submodules and then
# run the bazel integrations tests.

set -x

# Print Vulkan related information: SDK version and GPU ICD version
vulkaninfo 2> /tmp/vulkaninfo.stderr 1> /tmp/vulkaninfo.stdout
VULKAN_INSTANCE="$(grep "Vulkan Instance" /tmp/vulkaninfo.stdout)"
VK_PHYSICAL_DEVICE_PROPERTIES="$(grep -A7 "VkPhysicalDeviceProperties" /tmp/vulkaninfo.stdout)"

if [[ -z "${VULKAN_INSTANCE?}" ]] || [[ -z "${VK_PHYSICAL_DEVICE_PROPERTIES?}" ]]; then
  echo "Vulkan not found!"
  cat /tmp/vulkaninfo.stdout
  cat /tmp/vulkaninfo.stderr
  exit 1
fi

echo "${VULKAN_INSTANCE?}"
echo "${VK_PHYSICAL_DEVICE_PROPERTIES?}"
