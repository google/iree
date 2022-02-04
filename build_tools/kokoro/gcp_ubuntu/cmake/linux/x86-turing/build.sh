#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build the project with cmake using Kokoro.

set -e
set -x

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

# Check these exist and print the versions for later debugging
export CMAKE_BIN="$(which cmake)"
"${CMAKE_BIN?}" --version
"${CC?}" --version
"${CXX?}" --version
python3 --version

echo "Initializing submodules"
git submodule update --init --jobs 8 --depth 1

# Print NVIDIA GPU information inside the docker
dpkg -l | grep nvidia
nvidia-smi || true

./build_tools/kokoro/gcp_ubuntu/check_vulkan.sh

# TODO(gcmn): It would be nice to be able to build and test as much as possible,
# so a build failure only prevents building/testing things that depend on it and
# we can still run the other tests.
# TODO: Add "-DIREE_TARGET_BACKEND_CUDA=ON -DIREE_HAL_DRIVER_CUDA=ON" once the
# VMs have been updated with the correct CUDA SDK.
echo "Building with cmake"
./build_tools/cmake/clean_build.sh

export IREE_VULKAN_F16_DISABLE=0
export IREE_CUDA_DISABLE=0
echo "Testing with ctest"
./build_tools/cmake/test.sh
