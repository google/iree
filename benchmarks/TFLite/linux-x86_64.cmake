# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
#                                                                              #
# Default benchmark configurations                                             #
#                                                                              #
# Each suite benchmarks a list of modules with configurations specifying a     #
# target architecture and runtime characteristics (e.g. threads/cores). These  #
# benchmarks only configure IREE translation and runtime flags for the target  #
# architecture and do *not* include any non-default flags. No non-default      #
# flags should be added here.                                                  #
#                                                                              #
################################################################################

set(LINUX_X86_64_CASCADELAKE_CPU_TRANSLATION_FLAGS
  "--iree-input-type=tosa"
  "--iree-llvm-target-cpu=cascadelake"
  "--iree-llvm-target-triple=x86_64-unknown-linux-gnu"
)

# CPU, Dylib-Sync, x86_64, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  TRANSLATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib-sync"
)

# CPU, Dylib, 1 thread, x86_64, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "1-thread,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  TRANSLATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=1"
)

# CPU, Dylib, 4 threads, x86_64, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "4-thread,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  TRANSLATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=4"
)

# CPU, Dylib, 8 threads, x86_64, full-inference
iree_benchmark_suite(
  MODULES
    "${DEEPLABV3_FP32_MODULE}"
    "${MOBILESSD_FP32_MODULE}"
    "${POSENET_FP32_MODULE}"
    "${MOBILEBERT_FP32_MODULE}"
    "${MOBILENET_V2_MODULE}"
    "${MOBILENET_V3SMALL_MODULE}"

  BENCHMARK_MODES
    "8-thread,full-inference,default-flags"
  TARGET_BACKEND
    "dylib-llvm-aot"
  TARGET_ARCHITECTURE
    "CPU-x86_64-CascadeLake"
  TRANSLATION_FLAGS
    ${LINUX_X86_64_CASCADELAKE_CPU_TRANSLATION_FLAGS}
  BENCHMARK_TOOL
    iree-benchmark-module
  DRIVER
    "dylib"
  RUNTIME_FLAGS
    "--task_topology_group_count=8"
)
