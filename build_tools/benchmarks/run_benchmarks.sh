#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script to run benchmarks on CI with the proper docker image and benchmark tool
# based on the IREE_DEVICE_NAME. This script can also run locally, but some
# devices require docker to run benchmarks. By default it uses the wrapper
# build_tools/docker/docker_run.sh if IREE_DOCKER_WRAPPER is not specified. See
# the script to learn about the required setup.
#
# IREE_NORMAL_BENCHMARK_TOOLS_DIR needs to point to a directory contains IREE
# benchmark tools. See benchmarks/README.md for more information. The first
# argument is the path of e2e test artifacts directory. The second argument is
# the path of IREE benchmark run config. The third argument is the path to write
# benchmark results.

set -euo pipefail

DOCKER_WRAPPER="${IREE_DOCKER_WRAPPER:-./build_tools/docker/docker_run.sh}"
NORMAL_BENCHMARK_TOOLS_DIR="${IREE_NORMAL_BENCHMARK_TOOLS_DIR}"
TRACED_BENCHMARK_TOOLS_DIR="${IREE_TRACED_BENCHMARK_TOOLS_DIR}"
TRACY_CAPTURE_TOOL="${IREE_TRACY_CAPTURE_TOOL}"
E2E_TEST_ARTIFACTS_DIR="${1:-${IREE_E2E_TEST_ARTIFACTS_DIR}}"
EXECUTION_BENCHMARK_CONFIG="${2:-${IREE_EXECUTION_BENCHMARK_CONFIG}}"
TARGET_DEVICE_NAME="${3:-${IREE_TARGET_DEVICE_NAME}}"
BENCHMARK_RESULTS="${4:-${IREE_BENCHMARK_RESULTS}}"
BENCHMARK_TRACES="${5:-${IREE_BENCHMARK_TRACES}}"

if [[ "${TARGET_DEVICE_NAME}" == "a2-highgpu-1g" ]]; then
  ${DOCKER_WRAPPER} \
    --gpus all \
    --env NVIDIA_DRIVER_CAPABILITIES=all \
    gcr.io/iree-oss/nvidia-bleeding-edge@sha256:234f6efb1e9fd2eea67d40effd7d9cfe8725c5e5ba7a58c0428200a54493637a \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --traced_benchmark_tool_dir="${TRACED_BENCHMARK_TOOLS_DIR}" \
        --trace_capture_tool="${TRACY_CAPTURE_TOOL}" \
        --capture_tarball="${BENCHMARK_TRACES}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
        --target_device_name="${TARGET_DEVICE_NAME}" \
        --output="${BENCHMARK_RESULTS}" \
        --verbose
elif [[ "${TARGET_DEVICE_NAME}" == "c2-standard-16" ]]; then
  ${DOCKER_WRAPPER} \
    gcr.io/iree-oss/base-bleeding-edge@sha256:7d15ecdb60d65b61426b397382096982103dbd5355867832c91c097a97b925c5 \
      ./build_tools/benchmarks/run_benchmarks_on_linux.py \
        --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
        --traced_benchmark_tool_dir="${TRACED_BENCHMARK_TOOLS_DIR}" \
        --trace_capture_tool="${TRACY_CAPTURE_TOOL}" \
        --capture_tarball="${BENCHMARK_TRACES}" \
        --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
        --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
        --target_device_name="${TARGET_DEVICE_NAME}" \
        --output="${BENCHMARK_RESULTS}" \
        --device_model=GCP-c2-standard-16 \
        --cpu_uarch=CascadeLake \
        --verbose
elif [[ "${TARGET_DEVICE_NAME}" =~ ^(pixel-4|pixel-6-pro|moto-edge-x30)$ ]]; then
  ./build_tools/benchmarks/run_benchmarks_on_android.py \
    --normal_benchmark_tool_dir="${NORMAL_BENCHMARK_TOOLS_DIR}" \
    --traced_benchmark_tool_dir="${TRACED_BENCHMARK_TOOLS_DIR}" \
    --trace_capture_tool="${TRACY_CAPTURE_TOOL}" \
    --capture_tarball="${BENCHMARK_TRACES}" \
    --e2e_test_artifacts_dir="${E2E_TEST_ARTIFACTS_DIR}" \
    --execution_benchmark_config="${EXECUTION_BENCHMARK_CONFIG}" \
    --target_device_name="${TARGET_DEVICE_NAME}" \
    --output="${BENCHMARK_RESULTS}" \
    --pin-cpu-freq \
    --pin-gpu-freq \
    --verbose
else
  echo "${TARGET_DEVICE_NAME} is not supported yet."
  exit 1
fi
