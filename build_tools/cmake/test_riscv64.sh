#!/bin/bash

# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Test the cross-compiled RISCV 64-bit Linux targets.

set -xeuo pipefail

# Print the UTC time when set -x is on
export PS4='[$(date -u "+%T %Z")] '

ROOT_DIR="${ROOT_DIR:-$(git rev-parse --show-toplevel)}"
IREE_IMPORT_TFLITE_BIN="${IREE_IMPORT_TFLITE_BIN:-iree-import-tflite}"
LLVM_BIN_DIR="${LLVM_BIN_DIR}"

# Environment variable used by the emulator and iree-compile for the
# llvm-cpu bytecode codegen.
export RISCV_TOOLCHAIN_ROOT="${RISCV_RV64_LINUX_TOOLCHAIN_ROOT}"

function generate_llvm_cpu_vmfb {
  local target="${1}"; shift
  local compile_args=(
    --iree-hal-target-backends=llvm-cpu
    --iree-llvm-embedded-linker-path="${IREE_HOST_BINARY_ROOT}/bin/lld"
    --iree-llvm-target-triple=riscv64
    --iree-llvm-target-cpu=generic-rv64
    --iree-llvm-target-abi=lp64d
  )
  if [[ "${target}" == "mhlo" ]]; then
    compile_args+=(
      --iree-input-type=mhlo
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
    )
  elif [[ "${target}" == "tosa" ]]; then
    local input_file="${1}"; shift
    "${IREE_IMPORT_TFLITE_BIN}" -o "${BUILD_RISCV_DIR}/tosa.mlir" "${input_file}"
    compile_args+=(
      --iree-input-type=tosa
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c"
      "${BUILD_RISCV_DIR}/tosa.mlir"
    )
  elif [[ "${target}" == "tosa-rvv" ]]; then
    local input_file="${1}"; shift
    "${IREE_IMPORT_TFLITE_BIN}" -o "${BUILD_RISCV_DIR}/tosa.mlir" "${input_file}"
    compile_args+=(
      --iree-input-type=tosa
      --iree-llvm-target-cpu-features="+m,+a,+f,+d,+c,+v"
      --riscv-v-fixed-length-vector-lmul-max=8
      --riscv-v-vector-bits-min=512
      "${BUILD_RISCV_DIR}/tosa.mlir"
    )
  fi
  "${IREE_HOST_BINARY_ROOT}/bin/iree-compile" "${compile_args[@]}" "$@"
}

generate_llvm_cpu_vmfb mhlo \
  "${ROOT_DIR}/tools/test/iree-run-module.mlir" \
  -o "${BUILD_RISCV_DIR}/iree-run-module-llvm_cpu.vmfb"

wget -P "${BUILD_RISCV_DIR}/" "https://storage.googleapis.com/iree-model-artifacts/person_detect.tflite"

generate_llvm_cpu_vmfb tosa \
  "${BUILD_RISCV_DIR}/person_detect.tflite" \
  -o "${BUILD_RISCV_DIR}/person_detect.vmfb"

generate_llvm_cpu_vmfb tosa-rvv \
  "${BUILD_RISCV_DIR}/person_detect.tflite" \
  -o "${BUILD_RISCV_DIR}/person_detect_rvv.vmfb"

${PYTHON_BIN} "${ROOT_DIR}/third_party/llvm-project/llvm/utils/lit/lit.py" \
  -v --path "${LLVM_BIN_DIR}" "${ROOT_DIR}/tests/riscv64"

# Test e2e models. Excluding mobilebert for now.
ctest --test-dir ${BUILD_RISCV_DIR}/tests/e2e/models -R llvm-cpu_local-task_mobilenet -E bert
# Test all tosa ops
ctest --test-dir ${BUILD_RISCV_DIR}/tests/e2e/tosa_ops -R check_llvm-cpu_local-task
# Test all xla ops except fp16, which is not supported properly
ctest --test-dir ${BUILD_RISCV_DIR}/tests/e2e/xla_ops -R check_llvm-cpu_local-task -E fp16
