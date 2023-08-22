## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE RISC-V benchmarks."""

from typing import List

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.device_specs import riscv_specs
from e2e_test_framework.models import tflite_models


class Linux_RV64_Benchmarks(object):
    """Benchmarks RV64 on Linux devices."""

    RV64_CPU_TARGET = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
        target_abi=iree_definitions.TargetABI.LINUX_GNU,
    )
    DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_LINUX_RV64_GENERIC_DEFAULTS,
        tags=["default-flags"],
        compile_targets=[RV64_CPU_TARGET],
    )
    MODELS = [
        # TODO(#14775): Re-enable the benchmarks.
        # tflite_models.DEEPLABV3_FP32,
        tflite_models.MOBILEBERT_FP32,
        # tflite_models.MOBILENET_V1,
        # tflite_models.MOBILEBERT_INT8,
        # tflite_models.PERSON_DETECT_INT8,
        # tflite_models.EFFICIENTNET_INT8,
        tflite_models.MOBILENET_V2_INT8,
    ]

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        """Generates IREE compile and run configs."""
        gen_configs = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DEFAULT_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.MODELS
        ]
        run_configs = utils.generate_e2e_model_run_configs(
            module_generation_configs=gen_configs,
            module_execution_configs=[module_execution_configs.ELF_LOCAL_SYNC_CONFIG],
            device_specs=[riscv_specs.EMULATOR_RISCV_64],
            presets=[benchmark_presets.RISCV],
        )
        return run_configs


class Linux_RV32_Benchmarks(object):
    """Benchmarks RV32 on Linux devices."""

    RV32_CPU_TARGET = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.LLVM_CPU,
        target_architecture=common_definitions.DeviceArchitecture.RV32_GENERIC,
        target_abi=iree_definitions.TargetABI.LINUX_GNU,
    )
    DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_LINUX_RV32_GENERIC_DEFAULTS,
        tags=["default-flags"],
        compile_targets=[RV32_CPU_TARGET],
    )
    MODELS = [
        # TODO(#14775): Re-enable the benchmarks.
        # tflite_models.EFFICIENTNET_INT8,
        # tflite_models.MOBILEBERT_INT8,
        # tflite_models.PERSON_DETECT_INT8,
        tflite_models.MOBILENET_V2_INT8,
    ]

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        """Generates IREE compile and run configs."""
        gen_configs = [
            iree_definitions.ModuleGenerationConfig.build(
                compile_config=self.DEFAULT_COMPILE_CONFIG,
                imported_model=iree_definitions.ImportedModel.from_model(model),
            )
            for model in self.MODELS
        ]
        run_configs = utils.generate_e2e_model_run_configs(
            module_generation_configs=gen_configs,
            module_execution_configs=[module_execution_configs.ELF_LOCAL_SYNC_CONFIG],
            device_specs=[riscv_specs.EMULATOR_RISCV_32],
            presets=[benchmark_presets.RISCV],
        )
        return run_configs
