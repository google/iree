## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE x86_64 benchmarks."""

from typing import List, Tuple
from e2e_test_framework.device_specs import device_collections
from e2e_test_framework.models import model_groups, tf_models, torch_models
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework import unique_ids
from benchmark_suites.iree import module_execution_configs
import benchmark_suites.iree.utils


class Linux_x86_64_Benchmarks(object):
  """Benchmarks on x86_64 linux devices."""

  CASCADELAKE_CPU_TARGET = iree_definitions.CompileTarget(
      target_architecture=common_definitions.DeviceArchitecture.
      X86_64_CASCADELAKE,
      target_backend=iree_definitions.TargetBackend.LLVM_CPU,
      target_abi=iree_definitions.TargetABI.LINUX_GNU)

  CASCADELAKE_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE,
      tags=["default-flags"],
      compile_targets=[CASCADELAKE_CPU_TARGET])
  CASCADELAKE_FUSE_PADDING_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
      id=unique_ids.IREE_COMPILE_CONFIG_LINUX_CASCADELAKE_FUSE_PADDING,
      tags=["experimental-flags", "fuse-padding"],
      compile_targets=[CASCADELAKE_CPU_TARGET],
      extra_flags=[
          "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
          "--iree-llvmcpu-enable-pad-consumer-fusion"
      ])

  def _generate_default(
      self, device_specs: List[common_definitions.DeviceSpec]
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    # Create benchmarks for small workloads.
    gen_configs_small = [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=self.CASCADELAKE_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in model_groups.SMALL
    ]
    execution_configs_small = [
        module_execution_configs.get_elf_local_task_config(1),
        module_execution_configs.get_elf_local_task_config(8)
    ]
    run_configs_small = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs_small,
        module_execution_configs=execution_configs_small,
        device_specs=device_specs)

    # Create benchmarks for large workloads.
    gen_configs_large = [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=self.CASCADELAKE_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in model_groups.LARGE
    ]
    # We use higher thread counts for large workloads.
    execution_configs_large = [
        module_execution_configs.get_elf_local_task_config(1),
        module_execution_configs.get_elf_local_task_config(8),
        # We use (MAX_CORE - 2) here to determine the maximum number of threads to use on a NUMA node.
        # 13 threads are currently disabled until we switch to using c2-standard-60.
        # module_execution_configs.get_elf_local_task_config(13),
    ]
    run_configs_large = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs_large,
        module_execution_configs=execution_configs_large,
        device_specs=device_specs)

    return (gen_configs_small + gen_configs_large,
            run_configs_small + run_configs_large)

  def _generate_experimental(
      self, device_specs: List[common_definitions.DeviceSpec]
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    excluded_models_for_experiments = [
        # Disabled due to https://github.com/openxla/iree/issues/11174.
        tf_models.RESNET50_TF_FP32,
        # Disabled due to https://github.com/openxla/iree/issues/12772.
        torch_models.EFFICIENTNET_V2_S_FP32_TORCH,
    ]
    gen_configs = [
        iree_definitions.ModuleGenerationConfig.build(
            compile_config=self.CASCADELAKE_FUSE_PADDING_COMPILE_CONFIG,
            imported_model=iree_definitions.ImportedModel.from_model(model))
        for model in model_groups.SMALL + model_groups.LARGE
        if model not in excluded_models_for_experiments
    ]
    # We run the fuse padding compiler config under 4 threads only.
    default_execution_configs = [
        module_execution_configs.get_elf_local_task_config(4)
    ]

    run_configs = benchmark_suites.iree.utils.generate_e2e_model_run_configs(
        module_generation_configs=gen_configs,
        module_execution_configs=default_execution_configs,
        device_specs=device_specs)

    return (gen_configs, run_configs)

  def generate(
      self
  ) -> Tuple[List[iree_definitions.ModuleGenerationConfig],
             List[iree_definitions.E2EModelRunConfig]]:
    """Generates IREE compile and run configs."""

    cascadelake_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
        architecture=common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
        host_environment=common_definitions.HostEnvironment.LINUX_X86_64)

    default_gen_configs, default_run_configs = self._generate_default(
        cascadelake_devices)
    experimental_gen_configs, experimental_run_configs = self._generate_experimental(
        cascadelake_devices)

    return (default_gen_configs + experimental_gen_configs,
            default_run_configs + experimental_run_configs)


def generate() -> Tuple[List[iree_definitions.ModuleGenerationConfig],
                        List[iree_definitions.E2EModelRunConfig]]:
  """Generates all compile and run configs for IREE benchmarks."""
  return Linux_x86_64_Benchmarks().generate()
