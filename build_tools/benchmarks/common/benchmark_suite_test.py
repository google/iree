#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import tempfile
import unittest
from typing import Sequence
from common.benchmark_definition import IREE_DRIVERS_INFOS
from common.benchmark_suite import BenchmarkCase, BenchmarkSuite


class BenchmarkSuiteTest(unittest.TestCase):

  def test_list_categories(self):
    suite = BenchmarkSuite({
        "suite/TFLite": [],
        "suite/PyTorch": [],
    })

    self.assertEqual(suite.list_categories(), [("PyTorch", "suite/PyTorch"),
                                               ("TFLite", "suite/TFLite")])

  def test_filter_benchmarks_for_category(self):
    case1 = BenchmarkCase(model_name="deepnet",
                          model_tags=[],
                          bench_mode=["1-thread", "full-inference"],
                          target_arch="CPU-ARMv8",
                          driver_info=IREE_DRIVERS_INFOS["iree-dylib"],
                          benchmark_case_dir="case1",
                          benchmark_tool_name="tool")
    case2 = BenchmarkCase(model_name="deepnetv2",
                          model_tags=["f32"],
                          bench_mode=["full-inference"],
                          target_arch="GPU-Mali",
                          driver_info=IREE_DRIVERS_INFOS["iree-vulkan"],
                          benchmark_case_dir="case2",
                          benchmark_tool_name="tool")
    case3 = BenchmarkCase(model_name="deepnetv3",
                          model_tags=["f32"],
                          bench_mode=["full-inference"],
                          target_arch="CPU-x86_64",
                          driver_info=IREE_DRIVERS_INFOS["iree-dylib-sync"],
                          benchmark_case_dir="case3",
                          benchmark_tool_name="tool")
    suite = BenchmarkSuite({
        "suite/TFLite": [case1, case2, case3],
    })

    cpu_and_gpu_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=["local-task", "vulkan"],
        available_loaders=["embedded-elf"],
        cpu_target_arch_filter="cpu-armv8",
        gpu_target_arch_filter="gpu-mali",
        driver_filter=None,
        mode_filter=".*full-inference.*",
        model_name_filter="deepnet.*")
    gpu_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=["local-task", "vulkan"],
        available_loaders=["embedded-elf"],
        cpu_target_arch_filter="cpu-unknown",
        gpu_target_arch_filter="gpu-mali",
        driver_filter="vulkan",
        mode_filter=".*full-inference.*",
        model_name_filter="deepnet.*/case2")
    all_benchmarks = suite.filter_benchmarks_for_category(
        category="TFLite",
        available_drivers=None,
        cpu_target_arch_filter=None,
        gpu_target_arch_filter=None,
        driver_filter=None,
        mode_filter=None,
        model_name_filter=None)

    self.assertEqual(cpu_and_gpu_benchmarks, [case1, case2])
    self.assertEqual(gpu_benchmarks, [case2])
    self.assertEqual(all_benchmarks, [case1, case2, case3])

  def test_filter_benchmarks_for_nonexistent_category(self):
    suite = BenchmarkSuite({
        "suite/TFLite": [],
    })

    benchmarks = suite.filter_benchmarks_for_category(
        category="PyTorch",
        available_drivers=[],
        available_loaders=[],
        cpu_target_arch_filter="ARMv8",
        gpu_target_arch_filter="Mali-G78")

    self.assertEqual(benchmarks, [])

  def test_load_from_benchmark_suite_dir(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      tflite_dir = os.path.join(tmp_dir, "TFLite")
      pytorch_dir = os.path.join(tmp_dir, "PyTorch")
      BenchmarkSuiteTest.__create_bench(tflite_dir,
                                        model_name="DeepNet",
                                        model_tags=["f32"],
                                        bench_mode=["4-thread", "full"],
                                        target_arch="CPU-ARMv8",
                                        config="iree-dylib",
                                        tool="run-cpu-bench")
      case2 = BenchmarkSuiteTest.__create_bench(pytorch_dir,
                                                model_name="DeepNetv2",
                                                model_tags=[],
                                                bench_mode=["full-inference"],
                                                target_arch="GPU-Mali",
                                                config="iree-vulkan",
                                                tool="run-gpu-bench")

      suite = BenchmarkSuite.load_from_benchmark_suite_dir(tmp_dir)

      self.assertEqual(suite.list_categories(), [("PyTorch", pytorch_dir),
                                                 ("TFLite", tflite_dir)])
      self.assertEqual(
          suite.filter_benchmarks_for_category(
              category="PyTorch",
              available_drivers=["vulkan"],
              available_loaders=[],
              cpu_target_arch_filter="cpu-armv8",
              gpu_target_arch_filter="gpu-mali"), [case2])

  @staticmethod
  def __create_bench(dir_path: str, model_name: str, model_tags: Sequence[str],
                     bench_mode: Sequence[str], target_arch: str, config: str,
                     tool: str):
    case_name = f"{config}__{target_arch}__{','.join(bench_mode)}"
    model_name_with_tags = model_name
    if len(model_tags) > 0:
      model_name_with_tags += f"-{','.join(model_tags)}"
    bench_path = os.path.join(dir_path, model_name_with_tags, case_name)
    os.makedirs(bench_path)
    with open(os.path.join(bench_path, "tool"), "w") as f:
      f.write(tool)

    return BenchmarkCase(model_name=model_name,
                         model_tags=model_tags,
                         bench_mode=bench_mode,
                         target_arch=target_arch,
                         driver_info=IREE_DRIVERS_INFOS[config],
                         benchmark_case_dir=bench_path,
                         benchmark_tool_name=tool)


if __name__ == "__main__":
  unittest.main()
