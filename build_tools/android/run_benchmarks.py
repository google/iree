#!/usr/bin/env python3

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs all matched benchmark suites on an Android device.

This script probes the Android phone via `adb` and uses the device information
to filter and run suitable benchmarks on it.

It expects that `adb` is installed, and there is an `iree-benchmark-module`
tool cross-compiled towards Android. It also expects the benchmark artifacts
are generated by building the `iree-benchmark-suites` target.

Example usages:
  python3 run_benchmarks.py \
    --benchmark_tool=/path/to/android/target/iree-benchmark_module \
    /path/to/host/build/dir
"""

import argparse
import json
import os
import re
import subprocess

from common.benchmark_description import (AndroidDeviceInfo, BenchmarkInfo,
                                          BenchmarkResults, get_output)

# Relative path against build directory.
BENCHMARK_SUITE_REL_PATH = "benchmark_suites"
# Relative path against root benchmark suite directory.
MLIR_MODEL_SUITE_REL_PATH = "mlir_models"

# The flagfile's filename for compiled Python models.
MODEL_FLAGFILE_NAME = "flagfile"
# The artifact's filename for compiled Python models.
MODEL_VMFB_NAME = "compiled.vmfb"

# Root directory to perform benchmarks in on the Android device.
ANDROID_TMP_DIR = "/data/local/tmp/iree-benchmarks"

BENCHMARK_REPETITIONS = 10

# A map from Android CPU ABI to IREE's benchmark target architecture.
CPU_ABI_TO_TARGET_ARCH_MAP = {
    "arm64-v8a": "cpu-arm64-v8a",
}

# A map from Android GPU name to IREE's benchmark target architecture.
GPU_NAME_TO_TARGET_ARCH_MAP = {
    "adreno-640": "gpu-adreno",
    "adreno-650": "gpu-adreno",
    "adreno-660": "gpu-adreno",
    "mali-g77": "gpu-mali-valhall",
    "mali-g78": "gpu-mali-valhall",
}


def get_git_commit_hash(commit):
  return get_output(['git', 'rev-parse', commit],
                    cwd=os.path.dirname(os.path.realpath(__file__)))


def adb_push_to_tmp_dir(content, relative_dir, verbose=False):
  """Pushes content onto the Android device.

  Args:
  - content: the full path to the source file.
  - relative_dir: the directory to push to; relative to ANDROID_TMP_DIR.

  Returns:
  - The full path to the content on the Android device.
  """
  filename = os.path.basename(content)
  android_path = os.path.join(ANDROID_TMP_DIR, relative_dir, filename)
  get_output(
      ["adb", "push", os.path.abspath(content), android_path], verbose=verbose)
  return android_path


def adb_execute_in_dir(cmd_args, relative_dir, verbose=False):
  """Executes command with adb shell in a directory.

  Args:
  - cmd_args: a list containing the command to execute and its parameters
  - relative_dir: the directory to execute the command in; relative to
    ANDROID_TMP_DIR.

  Returns:
  - A string for the command output.
  """
  cmd = ["adb", "shell"]
  cmd.extend(["cd", f"{ANDROID_TMP_DIR}/{relative_dir}"])
  cmd.append("&&")
  cmd.extend(cmd_args)

  return get_output(cmd, verbose=verbose)


def compose_benchmark_info_object(device_info, root_build_dir,
                                  model_benchmark_dir):
  """Creates an BenchmarkInfo object to describe the benchmark.

    Args:
    - device_info: an AndroidDeviceInfo object.
    - root_build_dir: the root build directory.
    - model_benchmark_dir: a directory containing model benchmarks.

    Returns:
    - A BenchmarkInfo object.
  """
  model_root_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH,
                                MLIR_MODEL_SUITE_REL_PATH)

  # Extract the model name from the directory path. This uses the relative
  # path under the root model directory. If there are multiple segments,
  # additional ones will be placed in parentheses.
  model_name = os.path.relpath(model_benchmark_dir, model_root_dir)
  # Now we have <model-name>/.../<iree-driver>__<target-arch>__<bench_mode>,
  # Remove the last segment.
  model_name = os.path.dirname(model_name)
  main, rest = os.path.split(model_name)
  if main:
    # Tags coming from directory structure.
    model_name = main
    model_tags = [re.sub(r"\W+", "-", rest)]
  else:
    # Tags coming from the name itself.
    rest = re.sub(r"\W+", "-", rest).split("-")
    model_name = rest[0]
    model_tags = rest[1:]

  # Extract benchmark info from the directory path following convention:
  #   <iree-driver>__<target-architecture>__<benchmark_mode>
  root_immediate_dir = os.path.basename(model_benchmark_dir)
  iree_driver, target_arch, bench_mode = root_immediate_dir.split("__")

  return BenchmarkInfo(model_name=model_name,
                       model_tags=model_tags,
                       model_source="TensorFlow",
                       bench_mode=bench_mode,
                       runner=iree_driver,
                       device_info=device_info)


def filter_python_model_benchmark_suite(device_info,
                                        root_build_dir,
                                        verbose=False):
  """Filters Python model benchmark suite for the given CPU/GPU target.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - root_build_dir: the root build directory.

  Returns:
  - A list containing all matched benchmark's directories.
  """
  cpu_target_arch = CPU_ABI_TO_TARGET_ARCH_MAP[device_info.cpu_abi.lower()]
  gpu_target_arch = GPU_NAME_TO_TARGET_ARCH_MAP[device_info.gpu_name.lower()]

  model_root_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH,
                                MLIR_MODEL_SUITE_REL_PATH)
  matched_benchmarks = []

  # Go over all benchmarks in the model directory to find those matching the
  # current Android device's CPU/GPU architecture.
  for root, dirs, files in os.walk(model_root_dir):
    # Take the immediate directory name and try to see if it contains compiled
    # models and flagfiles. This relies on the following directory naming
    # convention:
    #   <iree-driver>__<target-architecture>__<benchmark_mode>
    root_immediate_dir = os.path.basename(root)
    segments = root_immediate_dir.split("__")
    if len(segments) != 3 or not segments[0].startswith("iree-"):
      continue

    iree_driver, target_arch, bench_mode = segments
    target_arch = target_arch.lower()
    # We can choose this benchmark if it matches the CPU/GPU architecture.
    should_choose = (target_arch == cpu_target_arch or
                     target_arch == gpu_target_arch)
    if should_choose:
      matched_benchmarks.append(root)

    if verbose:
      print(f"dir: {root}")
      print(f"  iree_driver: {iree_driver}")
      print(f"  target_arch: {target_arch}")
      print(f"  bench_mode: {bench_mode}")
      print(f"  chosen: {should_choose}")

  return matched_benchmarks


def run_python_model_benchmark_suite(device_info,
                                     root_build_dir,
                                     model_benchmark_dirs,
                                     benchmark_tool,
                                     verbose=False):
  """Runs all model benchmarks on the Android device and report results.

  Args:
  - device_info: an AndroidDeviceInfo object.
  - root_build_dir: the root build directory.
  - model_benchmark_dirs: a list of model benchmark directories.
  - benchmark_tool: the path to the benchmark tool.

  Returns:
  - A list containing (BenchmarkInfo, context, results) tuples.
  """
  # Push the benchmark tool to the Android device first.
  android_tool_path = adb_push_to_tmp_dir(benchmark_tool,
                                          relative_dir="tools",
                                          verbose=verbose)

  model_root_dir = os.path.join(root_build_dir, BENCHMARK_SUITE_REL_PATH,
                                MLIR_MODEL_SUITE_REL_PATH)

  results = []

  # Push all model artifacts to the device and run them.
  for model_benchmark_dir in model_benchmark_dirs:
    benchmark_info = compose_benchmark_info_object(device_info, root_build_dir,
                                                   model_benchmark_dir)
    print(f"--> benchmark: {benchmark_info} <--")
    android_relative_dir = os.path.relpath(model_benchmark_dir, model_root_dir)
    adb_push_to_tmp_dir(os.path.join(model_benchmark_dir, MODEL_VMFB_NAME),
                        android_relative_dir,
                        verbose=verbose)
    android_flagfile_path = adb_push_to_tmp_dir(os.path.join(
        model_benchmark_dir, MODEL_FLAGFILE_NAME),
                                                android_relative_dir,
                                                verbose=verbose)

    cmd = [
        android_tool_path,
        f"--flagfile={android_flagfile_path}",
        f"--benchmark_repetitions={BENCHMARK_REPETITIONS}",
        "--benchmark_format=json",
    ]
    resultjson = adb_execute_in_dir(cmd, android_relative_dir, verbose=verbose)

    print(resultjson)
    resultjson = json.loads(resultjson)

    for previous_result in results:
      if previous_result[0] == benchmark_info:
        raise ValueError(f"Duplicated benchmark: {benchmark_info}")

    results.append(
        (benchmark_info, resultjson["context"], resultjson["benchmarks"]))

  return results


def parse_arguments():
  """Parses command-line options."""

  def check_dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise argparse.ArgumentTypeError(path)

  def check_exe_path(path):
    if os.access(path, os.X_OK):
      return path
    else:
      raise argparse.ArgumentTypeError(f"'{path}' is not an executable")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "build_dir",
      metavar="<build-dir>",
      type=check_dir_path,
      help="Path to the build directory containing benchmark suites")
  parser.add_argument("--benchmark_tool",
                      type=check_exe_path,
                      default=None,
                      help="Path to the iree-benchmark-module tool (default to "
                      "iree/tools/iree-benchmark-module under <build-dir>)")
  parser.add_argument("-o",
                      dest="output",
                      default=None,
                      help="Path to the ouput file")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")

  args = parser.parse_args()

  if args.benchmark_tool is None:
    args.benchmark_tool = os.path.join(args.build_dir, "iree", "tools",
                                       "iree-benchmark-module")

  return args


def main(args):
  device_info = AndroidDeviceInfo.from_adb()
  if args.verbose:
    print(device_info)

  if device_info.cpu_abi.lower() not in CPU_ABI_TO_TARGET_ARCH_MAP:
    raise ValueError(f"Unrecognized CPU ABI: '{device_info.cpu_abi}'; "
                     "need to update the map")
  if device_info.gpu_name.lower() not in GPU_NAME_TO_TARGET_ARCH_MAP:
    raise ValueError(f"Unrecognized GPU name: '{device_info.gpu_name}'; "
                     "need to update the map")

  benchmarks = filter_python_model_benchmark_suite(device_info, args.build_dir,
                                                   args.verbose)
  run_results = run_python_model_benchmark_suite(device_info,
                                                 args.build_dir,
                                                 benchmarks,
                                                 args.benchmark_tool,
                                                 verbose=args.verbose)

  results = BenchmarkResults()

  for info, context, runs in run_results:
    results.append_one_benchmark(info, context, runs)

  # Attach commit information.
  results.set_commit(get_git_commit_hash("HEAD"))

  if args.output is not None:
    with open(args.output, "w") as f:
      f.write(results.to_json_str())
  if args.verbose:
    print(results.commit)
    print(results.benchmarks)

  # Clear the benchmark directory on the Android device.
  adb_execute_in_dir(["rm", "-rf", "*"], relative_dir="", verbose=args.verbose)


if __name__ == "__main__":
  main(parse_arguments())
