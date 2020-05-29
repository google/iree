# Copyright 2020 Google LLC
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

"""Macro for building e2e keras vision model tests."""

load("//bindings/python:build_defs.oss.bzl", "iree_py_test")

def iree_vision_test_suite(
        name,
        configurations,
        external_weights = None,
        deps = None,
        tags = None,
        size = "large",
        python_version = "PY3",
        **kwargs):
    """Creates one iree_py_test per configuration tuple and bundles them.

    Args:
      name: name of the generated test suite.
      configurations: a list of tuples of (dataset, include_top, model,
                      backends) that specifies which data, model and backends to use for a given test.
      external_weights: a base url to fetch trained model weights from.
      tags: tags to apply to the test. Note that as in standard test suites,
            manual is treated specially and will also apply to the test suite
            itself.
      size: size of the tests.
      python_version: the python version to run the tests with. Uses python3
                      by default.
      **kwargs: Any additional arguments that will be passed to the underlying
                tests and test_suite.
    """
    tests = []
    for dataset, include_top, model, backends in configurations:
        test_name = "{}_{}_top_{}_{}_{}_test".format(
            name, dataset, include_top, model, backends)
        tests.append(test_name)

        args = [
            "--data={}".format(dataset),
            "--include_top={}".format(include_top),
            "--model={}".format(model),
            "--override_backends={}".format(backends),
        ]
        if external_weights:
            args.append("--url={}".format(external_weights))

        iree_py_test(
            name = test_name,
            main = "keras_vision_model_test.py",
            srcs = ["keras_vision_model_test.py"],
            args = args,
            tags = tags,
            deps = deps,
            size = size,
            python_version = python_version,
        )

    native.test_suite(
        name = name,
        tests = tests,
        # Note that only the manual tag really has any effect here. Others are
        # used for test suite filtering, but all tests are passed the same tags.
        tags = tags,
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )
