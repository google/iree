# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io
from pathlib import Path
import re
import subprocess
import unittest
import tempfile
import sys

from iree.build import *

THIS_DIR = Path(__file__).resolve().parent


class MnistBuilderTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._temp_dir.__enter__()
        self.output_path = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.__exit__(None, None, None)

    # Tests that invoking via the tool works:
    #   python -m iree.build {path to py file}
    # We execute this out of process in order to verify the full flow.
    def testBuildEntrypoint(self):
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "iree.build",
                str(THIS_DIR / "mnist_builder.py"),
                "--output-dir",
                str(self.output_path),
            ]
        ).decode()
        print("OUTPUT:", output)
        output_paths = output.splitlines()
        self.assertEqual(len(output_paths), 1)
        output_path = Path(output_paths[0])
        self.assertTrue(output_path.is_relative_to(self.output_path))
        contents = output_path.read_text()
        self.assertIn("module", contents)

    # Tests that invoking via the build module itself works
    #   python {path to py file}
    # We execute this out of process in order to verify the full flow.
    def testTargetModuleEntrypoint(self):
        output = subprocess.check_output(
            [
                sys.executable,
                str(THIS_DIR / "mnist_builder.py"),
                "--output-dir",
                str(self.output_path),
            ]
        ).decode()
        print("OUTPUT:", output)
        output_paths = output.splitlines()
        self.assertEqual(len(output_paths), 1)

    def testListCommand(self):
        mod = load_build_module(THIS_DIR / "mnist_builder.py")
        out_file = io.StringIO()
        iree_build_main(mod, args=["--list"], stdout=out_file)
        output = out_file.getvalue().strip()
        self.assertEqual(output, "mnist")

    def testListAllCommand(self):
        mod = load_build_module(THIS_DIR / "mnist_builder.py")
        out_file = io.StringIO()
        iree_build_main(mod, args=["--list-all"], stdout=out_file)
        output = out_file.getvalue().splitlines()
        self.assertIn("mnist", output)
        self.assertIn("mnist/mnist.onnx", output)

    def testActionCLArg(self):
        mod = load_build_module(THIS_DIR / "mnist_builder.py")
        out_file = io.StringIO()
        err_file = io.StringIO()
        with self.assertRaisesRegex(
            IOError,
            re.escape("Failed to fetch URL 'https://github.com/iree-org/doesnotexist'"),
        ):
            iree_build_main(
                mod,
                args=[
                    "--mnist-onnx-url",
                    "https://github.com/iree-org/doesnotexist",
                ],
                stdout=out_file,
                stderr=err_file,
            )


if __name__ == "__main__":
    try:
        import onnx
    except ModuleNotFoundError:
        print(f"Skipping test {__file__} because Python dependency `onnx` is not found")
        sys.exit(0)

    unittest.main()
