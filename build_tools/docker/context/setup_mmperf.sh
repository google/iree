#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Sets up `mmperf` (https://github.com/mmperf/mmperf).
#
# `mmperf` benchmarks matrix-multiply workloads on IREE and other backends such
# as RUY, TVM, Halide, CuBLAS, etc. Some backends are included as submodules
# in the `mmperf` repo and built from source, and other backends are expected
# to already be installed.
#
# Usage:
#    ./setup_mmperf.sh \
#        <mmperf repo dir> \
#        <mmperf sha>

set -xeuo pipefail

export REPO_DIR=$1
export REPO_SHA=$2

pushd ${REPO_DIR}

mkdir mmperf
pushd mmperf
git init
git fetch --depth 1 https://github.com/mmperf/mmperf.git "${REPO_SHA}"
git checkout ${REPO_SHA}
git submodule update --init --recursive --jobs 8 --depth 1

# Create virtual environment.
python3 -m venv mmperf.venv
source mmperf.venv/bin/activate
pip install -r requirements.txt
pip install -r ./external/llvm-project/mlir/python/requirements.txt

# Since the root user clones the convperf repo, we update permissions so that a
# runner can access this repo, but we don't want to set the executable bit for
# non-executables because git tracks this, so we then restore any git-tracked
# changes.
chmod -R 777 .
git restore .
git submodule foreach --recursive git restore .

# Set all repos as a safe directory. Git will not run commands on this repo as a
# non-root user unless it is marked safe.
for i in $(find ${REPO_DIR} -name '.git' | xargs dirname); do
  git config --system --add safe.directory $i
done

popd # mmperf
popd # ${REPO_DIR}
