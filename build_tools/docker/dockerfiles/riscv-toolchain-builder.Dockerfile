# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The docker image to build the GCC/LLVM RISC-V toolchain

FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:eab9b04b5ac6df679995fc805e887ed8f2a9995a84918e52aa6e7a6024a2ed52 AS final

ENV TZ=UTC
RUN ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime && echo "${TZ}" > /etc/timezone

RUN yum -y install epel-release && yum -y update
RUN yum groupinstall -y "Development Tools"
RUN yum install -y \
  bison \
  ca-certificates \
  clang \
  cmake \
  curl \
  flex \
  expat-devel \
  gawk \
  gmp-devel \
  ncurses-devel \
  ninja-build \
  pixman-devel \
  python39 \
  python39-devel \
  texinfo \
  unzip \
  wget

# Update local installed cert to match Github updated cert.
RUN update-ca-trust

# Use clang to compile the toolchain
ENV CC=clang
ENV CXX=clang++
