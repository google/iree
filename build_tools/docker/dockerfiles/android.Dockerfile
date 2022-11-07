# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE towards Android.

FROM gcr.io/iree-oss/base@sha256:5d43683c6b50aebe1fca6c85f2012f3b0fa153bf4dd268e8767b619b1891423a
ARG NDK_VERSION=r25b
WORKDIR /install-ndk

ENV ANDROID_NDK "/usr/src/android-ndk-${NDK_VERSION}"

RUN wget -q "https://dl.google.com/android/repository/android-ndk-${NDK_VERSION?}-linux.zip" \
    && unzip -q "android-ndk-${NDK_VERSION?}-linux.zip" -d /usr/src/  \
    && rm -rf /install-ndk

WORKDIR /
