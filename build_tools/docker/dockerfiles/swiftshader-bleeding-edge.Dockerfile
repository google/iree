# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/base-bleeding-edge@sha256:c6ffb43fdff62540f0a89486325a33e5b4849cd24619545e78a4f6c733b4d77f AS install-swiftshader
WORKDIR /install-swiftshader

COPY build_tools/third_party/swiftshader/build_vk_swiftshader.sh ./
RUN ./build_vk_swiftshader.sh "/swiftshader" && rm -rf /install-swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json

WORKDIR /
