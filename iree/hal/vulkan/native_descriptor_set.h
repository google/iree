// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_
#define IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_

// clang-format off
#include "iree/hal/vulkan/vulkan_headers.h"
// clang-format on

#include "iree/hal/descriptor_set.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// A DescriptorSet implemented with the native VkDescriptorSet type.
class NativeDescriptorSet final : public DescriptorSet {
 public:
  NativeDescriptorSet(ref_ptr<VkDeviceHandle> logical_device,
                      VkDescriptorSet handle);
  ~NativeDescriptorSet() override;

  VkDescriptorSet handle() const { return handle_; }

 private:
  ref_ptr<VkDeviceHandle> logical_device_;
  VkDescriptorSet handle_;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_NATIVE_DESCRIPTOR_SET_H_
