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

#include "iree/modules/strings/strings_module.h"

#include <sstream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/logging.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/modules/strings/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/module.h"
#include "iree/vm/module_abi_cc.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"
#include "iree/vm/types.h"

static iree_vm_ref_type_descriptor_t string_descriptor = {0};
static iree_vm_ref_type_descriptor_t string_tensor_descriptor = {0};

IREE_VM_DEFINE_TYPE_ADAPTERS(string, string_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(string_tensor, string_tensor_t);

namespace iree {
namespace {
class StringsModuleState final {
 public:
  explicit StringsModuleState(iree_allocator_t allocator)
      : allocator_(allocator) {}
  ~StringsModuleState() = default;

  Status Initialize() { return OkStatus(); }

  // strings.print(%str)
  Status Print(vm::ref<string_t> str) {
    fwrite(str->value.data, 1, str->value.size, stdout);
    fputc('\n', stdout);
    fflush(stdout);
    return OkStatus();
  }

  // strings.i32_to_string(%value) -> %str
  StatusOr<vm::ref<string_t>> I32ToString(int32_t value) {
    vm::ref<string_t> new_string;
    std::string str = std::to_string(value);
    RETURN_IF_ERROR(
        FromApiStatus(string_create(iree_make_cstring_view(str.c_str()),
                                    allocator_, &new_string),
                      IREE_LOC));
    return new_string;
  }

  const iree_string_view_t* PrintTensorHelper(const iree_string_view_t* strs,
                                              const int32_t* shape,
                                              int32_t rank) {
    // Handle a scalar tensor value.
    if (rank == 0) {
      const auto& str = strs[0];
      fwrite(str.data, 1, str.size, stdout);
      return strs + 1;
    }

    // The row for the final tensor dimension.
    if (rank == 1) {
      fputc('[', stdout);
      for (int32_t i = 0, s = shape[0]; i < s; i++) {
        const auto& str = strs[i];
        fwrite(str.data, 1, str.size, stdout);
        if (i != s - 1) {
          fwrite(", ", 1, /*size=*/2, stdout);
        }
      }

      fputc(']', stdout);
      return strs + shape[0];
    }

    // Recurse to the lower dimension with the approrpiate brackets.
    fputc('[', stdout);
    for (int32_t i = 0, s = shape[0]; i < s; i++) {
      strs = PrintTensorHelper(strs, shape + 1, rank - 1);
      if (i != s - 1) {
        fwrite(",\n", 1, /*size=*/2, stdout);
      }
    }
    fputc(']', stdout);
    return strs;
  }

  // strings.print_tensor(%str_tensor)
  Status PrintTensor(vm::ref<string_tensor_t> str_tensor) {
    if (!str_tensor) {
      return OkStatus();
    }

    PrintTensorHelper(str_tensor->values, str_tensor->shape, str_tensor->rank);
    fputc('\n', stdout);
    fflush(stdout);
    return OkStatus();
  }

  // strings.to_string_tensor(%hal_buffer) -> %str_tensor
  StatusOr<vm::ref<string_tensor_t>> ToStringTensor(
      vm::ref<iree_hal_buffer_view_t> hal_buffer_view) {
    const size_t rank = iree_hal_buffer_view_shape_rank(hal_buffer_view.get());
    absl::InlinedVector<int32_t, 6> shape(rank);
    RETURN_IF_ERROR(
        FromApiStatus(iree_hal_buffer_view_shape(hal_buffer_view.get(), rank,
                                                 shape.data(), nullptr),
                      IREE_LOC));

    size_t num_elements = 1;
    for (auto val : shape) {
      num_elements *= val;
    }

    // Pull the values down.
    size_t element_size =
        iree_hal_buffer_view_element_size(hal_buffer_view.get());
    size_t tensor_size = element_size * num_elements;
    iree_hal_buffer_t* hal_buffer =
        iree_hal_buffer_view_buffer(hal_buffer_view.get());
    iree_hal_mapped_memory_t tensor_mapping;
    RETURN_IF_ERROR(FromApiStatus(
        iree_hal_buffer_map(hal_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                            /*element_offset=*/0, tensor_size, &tensor_mapping),
        IREE_LOC));

    iree_hal_element_type_t type =
        iree_hal_buffer_view_element_type(hal_buffer_view.get());

    std::vector<std::string> strings;
    strings.reserve(num_elements);
    const auto& contents = tensor_mapping.contents;
    if (type == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
      for (const float *
               p = (const float*)contents.data,
              *s = (const float*)(contents.data + contents.data_length);
           p < s; p++) {
        std::string str = std::to_string(*p);
        strings.push_back(std::move(str));
      }
    } else {
      return FromApiStatus(IREE_STATUS_UNIMPLEMENTED, IREE_LOC);
    }

    // Place into iree_string_views.
    std::vector<iree_string_view_t> string_views;
    string_views.reserve(num_elements);

    for (const auto& str : strings) {
      string_views.push_back(iree_make_cstring_view(str.data()));
    }

    for (const auto& view : string_views) {
    }

    string_tensor_t* string_tensor;
    RETURN_IF_ERROR(
        FromApiStatus(string_tensor_create(allocator_, string_views.data(),
                                           string_views.size(), shape.data(),
                                           rank, &string_tensor),
                      IREE_LOC));

    return string_tensor;
  }

 private:
  // Allocator that the caller requested we use for any allocations we need to
  // perform during operation.
  iree_allocator_t allocator_ = IREE_ALLOCATOR_SYSTEM;
};

static const vm::NativeFunction<StringsModuleState> kStringsModuleFunctions[] =
    {
        vm::MakeNativeFunction("print", &StringsModuleState::Print),
        vm::MakeNativeFunction("i32_to_string",
                               &StringsModuleState::I32ToString),
        vm::MakeNativeFunction("print_tensor",
                               &StringsModuleState::PrintTensor),
        vm::MakeNativeFunction("to_string_tensor",
                               &StringsModuleState::ToStringTensor),

};

class StringsModule final : public vm::NativeModule<StringsModuleState> {
 public:
  using vm::NativeModule<StringsModuleState>::NativeModule;

  // Example of global initialization (shared across all contexts), such as
  // loading native libraries or creating shared pools.
  Status Initialize() { return OkStatus(); }

  // Creates per-context state when the module is added to a new context.
  // May be called from any thread.
  StatusOr<std::unique_ptr<StringsModuleState>> CreateState(
      iree_allocator_t allocator) override {
    auto state = std::make_unique<StringsModuleState>(allocator);
    RETURN_IF_ERROR(state->Initialize());
    return state;
  }
};

}  // namespace
}  // namespace iree

extern "C" iree_status_t strings_module_register_types() {
  if (string_descriptor.type) {
    return IREE_STATUS_OK;  // Already registered.
  }

  // Register strings.string
  string_descriptor.type_name = iree_make_cstring_view("strings.string");
  string_descriptor.offsetof_counter = offsetof(string_t, ref_object.counter);
  string_descriptor.destroy = string_destroy;
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&string_descriptor));

  // Register strings.string_tensor
  string_tensor_descriptor.type_name =
      iree_make_cstring_view("strings.string_tensor");
  string_tensor_descriptor.offsetof_counter =
      offsetof(string_tensor_t, ref_object.counter);
  string_tensor_descriptor.destroy = string_tensor_destroy;
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&string_tensor_descriptor));

  return IREE_STATUS_OK;
}

extern "C" iree_status_t strings_module_create(iree_allocator_t allocator,
                                               iree_vm_module_t** out_module) {
  if (!out_module) return IREE_STATUS_INVALID_ARGUMENT;
  *out_module = NULL;
  auto module = std::make_unique<iree::StringsModule>(
      "strings", allocator, absl::MakeConstSpan(iree::kStringsModuleFunctions));
  auto status = module->Initialize();
  if (!status.ok()) {
    return iree::ToApiStatus(status);
  }
  *out_module = module.release()->interface();
  return IREE_STATUS_OK;
}
