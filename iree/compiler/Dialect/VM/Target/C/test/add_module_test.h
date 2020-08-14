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

#include "iree/vm/context.h"
#include "iree/vm/instance.h"
#include "iree/vm/native_module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

// This would be generated together with the functions in the header
#include "iree/compiler/Dialect/VM/Target/C/test/add_mlir_generated.h"

static const iree_vm_native_export_descriptor_t module_a_exports_[] = {
    {iree_make_cstring_view("test_function"), 0, 0, 0, NULL},
};
static const iree_vm_native_module_descriptor_t module_a_descriptor_ = {
    iree_make_cstring_view("module_a"),
    0,
    NULL,
    IREE_ARRAYSIZE(module_a_exports_),
    module_a_exports_,
    0,
    NULL,
};

struct module_a_s;
struct module_a_state_s;
typedef struct module_a_s module_a_t;
typedef struct module_a_state_s module_a_state_t;

static iree_status_t module_a_test_function(
    module_a_t* module, module_a_state_t* state, iree_vm_stack_t* stack,
    const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result) {
  // TODO(benvanik): iree_vm_stack native frame enter/leave.
  // By not enter/leaving a frame here we won't be able to set breakpoints or
  // tracing on the function. Fine for now.
  iree_vm_stack_frame_t* caller_frame = iree_vm_stack_current_frame(stack);
  const iree_vm_register_list_t* arg_list = call->argument_registers;
  const iree_vm_register_list_t* ret_list = call->result_registers;
  auto& regs = caller_frame->registers;

  // Load the input argument.
  // This should really be generated code (like module_abi_cc.h).
  int32_t arg0 = regs.i32[arg_list->registers[0] & regs.i32_mask];
  int32_t arg1 = regs.i32[arg_list->registers[1] & regs.i32_mask];

  int32_t out0;
  int32_t out1;

  test_function(arg0, arg1, &out0, &out1);

  // Store the result.
  regs.i32[ret_list->registers[0] & regs.i32_mask] = out0;
  regs.i32[ret_list->registers[1] & regs.i32_mask] = out1;

  return iree_ok_status();
}

typedef iree_status_t (*module_a_func_t)(
    module_a_t* module, module_a_state_t* state, iree_vm_stack_t* stack,
    const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result);
static const module_a_func_t module_a_funcs_[] = {
    module_a_test_function,
};
static_assert(IREE_ARRAYSIZE(module_a_funcs_) ==
                  IREE_ARRAYSIZE(module_a_exports_),
              "function pointer table must be 1:1 with exports");

static iree_status_t IREE_API_PTR module_a_begin_call(
    void* self, iree_vm_stack_t* stack, const iree_vm_function_call_t* call,
    iree_vm_execution_result_t* out_result) {
  // NOTE: we aren't using module state in this module.
  return module_a_funcs_[call->function.ordinal](
      /*module=*/NULL, /*module_state=*/NULL, stack, call, out_result);
}

static iree_status_t module_a_create(iree_allocator_t allocator,
                                     iree_vm_module_t** out_module) {
  // NOTE: this module has neither shared or per-context module state.
  iree_vm_module_t interface;
  IREE_RETURN_IF_ERROR(iree_vm_module_initialize(&interface, NULL));
  interface.begin_call = module_a_begin_call;
  return iree_vm_native_module_create(&interface, &module_a_descriptor_,
                                      allocator, out_module);
}
