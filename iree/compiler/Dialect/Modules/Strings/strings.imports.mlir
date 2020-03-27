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

// Describes a custom module implemented in native code.
// The imports are used for mapping higher-level IR types to the VM ABI and for
// attaching additional attributes for compiler optimization.
//
// Each import defined here has a matching function exported from the native
// module (strings/native_module.cc). In most cases an op in the source
// dialect will map directly to an import here, though it's possible for
// versioning and overrides to cause M:N mappings.
vm.module @strings {

// Returns the string representation of an i32.
// Maps to the IREE::Strings::I32ToString.
vm.import @i32_to_string(%value : i32) -> !vm.ref<!strings.string>
attributes {nosideeffects}

// Returns a tensor of strings of elementwise to string of
// buffer_view elements.
// Maps to the IREE::Strings::ToStringTensor.
vm.import @to_string_tensor(%value : !vm.ref<!hal.buffer_view>) -> !vm.ref<!strings.string_tensor>
attributes {nosideeffects}

// Prints the contents of a string.
// Maps to the IREE::Strings::Print.
vm.import @print(%value : !vm.ref<!strings.string>)

// Prints the contents of a string tensor.
// Maps to the IREE::Strings::PrintTensor.
vm.import @print_tensor(%value : !vm.ref<!strings.string_tensor>)

}  // vm.module
