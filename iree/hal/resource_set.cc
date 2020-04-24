// Copyright 2019 Google LLC
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

#include "iree/hal/resource_set.h"

namespace iree {
namespace hal {

ResourceSet::ResourceSet() = default;

ResourceSet::~ResourceSet() = default;

Status ResourceSet::Insert(ref_ptr<Resource> resource) {
  // DO NOT SUBMIT
  return UnimplementedErrorBuilder(IREE_LOC) << "Insert NYI";
}

Status ResourceSet::Union(const ResourceSet& other_set) {
  // DO NOT SUBMIT
  return UnimplementedErrorBuilder(IREE_LOC) << "Union NYI";
}

}  // namespace hal
}  // namespace iree
