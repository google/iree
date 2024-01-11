// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/executable_library_util.h"

iree_status_t iree_hal_executable_library_verify(
    const iree_hal_executable_params_t* executable_params,
    const iree_hal_executable_library_v0_t* library) {
  // Tooling and testing may disable verification to make it easier to define
  // libraries. The compiler should never produce anything that fails
  // verification, though, and should always have it enabled.
  const bool disable_verification =
      iree_all_bits_set(executable_params->caching_mode,
                        IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION);
  if (disable_verification) return iree_ok_status();

  // Check that there's one pipeline layout per export. Multiple exports may
  // share the same layout but it still needs to be declared.
  // NOTE: pipeline layouts are optional but if provided must be consistent.
  if (executable_params->pipeline_layout_count > 0) {
    if (library->exports.count != executable_params->pipeline_layout_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "executable provides %u entry points but caller "
                              "provided %" PRIhsz "; must match",
                              library->exports.count,
                              executable_params->pipeline_layout_count);
    }
  }

  // Check to make sure that the constant table has values for all constants.
  if (library->constants.count != executable_params->constant_count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "executable requires %u constants but caller "
                            "provided %" PRIhsz "; must match",
                            library->constants.count,
                            executable_params->constant_count);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_executable_library_initialize_imports(
    iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_import_provider_t import_provider,
    const iree_hal_executable_import_table_v0_t* import_table,
    iree_hal_executable_import_thunk_v0_t import_thunk,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(import_thunk);
  if (!import_table || !import_table->count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, import_table->count);

  // The thunk is used to give the loader a chance to intercept import calls
  // in cases where it needs to JIT, perform FFI/ABI conversion, etc.
  environment->import_thunk = import_thunk;

  // Allocate storage for the imports.
  iree_host_size_t import_funcs_size =
      iree_host_align(import_table->count * sizeof(*environment->import_funcs),
                      iree_max_align_t);
  iree_host_size_t import_contexts_size = iree_host_align(
      import_table->count * sizeof(*environment->import_contexts),
      iree_max_align_t);
  uint8_t* base_ptr = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                import_funcs_size + import_contexts_size,
                                (void**)&base_ptr));
  environment->import_funcs = (const iree_hal_executable_import_v0_t*)base_ptr;
  environment->import_contexts = (const void**)(base_ptr + import_funcs_size);

  // Try to resolve each import.
  // Will fail if any required import is not found.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_executable_import_provider_try_resolve(
              import_provider, import_table->count, import_table->symbols,
              (void**)environment->import_funcs,
              (void**)environment->import_contexts,
              /*out_resolution=*/NULL));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_executable_library_deinitialize_imports(
    iree_hal_executable_environment_v0_t* environment,
    iree_allocator_t host_allocator) {
  // NOTE: import_funcs and import_contexts are allocated as one block.
  if (environment->import_funcs != NULL) {
    iree_allocator_free(host_allocator, (void*)environment->import_funcs);
  }
  environment->import_funcs = NULL;
  environment->import_contexts = NULL;
  environment->import_thunk = NULL;
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

iree_zone_id_t iree_hal_executable_library_call_zone_begin(
    iree_string_view_t executable_identifier,
    const iree_hal_executable_library_v0_t* library, iree_host_size_t ordinal) {
  iree_string_view_t entry_point_name = iree_string_view_empty();
  if (library->exports.names != NULL) {
    entry_point_name = iree_make_cstring_view(library->exports.names[ordinal]);
  }
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_dylib_call");
  }

  const char* source_file = NULL;
  size_t source_file_length = 0;
  uint32_t source_line;
  if (library->exports.src_locs != NULL) {
    // We have source location data, so use it.
    source_file = library->exports.src_locs[ordinal].path;
    source_file_length = library->exports.src_locs[ordinal].path_length;
    source_line = library->exports.src_locs[ordinal].line;
    // printf("iree_hal_executable_library_call_zone_begin(): source file(%d):
    // %s\n", source_line, source_file);
  } else {
    // No source location data, so make do with what we have.
    source_file = executable_identifier.data;
    source_file_length = executable_identifier.size;
    source_line = ordinal;
  }

  IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, source_file, source_file_length,
                                 source_line, entry_point_name.data,
                                 entry_point_name.size, NULL, 0);

  if (library->exports.tags != NULL) {
    const char* tag = library->exports.tags[ordinal];
    if (tag) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, tag);
    }
  }

  return z0;
}
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

iree_status_t iree_hal_executable_library_setup_tracing(
    const iree_hal_executable_library_v0_t* library,
    iree_allocator_t host_allocator, tracy_file_mapping** out_file_mapping) {
  iree_status_t status = iree_ok_status();
#if (IREE_TRACING_FEATURES)
  // Do we have src files?
  bool contains_srs = false;
  for (uint32_t i = 0; i < library->exports.count; ++i) {
    if (library->exports.src_files[i].file_contents_length > 0) {
      contains_srs = true;
      break;
    }
  }
  if (!contains_srs) {
    fprintf(
        stdout,
        "iree_hal_executable_library_setup_tracing(): found no sources. not "
        "registering\n");
    return iree_ok_status();
  }

  // Alloc tracy file mapping
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, sizeof(**out_file_mapping),
                                   (void**)out_file_mapping);
  }

  tracy_file_mapping* file_mapping = *out_file_mapping;

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        host_allocator,
        library->exports.count * sizeof(*file_mapping->file_contents),
        (void**)&file_mapping->file_contents);
  }

  // Fill tracy mapping
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < library->exports.count; ++i) {
      // TODO(kooljblack): use custom names for custom file contents
      const char* name = library->exports.names[i];
      const char* file_name = library->exports.src_locs[i].path;
      uint32_t file_name_length = library->exports.src_locs[i].path_length;
      const char* file_contents = library->exports.src_files[i].file_contents;
      uint32_t file_contents_length =
          library->exports.src_files[i].file_contents_length;
      fprintf(stdout,
              "iree_hal_executable_library_setup_tracing(): adding src for "
              "entry point: '%s'\n",
              name);
      fprintf(stdout,
              "iree_hal_executable_library_setup_tracing(): ...with file name: "
              "'%s'\n",
              file_name);
      fprintf(stdout,
              "iree_hal_executable_library_setup_tracing(): ...with contents: "
              "'%s'\n",
              file_contents);
      file_mapping->file_contents[i].file_name = file_name;
      file_mapping->file_contents[i].file_name_length = file_name_length;
      file_mapping->file_contents[i].file_contents = file_contents;
      file_mapping->file_contents[i].file_contents_length =
          file_contents_length;
    }
    file_mapping->file_mapping_length = library->exports.count;
  }

  // Sort file mappings
  qsort(file_mapping->file_contents, file_mapping->file_mapping_length,
        sizeof(*file_mapping->file_contents), tracy_file_contents_sort_cmp);

  // Register the mapping
  if (iree_status_is_ok(status)) {
    iree_tracing_register_custom_file_contents(file_mapping);
  }
#endif
  return status;
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
