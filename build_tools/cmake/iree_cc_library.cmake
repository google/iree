# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# TEXTUAL_HDRS: List of public header files that cannot be compiled on their own
# SRCS: List of source files for the library
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# PUBLIC: Add this so that this library will be exported under iree::
# Also in IDE, target will appear in IREE folder while non PUBLIC will be in IREE/internal.
# TESTONLY: When added, this target will only be built if user passes -DIREE_BUILD_TESTS=ON to CMake.
# SHARED: If set, will compile to a shared object.
#
# Note:
# By default, iree_cc_library will always create a library named iree_${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
#
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# iree_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     iree::package::awesome # not "awesome" !
#   PUBLIC
# )
#
# iree_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     iree::package::fantastic_lib
# )
function(iree_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY;SHARED"
    "NAME"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Replace dependencies passed by ::name with iree::package::name
  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_OBJECTS_NAME ${_NAME}.objects)

  # Check if this is a header-only library.
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(_CC_SRCS "${_RULE_SRCS}")
  foreach(src_file IN LISTS _CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|inc)")
      list(REMOVE_ITEM _CC_SRCS "${src_file}")
    endif()
  endforeach()
  if("${_CC_SRCS}" STREQUAL "")
    set(_RULE_IS_INTERFACE 1)
  else()
    set(_RULE_IS_INTERFACE 0)
  endif()

  if(NOT _RULE_IS_INTERFACE)
    add_library(${_OBJECTS_NAME} OBJECT)
    if(_RULE_SHARED)
      add_library(${_NAME} SHARED "$<TARGET_OBJECTS:${_OBJECTS_NAME}>")
    else()
      add_library(${_NAME} STATIC "$<TARGET_OBJECTS:${_OBJECTS_NAME}>")
    endif()

    # Sources get added to the object library.
    target_sources(${_OBJECTS_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_TEXTUAL_HDRS}
        ${_RULE_HDRS}
    )

    # We define everything else on the regular rule. However, the object
    # library needs compiler definition related properties, so we forward them.
    # We also forward link libraries -- not because the OBJECT libraries do
    # linking but because they get transitive compile definitions from them.
    # Yes. This is state of the art.
    # Note that SYSTEM scope matches here, in the property name and in the
    # include directories below on the main rule. If ever removing this,
    # remove it from all places.
    target_include_directories(${_OBJECTS_NAME} SYSTEM
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
    )
    target_include_directories(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_compile_options(${_OBJECTS_NAME}
      PRIVATE
        $<TARGET_PROPERTY:${_NAME},COMPILE_OPTIONS>
    )
    target_compile_definitions(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_COMPILE_DEFINITIONS>
    )
    target_link_libraries(${_OBJECTS_NAME}
      PUBLIC
        $<TARGET_PROPERTY:${_NAME},INTERFACE_LINK_LIBRARIES>
    )

    # Only OBJECT libraries need the CXX_STANDARD set.
    set_property(TARGET ${_OBJECTS_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
    set_property(TARGET ${_OBJECTS_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

    # We set everything on the main target.
    target_include_directories(${_NAME} SYSTEM
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
    )
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
    )
    target_compile_options(${_NAME}
      PRIVATE
        ${IREE_DEFAULT_COPTS}
        ${_RULE_COPTS}
    )
    target_link_options(${_NAME}
      PRIVATE
        ${IREE_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
    )
    target_link_libraries(${_NAME}
      PUBLIC
        ${_RULE_DEPS}
    )

    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )

    # Add our objects to the __IREE_OBJECT_MAPPINGS__ target property. See
    # the comment on that target in the main CMakeLists.txt.
    set_property(TARGET __IREE_OBJECT_MAPPINGS__
      APPEND PROPERTY
        IREE_OBJECTS_${_PACKAGE_NAME} $<TARGET_OBJECTS:${_OBJECTS_NAME}>
    )

    # Add all IREE targets to a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
      set_property(TARGET ${_OBJECTS_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
      set_property(TARGET ${_OBJECTS_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
      set_property(TARGET ${_OBJECTS_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
    endif()
  else()
    # Generating header-only library.
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME} SYSTEM
      INTERFACE
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
    )
    target_link_options(${_NAME}
      INTERFACE
        ${IREE_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_RULE_DEPS}
    )
    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  # If the library name matches the final component of the package then treat
  # it as a default. For example, foo/bar/ library 'bar' would end up as
  # 'foo::bar'.
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()
endfunction()
