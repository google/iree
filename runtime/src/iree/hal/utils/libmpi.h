// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_LIBMPI_H_
#define IREE_HAL_UTILS_LIBMPI_H_

#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Data types and enumerations
//===----------------------------------------------------------------------===//

// OpenMPI uses pointers for common data types instead of `int` like most other
// implementations. This is unfortunate as when dynamically importing we can't
// easily switch between the two calling conventions. Today we assume OpenMPI
// everywhere but Windows where we use MSMPI which follows the MPICH values.
//
// TODO: make the values dynamic and query them consistently from the dynamic
// libraries, snoop the library version to populate values from a table, or
// some other trick.
#if !defined(IREE_PLATFORM_WINDOWS)
#define IREE_MPI_TYPES_ARE_POINTERS 1
#endif  // !IREE_PLATFORM_WINDOWS

#if IREE_MPI_TYPES_ARE_POINTERS

typedef void* IREE_MPI_Datatype;
#define IREE_MPI_BYTE(syms) (IREE_MPI_Datatype)((syms)->ompi_mpi_byte)
#define IREE_MPI_INT(syms) (IREE_MPI_Datatype)((syms)->ompi_mpi_int)
#define IREE_MPI_FLOAT(syms) (IREE_MPI_Datatype)((syms)->ompi_mpi_float)
#define IREE_MPI_DOUBLE(syms) (IREE_MPI_Datatype)((syms)->ompi_mpi_double)

typedef void* IREE_MPI_Comm;
#define IREE_MPI_COMM_WORLD(syms) (IREE_MPI_Comm)((syms)->ompi_mpi_comm_world)

typedef void* IREE_MPI_Op;
#define IREE_MPI_OP_MAX(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_max)
#define IREE_MPI_OP_MIN(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_min)
#define IREE_MPI_OP_SUM(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_sum)
#define IREE_MPI_OP_PROD(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_prod)
#define IREE_MPI_OP_LAND(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_land)
#define IREE_MPI_OP_BAND(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_band)
#define IREE_MPI_OP_LOR(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_lor)
#define IREE_MPI_OP_BOR(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_bor)
#define IREE_MPI_OP_LXOR(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_lxor)
#define IREE_MPI_OP_BXOR(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_bxor)
#define IREE_MPI_OP_MAXLOC(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_maxloc)
#define IREE_MPI_OP_MINLOC(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_minloc)
#define IREE_MPI_OP_REPLACE(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_replace)
#define IREE_MPI_OP_NO_OP(syms) (IREE_MPI_Op)((syms)->ompi_mpi_op_no_op)

#else

typedef int IREE_MPI_Datatype;
#define IREE_MPI_BYTE(syms) ((IREE_MPI_Datatype)0x4C00010D)
#define IREE_MPI_INT(syms) ((IREE_MPI_Datatype)0x4C000405)
#define IREE_MPI_FLOAT(syms) ((IREE_MPI_Datatype)0x4C00040A)
#define IREE_MPI_DOUBLE(syms) ((IREE_MPI_Datatype)0x4C00080B)

typedef int IREE_MPI_Comm;
#define IREE_MPI_COMM_WORLD(syms) ((IREE_MPI_Comm)0x44000000)

typedef int IREE_MPI_Op;
#define IREE_MPI_OP_MAX(syms) ((IREE_MPI_Op)0x58000001)
#define IREE_MPI_OP_MIN(syms) ((IREE_MPI_Op)0x58000002)
#define IREE_MPI_OP_SUM(syms) ((IREE_MPI_Op)0x58000003)
#define IREE_MPI_OP_PROD(syms) ((IREE_MPI_Op)0x58000004)
#define IREE_MPI_OP_LAND(syms) ((IREE_MPI_Op)0x58000005)
#define IREE_MPI_OP_BAND(syms) ((IREE_MPI_Op)0x58000006)
#define IREE_MPI_OP_LOR(syms) ((IREE_MPI_Op)0x58000007)
#define IREE_MPI_OP_BOR(syms) ((IREE_MPI_Op)0x58000008)
#define IREE_MPI_OP_LXOR(syms) ((IREE_MPI_Op)0x58000009)
#define IREE_MPI_OP_BXOR(syms) ((IREE_MPI_Op)0x5800000a)
#define IREE_MPI_OP_MINLOC(syms) ((IREE_MPI_Op)0x5800000b)
#define IREE_MPI_OP_MAXLOC(syms) ((IREE_MPI_Op)0x5800000c)
#define IREE_MPI_OP_REPLACE(syms) ((IREE_MPI_Op)0x5800000d)
#define IREE_MPI_OP_NO_OP(syms) ((IREE_MPI_Op)0x5800000e)

#endif  // IREE_MPI_TYPES_ARE_POINTERS

//===----------------------------------------------------------------------===//
// Dynamic symbol table
//===----------------------------------------------------------------------===//

// A collection of known MPI symbols (functions and handles)
//
// To support additional functions, define the function or handles
// in libmpi_dynamic_symbols.h
typedef struct iree_hal_mpi_dynamic_symbols_t {
#define MPI_PFN_DECL(mpiSymbolName, ...) int (*mpiSymbolName)(__VA_ARGS__);
#include "iree/hal/utils/libmpi_dynamic_symbols.h"
#undef MPI_PFN_DECL
} iree_hal_mpi_dynamic_symbols_t;

// Dynamically loads the MPI library and sets up the symbol table.
// |out_library| must be released by the caller with
// iree_dynamic_library_release. |out_syms| is only valid as long as the
// library is live.
iree_status_t iree_hal_mpi_library_load(
    iree_allocator_t host_allocator, iree_dynamic_library_t** out_library,
    iree_hal_mpi_dynamic_symbols_t* out_syms);

//===----------------------------------------------------------------------===//
// Error handling utilities
//===----------------------------------------------------------------------===//

// Converts an MPI_Status result to an iree_status_t.
//
// Usage:
//   iree_status_t status = MPI_RESULT_TO_STATUS(mpiDoThing(...));
#define MPI_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_mpi_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// Converts an MPI_Status result to an iree_status_t.
iree_status_t iree_hal_mpi_result_to_status(
    iree_hal_mpi_dynamic_symbols_t* syms, int result, const char* file,
    uint32_t line);

// IREE_RETURN_IF_ERROR but implicitly converts the mpi return value to
// a Status.
//
// Usage:
//   MPI_RETURN_IF_ERROR(mpiDoThing(...), "message");
#define MPI_RETURN_IF_ERROR(syms, expr, ...)                                 \
  IREE_RETURN_IF_ERROR(iree_hal_mpi_result_to_status((syms), ((syms)->expr), \
                                                     __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the mpi return value to a
// Status.
//
// Usage:
//   MPI_IGNORE_ERROR(mpiDoThing(...));
#define MPI_IGNORE_ERROR(syms, expr)                                      \
  IREE_IGNORE_ERROR(iree_hal_mpi_result_to_status((syms), ((syms)->expr), \
                                                  __FILE__, __LINE__))

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_LIBMPI_H_
