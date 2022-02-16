// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_COMMAND_BUFFER_TEST_H_
#define IREE_HAL_CTS_COMMAND_BUFFER_TEST_H_

#include <cstdint>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

using ::testing::ContainerEq;

class command_buffer_test : public CtsTestBase {
 protected:
  void CreateZeroedDeviceBuffer(iree_hal_command_buffer_t* command_buffer,
                                iree_device_size_t buffer_size,
                                iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_t* device_buffer = NULL;
    IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device_),
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL, buffer_size, iree_const_byte_span_empty(),
        &device_buffer));

    uint8_t zero_val = 0x0;
    IREE_CHECK_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, device_buffer, /*target_offset=*/0,
        /*length=*/buffer_size, &zero_val,
        /*pattern_length=*/sizeof(zero_val)));
    // (buffer barrier between the fill operations)
    iree_hal_buffer_barrier_t buffer_barrier;
    buffer_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE;
    buffer_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE |
                                  IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
    buffer_barrier.buffer = device_buffer;
    buffer_barrier.offset = 0;
    buffer_barrier.length = buffer_size;
    IREE_CHECK_OK(iree_hal_command_buffer_execution_barrier(
        command_buffer, IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_TRANSFER | IREE_HAL_EXECUTION_STAGE_DISPATCH,
        IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0, NULL,
        /*buffer_barrier_count=*/1, &buffer_barrier));

    *out_buffer = device_buffer;
  }

  std::vector<uint8_t> RunFillBufferTest(iree_device_size_t buffer_size,
                                         iree_device_size_t target_offset,
                                         iree_device_size_t fill_length,
                                         const void* pattern,
                                         iree_host_size_t pattern_length) {
    iree_hal_command_buffer_t* command_buffer;
    IREE_CHECK_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        &command_buffer));
    IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

    iree_hal_buffer_t* device_buffer;
    CreateZeroedDeviceBuffer(command_buffer, buffer_size, &device_buffer);

    // Fill the pattern on top.
    IREE_CHECK_OK(iree_hal_command_buffer_fill_buffer(
        command_buffer, device_buffer, target_offset, fill_length, pattern,
        pattern_length));
    IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
    IREE_CHECK_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_ANY,
                                             command_buffer));

    // Read data for returning.
    std::vector<uint8_t> actual_data(buffer_size);
    IREE_CHECK_OK(
        iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                  /*target_buffer=*/actual_data.data(),
                                  /*data_length=*/buffer_size));

    // Cleanup and return.
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(device_buffer);
    return actual_data;
  }

  std::vector<uint8_t> RunUpdateBufferTest(
      const void* source_buffer, iree_host_size_t source_offset,
      iree_device_size_t target_buffer_size, iree_device_size_t target_offset,
      iree_device_size_t length) {
    iree_hal_command_buffer_t* command_buffer;
    IREE_CHECK_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        &command_buffer));
    IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

    iree_hal_buffer_t* device_buffer;
    CreateZeroedDeviceBuffer(command_buffer, target_buffer_size,
                             &device_buffer);

    // Issue the update_buffer command.
    IREE_CHECK_OK(iree_hal_command_buffer_update_buffer(
        command_buffer, source_buffer, source_offset, device_buffer,
        target_offset, length));
    IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
    IREE_CHECK_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_ANY,
                                             command_buffer));

    // Read data for returning.
    std::vector<uint8_t> actual_data(target_buffer_size);
    IREE_CHECK_OK(
        iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                  /*target_buffer=*/actual_data.data(),
                                  /*data_length=*/target_buffer_size));

    // Cleanup and return.
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(device_buffer);
    return actual_data;
  }

  std::vector<uint8_t> RunUpdateBufferSubspanTest(
      const void* source_buffer, iree_host_size_t source_offset,
      iree_device_size_t full_buffer_size, iree_device_size_t subspan_offset,
      iree_device_size_t target_offset, iree_device_size_t length) {
    iree_hal_command_buffer_t* command_buffer;
    IREE_CHECK_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
        &command_buffer));
    IREE_CHECK_OK(iree_hal_command_buffer_begin(command_buffer));

    iree_hal_buffer_t* full_device_buffer;
    CreateZeroedDeviceBuffer(command_buffer, full_buffer_size,
                             &full_device_buffer);

    // Slice out a subspan of the full buffer.
    iree_device_size_t subspan_length = full_buffer_size - subspan_offset;
    iree_hal_buffer_t* subspan_buffer;
    IREE_CHECK_OK(iree_hal_buffer_subspan(full_device_buffer, subspan_offset,
                                          subspan_length, &subspan_buffer));

    // Issue the update_buffer command onto the subspan buffer.
    IREE_CHECK_OK(iree_hal_command_buffer_update_buffer(
        command_buffer, source_buffer, source_offset, subspan_buffer,
        target_offset, length));
    IREE_CHECK_OK(iree_hal_command_buffer_end(command_buffer));
    IREE_CHECK_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_ANY,
                                             command_buffer));

    // Read data for returning.
    std::vector<uint8_t> actual_data(subspan_length);
    IREE_CHECK_OK(
        iree_hal_buffer_read_data(subspan_buffer, /*source_offset=*/0,
                                  /*target_buffer=*/actual_data.data(),
                                  /*data_length=*/subspan_length));

    // Cleanup and return.
    iree_hal_command_buffer_release(command_buffer);
    iree_hal_buffer_release(subspan_buffer);
    iree_hal_buffer_release(full_device_buffer);
    return actual_data;
  }

  static constexpr iree_device_size_t kBufferSize = 4096;
};

TEST_P(command_buffer_test, Create) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  EXPECT_TRUE((iree_hal_command_buffer_allowed_categories(command_buffer) &
               IREE_HAL_COMMAND_CATEGORY_DISPATCH) ==
              IREE_HAL_COMMAND_CATEGORY_DISPATCH);

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(command_buffer_test, BeginEnd) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(command_buffer_test, SubmitEmpty) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                            command_buffer));

  iree_hal_command_buffer_release(command_buffer);
}

TEST_P(command_buffer_test, CopyWholeBuffer) {
  iree_hal_command_buffer_t* command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  uint8_t i8_val = 0x54;
  std::vector<uint8_t> reference_buffer(kBufferSize);
  std::memset(reference_buffer.data(), i8_val, kBufferSize);

  // Create and fill a host buffer.
  iree_hal_buffer_t* host_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize,
      iree_make_const_byte_span(reference_buffer.data(),
                                reference_buffer.size()),
      &host_buffer));

  // Create a device buffer.
  iree_hal_buffer_t* device_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize, iree_const_byte_span_empty(),
      &device_buffer));

  // Copy the host buffer to the device buffer.
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/0,
      /*target_buffer=*/device_buffer, /*target_offset=*/0,
      /*length=*/kBufferSize));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                            command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                           /*target_buffer=*/actual_data.data(),
                                           /*data_length=*/kBufferSize));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(command_buffer_test, CopySubBuffer) {
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_TRANSFER, IREE_HAL_QUEUE_AFFINITY_ANY,
      &command_buffer));

  iree_hal_buffer_t* device_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, kBufferSize, iree_const_byte_span_empty(),
      &device_buffer));

  uint8_t i8_val = 0x88;
  std::vector<uint8_t> reference_buffer(kBufferSize);
  std::memset(reference_buffer.data() + 8, i8_val, kBufferSize / 2 - 4);

  // Create another host buffer with a smaller size.
  std::vector<uint8_t> host_buffer_data(kBufferSize, i8_val);
  iree_hal_buffer_t* host_buffer = NULL;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator_,
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_HOST_CACHED |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_ALL, host_buffer_data.size() / 2,
      iree_make_const_byte_span(host_buffer_data.data(),
                                host_buffer_data.size() / 2),
      &host_buffer));

  // Copy the host buffer to the device buffer; zero fill the untouched bytes.
  uint8_t zero_val = 0x0;
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer, /*target_offset=*/0, /*length=*/8,
      &zero_val, /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, /*source_buffer=*/host_buffer, /*source_offset=*/4,
      /*target_buffer=*/device_buffer, /*target_offset=*/8,
      /*length=*/kBufferSize / 2 - 4));
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      command_buffer, device_buffer, /*target_offset=*/8 + kBufferSize / 2 - 4,
      /*length=*/kBufferSize - (8 + kBufferSize / 2 - 4), &zero_val,
      /*pattern_length=*/sizeof(zero_val)));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(IREE_HAL_COMMAND_CATEGORY_TRANSFER,
                                            command_buffer));

  // Read the device buffer and compare.
  std::vector<uint8_t> actual_data(kBufferSize);
  IREE_ASSERT_OK(iree_hal_buffer_read_data(device_buffer, /*source_offset=*/0,
                                           /*target_buffer=*/actual_data.data(),
                                           /*data_length=*/kBufferSize));
  EXPECT_THAT(actual_data, ContainerEq(reference_buffer));

  // Must release the command buffer before resources used by it.
  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(device_buffer);
  iree_hal_buffer_release(host_buffer);
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size1_offset0_length1) {
  iree_device_size_t buffer_size = 1;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size5_offset0_length5) {
  iree_device_size_t buffer_size = 5;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 5;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset0_length1) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 1;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset0_length3) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 3;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern1_size16_offset2_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 2;
  iree_device_size_t fill_length = 8;
  uint8_t pattern = 0x07;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x07, 0x07,  //
                                        0x07, 0x07, 0x07, 0x07,  //
                                        0x07, 0x07, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size2_offset0_length2) {
  iree_device_size_t buffer_size = 2;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 2;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size16_offset0_length10) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 10;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern2_size16_offset2_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 2;
  iree_device_size_t fill_length = 8;
  uint16_t pattern = 0xAB23;
  std::vector<uint8_t> reference_buffer{0x00, 0x00, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x23, 0xAB,  //
                                        0x23, 0xAB, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern4_size4_offset0_length4) {
  iree_device_size_t buffer_size = 4;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 4;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, FillBuffer_pattern4_size16_offset0_length8) {
  iree_device_size_t buffer_size = 16;
  iree_device_size_t target_offset = 0;
  iree_device_size_t fill_length = 8;
  uint32_t pattern = 0xAB23CD45;
  std::vector<uint8_t> reference_buffer{0x45, 0xCD, 0x23, 0xAB,  //
                                        0x45, 0xCD, 0x23, 0xAB,  //
                                        0x00, 0x00, 0x00, 0x00,  //
                                        0x00, 0x00, 0x00, 0x00};
  std::vector<uint8_t> actual_buffer =
      RunFillBufferTest(buffer_size, target_offset, fill_length,
                        (void*)&pattern, sizeof(pattern));
  EXPECT_THAT(actual_buffer, ContainerEq(reference_buffer));
}

TEST_P(command_buffer_test, UpdateBuffer_sourceoffset0_targetoffset0) {
  iree_host_size_t source_offset = 0;
  iree_device_size_t target_offset = 0;
  iree_device_size_t length = 4;
  std::vector<uint8_t> source_buffer{
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> reference_target_buffer{
      0x01, 0x02, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  std::vector<uint8_t> actual_buffer =
      RunUpdateBufferTest(source_buffer.data(), source_offset,
                          target_buffer_size, target_offset, length);
  EXPECT_THAT(actual_buffer, ContainerEq(reference_target_buffer));
}

TEST_P(command_buffer_test, UpdateBuffer_sourceoffset0_targetoffset4) {
  iree_host_size_t source_offset = 0;
  iree_device_size_t target_offset = 4;
  iree_device_size_t length = 4;
  std::vector<uint8_t> source_buffer{
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> reference_target_buffer{
      0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,  //
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  std::vector<uint8_t> actual_buffer =
      RunUpdateBufferTest(source_buffer.data(), source_offset,
                          target_buffer_size, target_offset, length);
  EXPECT_THAT(actual_buffer, ContainerEq(reference_target_buffer));
}

TEST_P(command_buffer_test, UpdateBuffer_sourceoffset4_targetoffset0) {
  iree_host_size_t source_offset = 4;
  iree_device_size_t target_offset = 0;
  iree_device_size_t length = 4;
  std::vector<uint8_t> source_buffer{
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> reference_target_buffer{
      0x05, 0x06, 0x07, 0x08, 0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  std::vector<uint8_t> actual_buffer =
      RunUpdateBufferTest(source_buffer.data(), source_offset,
                          target_buffer_size, target_offset, length);
  EXPECT_THAT(actual_buffer, ContainerEq(reference_target_buffer));
}

TEST_P(command_buffer_test, UpdateBuffer_sourceoffset4_targetoffset4) {
  iree_host_size_t source_offset = 4;
  iree_device_size_t target_offset = 4;
  iree_device_size_t length = 4;
  std::vector<uint8_t> source_buffer{
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  iree_device_size_t target_buffer_size = 16;
  std::vector<uint8_t> reference_target_buffer{
      0x00, 0x00, 0x00, 0x00, 0x05, 0x06, 0x07, 0x08,  //
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

  std::vector<uint8_t> actual_buffer =
      RunUpdateBufferTest(source_buffer.data(), source_offset,
                          target_buffer_size, target_offset, length);
  EXPECT_THAT(actual_buffer, ContainerEq(reference_target_buffer));
}

// TODO(scotttodd): fix test failures here, maybe iree_hal_buffer_read_data
//                  isn't respecting subspan buffer offsets?
TEST_P(command_buffer_test,
       DISABLED_UpdateBuffer_sourceoffset0_targetoffset0_subspan4) {
  iree_host_size_t source_offset = 0;
  iree_device_size_t target_offset = 0;
  iree_device_size_t subspan_offset = 4;
  iree_device_size_t length = 4;
  std::vector<uint8_t> source_buffer{
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  iree_device_size_t full_buffer_size = 16;
  std::vector<uint8_t> reference_target_buffer{0x01, 0x02, 0x03, 0x04,
                                               0x00, 0x00, 0x00, 0x00,  //
                                               0x00, 0x00, 0x00, 0x00};

  std::vector<uint8_t> actual_buffer = RunUpdateBufferSubspanTest(
      source_buffer.data(), source_offset, full_buffer_size, target_offset,
      subspan_offset, length);
  EXPECT_THAT(actual_buffer, ContainerEq(reference_target_buffer));
}

// TODO(scotttodd): fix test failures here, maybe iree_hal_buffer_read_data
//                  isn't respecting subspan buffer offsets?
TEST_P(command_buffer_test,
       DISABLED_UpdateBuffer_sourceoffset0_targetoffset4_subspan4) {
  iree_host_size_t source_offset = 0;
  iree_device_size_t target_offset = 4;
  iree_device_size_t subspan_offset = 4;
  iree_device_size_t length = 4;
  std::vector<uint8_t> source_buffer{
      0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,  //
      0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18};

  iree_device_size_t full_buffer_size = 16;
  std::vector<uint8_t> reference_target_buffer{0x00, 0x00, 0x00, 0x00,
                                               0x01, 0x02, 0x03, 0x04,  //
                                               0x00, 0x00, 0x00, 0x00};

  std::vector<uint8_t> actual_buffer = RunUpdateBufferSubspanTest(
      source_buffer.data(), source_offset, full_buffer_size, target_offset,
      subspan_offset, length);
  EXPECT_THAT(actual_buffer, ContainerEq(reference_target_buffer));
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_COMMAND_BUFFER_TEST_H_
