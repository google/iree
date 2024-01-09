// RUN: iree-opt --iree-codegen-generic-vectorization=enable-vector-masking \
// RUN:          --iree-codegen-optimize-vector-transfer %s | FileCheck --implicit-check-not=mask %s

// The following loop nest was generated by tiling a linalg.matmul operation,
// followed by peeling. Only the main loop is included here. Given that
// scalable tiling was used, some dimensions are dynamic and hence the need for
// masked vectorisation. However, note that all tile sizes in the main loop
// after peeling are guaranteed to match the tile sizes, i.e. for the example
// below the following always holds:
//    * ? = vscale * 16
// Hence, masking is not really needed. Vector mask canonicalisations are
// capable of recognising that and remove the masks inserted by the Linalg
// vectoriser.
//
// Following vectorization, the hoisting logic recognises that there's no need
// to store/load the output tensor in the most inner loop and instead hoists it
// so that it becomes a loop variable. This is highly desirable as the opposite
// leads to very poor performance.

// CHECK-LABEL: @pipeline()
// CHECK:       scf.for {{.*}} iter_args(%[[OUT_TENSOR:.*]] = {{.*}}) -> (tensor<1024x1024xf32>) {
// CHECK-NEXT:    scf.for {{.*}} iter_args(%[[OUT_TENSOR_1:.*]] = %[[OUT_TENSOR]]) -> (tensor<1024x1024xf32>) {
// CHECK-NEXT:      %[[OUT_SLICE:.*]] = tensor.extract_slice %[[OUT_TENSOR_1]]{{.*}} : tensor<1024x1024xf32> to tensor<8x?xf32>
// CHECK-NEXT:      %[[OUT_SLICE_1:.*]] = tensor.extract_slice %[[OUT_SLICE]]{{.*}} : tensor<8x?xf32> to tensor<8x?xf32>
// CHECK-NEXT:      %[[OUT_VEC:.*]] = vector.transfer_read %[[OUT_TENSOR_1]]{{.*}} : tensor<1024x1024xf32>, vector<8x[16]xf32>
// CHECK-NEXT:      %[[INNER_LOOP:.*]]:3 = scf.for {{.*}} iter_args({{.*}}, %[[RES:.*]] = %[[OUT_VEC]]) -> (tensor<8x?xf32>, tensor<8x?xf32>, vector<8x[16]xf32>) {
// CHECK-NEXT:        %[[LHS:.*]] = vector.transfer_read {{.*}} : tensor<1024x1024xf32>, vector<8x1xf32>
// CHECK-NEXT:        %[[RHS:.*]] = vector.transfer_read {{.*}} : tensor<1024x1024xf32>, vector<1x[16]xf32>
// CHECK-NEXT:        %[[CONTRACT:.*]] = vector.contract {indexing_maps = [#map1, #map2, #map3],
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} 
// CHECK-SAME:      %[[LHS]], %[[RHS]], %[[RES]] : vector<8x1xf32>, vector<1x[16]xf32> into vector<8x[16]xf32>
// CHECK-NEXT:        scf.yield {{.*}}, %[[CONTRACT]] : tensor<8x?xf32>, tensor<8x?xf32>, vector<8x[16]xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[OUT_WRITE:.*]] = vector.transfer_write %[[INNER_LOOP]]#2, %[[INNER_LOOP]]#1{{.*}} {{.*}} : vector<8x[16]xf32>, tensor<8x?xf32>
// CHECK-NEXT:  %[[INSERT_SLICE:.*]] = tensor.insert_slice %[[OUT_WRITE]] into %[[INNER_LOOP]]#0{{.*}} : tensor<8x?xf32> into tensor<8x?xf32>
// CHECK-NEXT:  tensor.insert_slice %[[INSERT_SLICE]] into %[[OUT_TENSOR_1]]{{.*}} : tensor<8x?xf32> into tensor<1024x1024xf32>

func.func @pipeline() {
  %c1 = arith.constant 1 : index
  %c1024 = arith.constant 1024 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1024x1024xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<1024x1024xf32>> -> tensor<1024x1024xf32>
  %6 = vector.vscale
  %7 = arith.muli %6, %c16 : index
  %8 = scf.for %arg0 = %c0 to %c1024 step %c8 iter_args(%arg1 = %5) -> (tensor<1024x1024xf32>) {
    // This affine expression guarantees that every iteration in the following
    // loop will process exactly %6 * %c16, i.e. vscale * 16, elements.
    %9 = affine.apply affine_map<()[s0] -> (-(1024 mod s0) + 1024)>()[%7]
    %10 = scf.for %arg2 = %c0 to %9 step %7 iter_args(%arg3 = %arg1) -> (tensor<1024x1024xf32>) {
      %extracted_slice = tensor.extract_slice %3[%arg0, 0] [8, 1024] [1, 1] : tensor<1024x1024xf32> to tensor<8x1024xf32>
      %extracted_slice_0 = tensor.extract_slice %4[0, %arg2] [1024, %7] [1, 1] : tensor<1024x1024xf32> to tensor<1024x?xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg0, %arg2] [8, %7] [1, 1] : tensor<1024x1024xf32> to tensor<8x?xf32>
      %12 = scf.for %arg4 = %c0 to %c1024 step %c1 iter_args(%arg5 = %extracted_slice_1) -> (tensor<8x?xf32>) {
        %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg4] [8, 1] [1, 1] : tensor<8x1024xf32> to tensor<8x1xf32>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg4, 0] [1, %7] [1, 1] : tensor<1024x?xf32> to tensor<1x?xf32>
        %extracted_slice_4 = tensor.extract_slice %arg5[0, 0] [8, %7] [1, 1] : tensor<8x?xf32> to tensor<8x?xf32>
        %13 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [8, [16], 0], [0, 0, 1], [0, 0, 0]]>} ins(%extracted_slice_2, %extracted_slice_3 : tensor<8x1xf32>, tensor<1x?xf32>) outs(%extracted_slice_4 : tensor<8x?xf32>) -> tensor<8x?xf32>
        %inserted_slice_5 = tensor.insert_slice %13 into %arg5[0, 0] [8, %7] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
        scf.yield %inserted_slice_5 : tensor<8x?xf32>
      }
      %inserted_slice = tensor.insert_slice %12 into %arg3[%arg0, %arg2] [8, %7] [1, 1] : tensor<8x?xf32> into tensor<1024x1024xf32>
      scf.yield %inserted_slice : tensor<1024x1024xf32>
    }
    scf.yield %10 : tensor<1024x1024xf32>
  }
  flow.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xf32> -> !flow.dispatch.tensor<readwrite:tensor<1024x1024xf32>>
  return
}

