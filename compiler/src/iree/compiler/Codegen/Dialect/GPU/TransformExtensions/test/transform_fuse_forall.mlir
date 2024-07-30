// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %0 = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %0) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//   CHECK-DAG:     %[[OUTID0:.+]] = affine.apply #[[$MAP]](%[[IDX]])
//   CHECK-DAG:     %[[OUTID1:.+]] = affine.apply #[[$MAP]](%[[IDY]])
//       CHECK:     %[[LINEARID:.+]] = affine.apply #[[$MAP1]](%[[IDX]], %[[IDY]])
//       CHECK:     %[[IDS:.+]]:2 = affine.delinearize_index %[[LINEARID]] into (%c64, %c1) : index, index
//       CHECK:     %[[INID0:.+]] = affine.apply #[[$MAP2]](%[[IDS]]#0)
//       CHECK:     %[[INSLICE0:.+]] = tensor.extract_slice %[[ARG0]][%[[INID0]], %[[IDS]]#1] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
//       CHECK:     %[[INSLICE1:.+]] = tensor.extract_slice %[[EMPTY]][%[[INID0]], %[[IDS]]#1] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
//       CHECK:     %[[COPY:.+]] = linalg.copy ins(%[[INSLICE0]] : tensor<2x128xf32>) outs(%[[INSLICE1]] : tensor<2x128xf32>) -> tensor<2x128xf32>
//       CHECK:     %[[SHUFFLE:.+]] = iree_gpu.shuffle_tensor %[[COPY]][%[[INID0]], %[[IDS]]#1] [2, 128] [1, 1] to %[[ALLOC]]
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<128x128xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][%[[OUTID0]], %[[OUTID1]]] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]]
//       CHECK:     } : tensor<2x128xf32> -> tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:     %[[OUTSLICE:.+]] = tensor.extract_slice %[[INIT]][%[[OUTID0]], %[[OUTID1]]] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
//       CHECK:     %[[MM:.+]] = linalg.matmul ins(%[[SHUFFLE]], %[[SHUFFLE]] : tensor<16x16xf32>, tensor<16x16xf32>)
//  CHECK-SAME:       outs(%[[OUTSLICE]] : tensor<16x16xf32>) -> tensor<16x16xf32>
//       CHECK:     scf.forall.in_parallel {
//       CHECK:       tensor.parallel_insert_slice %[[MM]] into %[[INIT]][%[[OUTID0]], %[[OUTID1]]] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
//       CHECK:     }
//       CHECK:   } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_forall
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//       CHECK:     %[[SHUFFLE:.+]] = iree_gpu.shuffle_tensor %{{.*}} to %[[ALLOC]]
//       CHECK:       } : tensor<2x128xf32> -> tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:   } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @fuse_forall_with_reshape(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    %expand = tensor.expand_shape %2 [[0, 1], [2]] output_shape [2, 64, 128] : tensor<128x128xf32> into tensor<2x64x128xf32>
    %3 = scf.forall (%arg5, %arg6) in (8, 8) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map1(%arg6)
      %extracted_slice_0 = tensor.extract_slice %expand[0, %6, %7] [1, 16, 16] [1, 1, 1] : tensor<2x64x128xf32> to tensor<16x16xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
      %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1) -> (d0 * 8 + d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_forall_with_reshape
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[IDX:.+]], %[[IDY:.+]]) in (8, 8) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//       CHECK:     %[[SHUFFLE:.+]] = iree_gpu.shuffle_tensor %{{.*}} to %[[ALLOC]]
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<128x128xf32>):
//       CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[INTERMEDIATE]] {{\[}}[0, 1], [2]{{\]}} output_shape [2, 64, 128]
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[EXPAND]][0, %{{.*}}, %{{.*}}] [1, 16, 16] [1, 1, 1] : tensor<2x64x128xf32> to tensor<16x16xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]]
//       CHECK:       } : tensor<2x128xf32> -> tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:   } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}

// -----

#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d1 + d0 * 16)>
module {
  func.func @fuse_thread_forall_with_warp_and_lane(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %2 = scf.forall (%arg5, %arg6) in (64, 1) shared_outs(%arg7 = %empty) -> (tensor<128x128xf32>) {
      %4 = affine.apply #map(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<128x128xf32> to tensor<2x128xf32>
      %5 = linalg.copy ins(%extracted_slice : tensor<2x128xf32>) outs(%extracted_slice_0 : tensor<2x128xf32>) -> tensor<2x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg7[%4, %arg6] [2, 128] [1, 1] : tensor<2x128xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    %3 = scf.forall (%arg9, %arg10) in (2, 2) shared_outs(%arg8 = %empty) -> (tensor<128x128xf32>) {
      %extracted_slice_2 = tensor.extract_slice %arg8[%arg9, %arg10] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
      %9 = scf.forall (%arg5, %arg6) in (4, 4) shared_outs(%arg7 = %extracted_slice_2) -> (tensor<64x64xf32>) {
        %6 = affine.apply #map1(%arg5, %arg9)
        %7 = affine.apply #map1(%arg6, %arg10)
        %extracted_slice_0 = tensor.extract_slice %2[%6, %7] [16, 16] [1, 1] : tensor<128x128xf32> to tensor<16x16xf32>
        %extracted_slice_1 = tensor.extract_slice %arg7[%6, %7] [16, 16] [1, 1] : tensor<64x64xf32> to tensor<16x16xf32>
        %8 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_0 : tensor<16x16xf32>, tensor<16x16xf32>) outs(%extracted_slice_1 : tensor<16x16xf32>) -> tensor<16x16xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %8 into %arg7[%6, %7] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<64x64xf32>
        }
      } {mapping = [#iree_gpu.lane_id<1>, #iree_gpu.lane_id<0>]}
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg8[%arg9, %arg10] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
      }
    } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
    return %3 : tensor<128x128xf32>
  }
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.forall"]} in %root : (!transform.any_op) -> !transform.any_op
    %producer, %lane_consumer, %warp_consumer = transform.split_handle %loops : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.iree.fuse_forall %producer into %lane_consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 8 + d2 * 4)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 + d1 * 4 + d2 * 32 + d3 * 16)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0) -> (d0 * 2)>

// CHECK-LABEL: func @fuse_thread_forall_with_warp_and_lane
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<128x128xf32>

//   CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
//   CHECK-DAG:   %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<128x128xf32>
//       CHECK:   scf.forall (%[[W_IDX:.+]], %[[W_IDY:.+]]) in (2, 2) shared_outs(%[[INIT:.+]] = %[[EMPTY]]) -> (tensor<128x128xf32>) {
//       CHECK:     scf.forall (%[[L_IDX:.+]], %[[L_IDY:.+]]) in (4, 4) {{.*}} -> (tensor<64x64xf32>)
//   CHECK-DAG:       %[[FLAT_ID:.+]] = affine.apply #[[$MAP3]](%[[L_IDY]], %[[L_IDX]], %[[W_IDX]], %[[W_IDY]])
//   CHECK-DAG:       %[[IDS:.+]]:2 = affine.delinearize_index %[[FLAT_ID]] into (%c64, %c1) : index, index
//   CHECK-DAG:       %[[IDX:.+]] = affine.apply #[[$MAP4]](%[[IDS]]#0)
//       CHECK:       %[[COPY:.+]] = linalg.copy
//       CHECK:       %[[SHUFFLE:.+]] = iree_gpu.shuffle_tensor
//  CHECK-SAME:         %[[COPY]][%[[IDX]], %[[IDS]]#1] [2, 128] [1, 1] to %[[ALLOC]]
//       CHECK:       } : tensor<2x128xf32> -> tensor<128x128xf32> -> tensor<16x16xf32>
//       CHECK:     } {mapping = [#iree_gpu.lane_id<1>, #iree_gpu.lane_id<0>]}
//       CHECK:   } {mapping = [#gpu.warp<y>, #gpu.warp<x>]}
