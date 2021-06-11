// RUN: iree-opt -iree-flow-convert-to-flow-tensor-ops-pass -canonicalize -cse -split-input-file %s | IreeFileCheck %s

func @subtensor1(%arg0 : tensor<5x24x48xf32>) -> tensor<4xf32> {
  %0 = subtensor %arg0[2, 3, 4] [1, 1, 4] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func @subtensor1(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>)
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C4]] for %[[C1]], %[[C1]], %[[C4]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   return %[[RESULT]]

// -----

func @subtensor2(%arg0 : tensor<5x24x48xf32>) -> tensor<2x48xf32> {
  %0 = subtensor %arg0[2, 3, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x48xf32>
  return %0 : tensor<2x48xf32>
}
// CHECK-LABEL: func @subtensor2
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x24x48xf32>)
//   CHECK-DAG:   %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[C48:.+]] = constant 48 : index
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C0]] for %[[C1]], %[[C2]], %[[C48]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   return %[[RESULT]]

// -----

func @subtensor3(%arg0 : tensor<5x24x48xf32>) -> tensor<2x24xf32> {
  %0 = subtensor %arg0[2, 3, 0] [1, 2, 24] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x24xf32>
  return %0 : tensor<2x24xf32>
}
// CHECK-LABEL: func @subtensor3
//       CHECK:   subtensor

// -----

func @subtensor4(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<2x24xf32> {
  %0 = subtensor %arg0[2, 3, 0] [1, 2, 24] [1, %arg1, 1]
      : tensor<5x24x48xf32> to tensor<2x24xf32>
  return %0 : tensor<2x24xf32>
}
// CHECK-LABEL: func @subtensor4
//       CHECK:   subtensor

// -----

func @subtensor5(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<2x48xf32> {
  %0 = subtensor %arg0[2, %arg1, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<2x48xf32>
  return %0 : tensor<2x48xf32>
}
// CHECK-LABEL: func @subtensor5
//       CHECK:   subtensor

// -----

func @subtensor6(%arg0 : tensor<5x24x48xf32>, %arg1 : index) -> tensor<?x48xf32> {
  %0 = subtensor %arg0[2, 3, 0] [1, %arg1, 48] [1, 1, 1]
      : tensor<5x24x48xf32> to tensor<?x48xf32>
  return %0 : tensor<?x48xf32>
}
// CHECK-LABEL: func @subtensor6
//       CHECK:   subtensor

// -----

func @subtensor7(%arg0 : tensor<5x?x48xf32>, %arg1 : index) -> tensor<2x48xf32> {
  %0 = subtensor %arg0[2, 3, 0] [1, 2, 48] [1, 1, 1]
      : tensor<5x?x48xf32> to tensor<2x48xf32>
  return %0 : tensor<2x48xf32>
}
// CHECK-LABEL: func @subtensor7(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<5x?x48xf32>
//  CHECK-SAME:   %[[ARG1:.+]]: index)
//   CHECK-DAG:   %[[C2:.+]] = constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = constant 3 : index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C48:.+]] = constant 48 : index
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C1]] : tensor<5x?x48xf32>
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]][%[[C2]], %[[C3]], %[[C0]] for %[[C1]], %[[C2]], %[[C48]]]
//       CHECK:   %[[RESULT:.+]] = flow.tensor.reshape %[[SLICE]]
//       CHECK:   return %[[RESULT]]

// -----

func @rank_reducing_subtensor(%arg0: tensor<?x513xi32>) -> tensor<513xi32> {
  %0 = subtensor %arg0[4, 0] [1, 513] [1, 1] : tensor<?x513xi32> to tensor<513xi32>
  return %0 : tensor<513xi32>
}
// CHECK-LABEL: func @rank_reducing_subtensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = constant 4 : index
//   CHECK-DAG:   %[[C513:.+]] = constant 513 : index
//       CHECK:   %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[SLICE:.+]] = flow.tensor.slice %[[ARG0]]
//  CHECK-SAME:       [%[[C4]], %[[C0]] for %[[C1]], %[[C513]]]
//  CHECK-SAME:       : tensor<?x513xi32>{%[[DIM]]} -> tensor<1x513xi32>
//       CHECK:   %[[RESHAPE:.+]] = flow.tensor.reshape %[[SLICE]] : tensor<1x513xi32> -> tensor<513xi32>
//       CHECK:   return %[[RESHAPE]] : tensor<513xi32>

// -----

func @tensor_reshape(%arg0 : tensor<?x4x?x5x?x6xf32>, %arg1 : tensor<20x?x40xf32>)
    -> (tensor<?x5x?xf32>, tensor<5x4x?x4x2x4x5xf32>)
{
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1, 2], [3], [4, 5]]
      : tensor<?x4x?x5x?x6xf32> into tensor<?x5x?xf32>
  %1 = linalg.tensor_expand_shape %arg1 [[0, 1], [2, 3], [4, 5, 6]]
      : tensor<20x?x40xf32> into tensor<5x4x?x4x2x4x5xf32>
  return %0, %1 : tensor<?x5x?xf32>, tensor<5x4x?x4x2x4x5xf32>
}
// CHECK-LABEL: func @tensor_reshape
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x4x?x5x?x6xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<20x?x40xf32>
//   CHECK-DAG:   %[[R0:.+]] = flow.tensor.reshape %[[ARG0]]
//   CHECK-DAG:   %[[R1:.+]] = flow.tensor.reshape %[[ARG1]]
//       CHECK:   return %[[R0]], %[[R1]]

// -----

func @subtensor_insert_convert
    (%arg0 : tensor<?x24x48xf32>, %arg1 : tensor<1x4x48xf32>) ->
    tensor<?x24x48xf32> {
  %c0 = constant 0 : index
  %0 = subtensor_insert %arg1 into %arg0[4, 2, 0] [1, 4, 48] [1, 1, 1] :
      tensor<1x4x48xf32> into tensor<?x24x48xf32>
  return %0 : tensor<?x24x48xf32>
}
// CHECK-LABEL: func @subtensor_insert_convert
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[DIM0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[ARG1]], %[[ARG0]][%[[C4]], %[[C2]], %[[C0]]]
//  CHECK-SAME:     : tensor<1x4x48xf32> -> tensor<?x24x48xf32>{%[[DIM0]]}

// -----

func @subtensor_insert_convert_rank_reducing
    (%arg0 : tensor<?x24x48xf32>, %arg1 : tensor<4x48xf32>) ->
    tensor<?x24x48xf32> {
  %c0 = constant 0 : index
  %0 = subtensor_insert %arg1 into %arg0[4, 2, 0] [1, 4, 48] [1, 1, 1] :
      tensor<4x48xf32> into tensor<?x24x48xf32>
  return %0 : tensor<?x24x48xf32>
}
// CHECK-LABEL: func @subtensor_insert_convert_rank_reducing
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = constant 0
//   CHECK-DAG:   %[[C2:.+]] = constant 2
//   CHECK-DAG:   %[[C4:.+]] = constant 4
//   CHECK-DAG:   %[[RESHAPE:.+]] = flow.tensor.reshape %[[ARG1]] : tensor<4x48xf32> -> tensor<1x4x48xf32>
//   CHECK-DAG:   %[[DIM:.+]] = memref.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.update %[[RESHAPE]], %[[ARG0]][%[[C4]], %[[C2]], %[[C0]]]
//  CHECK-SAME:     : tensor<1x4x48xf32> -> tensor<?x24x48xf32>{%[[DIM]]}

// -----

func @rank_reducing_subtensor_insert_trailing_unit_dims
   (%arg0 : tensor<49x20xf32>, %arg1 : tensor<1x50x20x1xf32>) -> tensor<1x50x20x1xf32> {
  %0 = subtensor_insert %arg0 into %arg1[0, 1, 0, 0] [1, 49, 20, 1] [1, 1, 1, 1] : tensor<49x20xf32> into tensor<1x50x20x1xf32>
  return %0 : tensor<1x50x20x1xf32>
}
// CHECK-LABEL: func @rank_reducing_subtensor_insert_trailing_unit_dims
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//       CHECK:   %[[RESHAPE:.+]] = flow.tensor.reshape %{{.+}} : tensor<49x20xf32> -> tensor<1x49x20x1xf32>
//       CHECK:   flow.tensor.update %[[RESHAPE]], %{{.+}}[%[[C0]], %[[C1]], %[[C0]], %[[C0]]] : tensor<1x49x20x1xf32> -> tensor<1x50x20x1xf32>

// -----

func @rank_reducing_subtensor_trailing_unit_dims
   (%arg0 : tensor<1x50x20x1xf32>) -> tensor<49x20xf32> {
  %0 = subtensor %arg0[0, 1, 0, 0] [1, 49, 20, 1] [1, 1, 1, 1] : tensor<1x50x20x1xf32> to tensor<49x20xf32>
  return %0 : tensor<49x20xf32>
}
// CHECK-LABEL: func @rank_reducing_subtensor_trailing_unit_dims
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//   CHECK-DAG:   %[[C49:.+]] = constant 49 : index
//   CHECK-DAG:   %[[C20:.+]] = constant 20 : index
//       CHECK:   %[[SUBTENSOR:.+]] = flow.tensor.slice %{{.+}}[%[[C0]], %[[C1]], %[[C0]], %[[C0]] for %[[C1]], %[[C49]], %[[C20]], %[[C1]]] : tensor<1x50x20x1xf32> -> tensor<1x49x20x1xf32>
//       CHECK:   flow.tensor.reshape %[[SUBTENSOR]] : tensor<1x49x20x1xf32> -> tensor<49x20xf32>

// -----

func @turn_fill_into_splat_if_no_linalg_users(%arg0: tensor<?x?xf32>, %arg1: tensor<f32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = tensor.extract %arg1[] : tensor<f32>
  %1 = memref.dim %arg0, %c0 : tensor<?x?xf32>
  %2 = memref.dim %arg0, %c1 : tensor<?x?xf32>
  %3 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%1)[%arg2, %arg4]
  %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 + s1)>(%2)[%arg3, %arg5]
  %5 = linalg.init_tensor [%3, %4] : tensor<?x?xf32>
  %6 = linalg.fill(%5, %0) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %7 = flow.tensor.update %arg0, %6[%arg2, %arg3] : tensor<?x?xf32>{%1, %2} -> tensor<?x?xf32>{%3, %4}
  return %7 : tensor<?x?xf32>
}

//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
//       CHECK: func @turn_fill_into_splat_if_no_linalg_users
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<f32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG4:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:   %[[ARG5:[a-zA-Z0-9]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = constant 1 : index
//       CHECK:   %[[VAL:.+]] = tensor.extract %[[ARG1]][]
//   CHECK-DAG:   %[[D0:.+]] = memref.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[RD0:.+]] = affine.apply #[[MAP]]()[%[[ARG2]], %[[ARG4]], %[[D0]]]
//   CHECK-DAG:   %[[RD1:.+]] = affine.apply #[[MAP]]()[%[[ARG3]], %[[ARG5]], %[[D1]]]
// 	     CHECK:   %[[SPLAT:.+]] = flow.tensor.splat %[[VAL]] : tensor<?x?xf32>{%[[RD0]], %[[RD1]]}
//       CHECK:   flow.tensor.update %[[ARG0]], %[[SPLAT]]

// -----

func @keep_fill_if_with_linalg_users(%A : tensor<?x?xf32>, %B : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %zero = constant 0.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = memref.dim %A, %c0 : tensor<?x?xf32>
  %N = memref.dim %B, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor [%M, %N] : tensor<?x?xf32>
  %1 = linalg.fill(%0, %zero) : tensor<?x?xf32>, f32 -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func @keep_fill_if_with_linalg_users
// 		   CHECK:   linalg.fill
