// RUN: iree-opt -iree-util-test-partitionable-loops-interface -split-input-file %s | IreeFileCheck %s

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
func @generic_dynamic(%arg0 : tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %init = linalg.init_tensor [%d0, %d2] : tensor<?x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0: tensor<?x?x?xf32>) outs(%init : tensor<?x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @generic_dynamic(
//       CHECK:   util.unfoldable_constant dense<[1, 0, 1]> : tensor<3xindex>

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
func @generic_unit_dim(%arg0 : tensor<1x?x?xf32>) -> tensor<1x?xf32> {
  %c2 = arith.constant 2 : index
  %d2 = tensor.dim %arg0, %c2 : tensor<1x?x?xf32>
  %init = linalg.init_tensor [1, %d2] : tensor<1x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0: tensor<1x?x?xf32>) outs(%init : tensor<1x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}
// CHECK-LABEL: func @generic_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[0, 0, 1]> : tensor<3xindex>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_4D(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf32>
  %d2 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
  %d3 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
  %init = linalg.init_tensor [%d0, %d1, %d2, %d3] : tensor<?x?x?x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<?x?x?x?xf32>) outs(%init : tensor<?x?x?x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32> 
}
// CHECK-LABEL: func @generic_4D(
//       CHECK:   util.unfoldable_constant dense<[0, 1, 1, 1]> : tensor<4xindex>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_4D_unit_dim(%arg0: tensor<?x?x1x?xf32>) -> tensor<?x?x1x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x1x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x1x?xf32>
  %d3 = tensor.dim %arg0, %c3 : tensor<?x?x1x?xf32>
  %init = linalg.init_tensor [%d0, %d1, 1, %d3] : tensor<?x?x1x?xf32>
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%arg0: tensor<?x?x1x?xf32>) outs(%init : tensor<?x?x1x?xf32>)
    attrs = {__test_interface__ = true} {
      ^bb0(%arg1 : f32, %arg2 : f32):
        linalg.yield %arg1 : f32
    } -> tensor<?x?x1x?xf32>
  return %0 : tensor<?x?x1x?xf32> 
}
// CHECK-LABEL: func @generic_4D_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[1, 1, 0, 1]> : tensor<4xindex>

// -----

func @named_op(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @named_op(
//       CHECK:   util.unfoldable_constant dense<[1, 1, 0]> : tensor<3xindex>

// -----

func @named_op_unit_dim(%lhs : tensor<1x?xf32>, %rhs : tensor<?x?xf32>,
    %init : tensor<1x?xf32>) -> tensor<1x?xf32> {
  %0 = linalg.matmul {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<1x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<1x?xf32>) -> tensor<1x?xf32>
  return %0 : tensor<1x?xf32>
}


// CHECK-LABEL: func @named_op_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[0, 1, 0]> : tensor<3xindex>

// -----

func @mmt4d(%lhs : tensor<?x?x?x?xf32>, %rhs : tensor<?x?x?x?xf32>,
    %init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = linalg.mmt4d {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%init : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func @mmt4d(
//       CHECK:   util.unfoldable_constant dense<[1, 1, 0, 0, 0, 0]> : tensor<6xindex>

// -----

func @mmt4d_unit_dim(%lhs : tensor<1x?x?x?xf32>, %rhs : tensor<?x?x?x?xf32>,
    %init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32> {
  %0 = linalg.mmt4d {__test_interface__ = true}
      ins(%lhs, %rhs : tensor<1x?x?x?xf32>, tensor<?x?x?x?xf32>)
      outs(%init : tensor<1x?x?x?xf32>) -> tensor<1x?x?x?xf32>
  return %0 : tensor<1x?x?x?xf32>
}
// CHECK-LABEL: func @mmt4d_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[1, 1, 0, 0, 0, 0]> : tensor<6xindex>

// -----

func @sort(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.sort dimension(0)
      {__test_interface__ = true}
      outs(%arg0 : tensor<?x?xf32>) {
        ^bb0(%arg1 : f32, %arg2 : f32):
          %1  = arith.cmpf ogt, %arg1, %arg2 : f32
          iree_linalg_ext.yield %1 : i1
      } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @sort(
//       CHECK:   util.unfoldable_constant dense<[0, 1]> : tensor<2xindex>

// -----

func @sort_unit_dim(%arg0 : tensor<?x1xf32>) -> tensor<?x1xf32> {
  %0 = iree_linalg_ext.sort dimension(0)
      {__test_interface__ = true}
      outs(%arg0 : tensor<?x1xf32>) {
        ^bb0(%arg1 : f32, %arg2 : f32):
          %1  = arith.cmpf ogt, %arg1, %arg2 : f32
          iree_linalg_ext.yield %1 : i1
      } -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}
// CHECK-LABEL: func @sort_unit_dim(
//       CHECK:   util.unfoldable_constant dense<[0, 1]> : tensor<2xindex>
