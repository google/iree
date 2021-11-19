// RUN: iree-opt -split-input-file -iree-spirv-vectorize %s | IreeFileCheck %s

func @add(%lhs: tensor<2x8xf32>, %rhs: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %init = linalg.init_tensor [2, 8] : tensor<2x8xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<2x8xf32>, tensor<2x8xf32>)
    outs(%init : tensor<2x8xf32>) {
        ^bb0(%a : f32, %b : f32, %c : f32):
        %add = arith.addf %a, %b : f32
        %mul = arith.mulf %a, %b : f32
        %sub = arith.subf %add, %mul: f32
        linalg.yield %sub : f32
  } -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

//   CHECK-LABEL: func @add
// CHECK-COUNT-8:   vector.transfer_read %{{.+}} : tensor<2x8xf32>, vector<4xf32>
// CHECK-COUNT-4:   arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-4:   arith.mulf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-4:   arith.subf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-COUNT-4:   vector.transfer_write {{.*}} : vector<4xf32>, tensor<2x8xf32>

// -----

func @transpose_add(%lhs: tensor<4x2xf32>, %rhs: tensor<2xf32>) -> tensor<2x4xf32> {
  %init = linalg.init_tensor [2, 4] : tensor<2x4xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %rhs : tensor<4x2xf32>, tensor<2xf32>)
    outs(%init : tensor<2x4xf32>) {
        ^bb0(%a : f32, %b : f32, %c : f32):
        %add = arith.addf %a, %b : f32
        linalg.yield %add : f32
  } -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @transpose_add
//  CHECK-SAME: (%[[LHS:.+]]: tensor<4x2xf32>, %[[RHS:.+]]: tensor<2xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[RINIT:.+]] = arith.constant dense<0.000000e+00> : vector<4x2xf32>
//       CHECK:   %[[OINIT:.+]] = linalg.init_tensor [2, 4] : tensor<2x4xf32>
//       CHECK:   %[[LREAD:.+]] = vector.transfer_read %[[LHS]]{{.+}} : tensor<4x2xf32>, vector<4x2xf32>
//       CHECK:   %[[LT:.+]] = vector.transpose %[[LREAD]], [1, 0] : vector<4x2xf32> to vector<2x4xf32>
//       CHECK:   %[[READ:.+]] = vector.transfer_read %[[RHS]]{{.+}} : tensor<2xf32>, vector<2xf32>
//       CHECK:   %[[INSERT0:.+]] = vector.insert %[[READ]], %[[RINIT]] [0] : vector<2xf32> into vector<4x2xf32>
//       CHECK:   %[[INSERT1:.+]] = vector.insert %[[READ]], %[[INSERT0]] [1] : vector<2xf32> into vector<4x2xf32>
//       CHECK:   %[[INSERT2:.+]] = vector.insert %[[READ]], %[[INSERT1]] [2] : vector<2xf32> into vector<4x2xf32>
//       CHECK:   %[[INSERT3:.+]] = vector.insert %[[READ]], %[[INSERT2]] [3] : vector<2xf32> into vector<4x2xf32>
//       CHECK:   %[[RT:.+]] = vector.transpose %[[INSERT3]], [1, 0] : vector<4x2xf32> to vector<2x4xf32>
//       CHECK:   %[[EXTRACT0:.+]] = vector.extract_strided_slice %[[LT]] {offsets = [0, 0]
//       CHECK:   %[[EXTRACT1:.+]] = vector.extract_strided_slice %[[RT]] {offsets = [0, 0]
//       CHECK:   %[[CAST0:.+]] = vector.shape_cast %[[EXTRACT0]]
//       CHECK:   %[[CAST1:.+]] = vector.shape_cast %[[EXTRACT1]]
//       CHECK:   %[[ADD0:.+]] = arith.addf %[[CAST0]], %[[CAST1]] : vector<4xf32>
//       CHECK:   %[[EXTRACT0:.+]] = vector.extract_strided_slice %[[LT]] {offsets = [1, 0]
//       CHECK:   %[[EXTRACT1:.+]] = vector.extract_strided_slice %[[RT]] {offsets = [1, 0]
//       CHECK:   %[[CAST0:.+]] = vector.shape_cast %[[EXTRACT0]]
//       CHECK:   %[[CAST1:.+]] = vector.shape_cast %[[EXTRACT1]]
//       CHECK:   %[[ADD1:.+]] = arith.addf %[[CAST0]], %[[CAST1]] : vector<4xf32>
//       CHECK:   %[[WRITE0:.+]] = vector.transfer_write %[[ADD0]], %[[OINIT]][%[[C0]], %[[C0]]]
//       CHECK:   %[[WRITE1:.+]] = vector.transfer_write %[[ADD1]], %[[WRITE0]][%[[C1]], %[[C0]]]
//       CHECK:   return %[[WRITE1]]
