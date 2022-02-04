// RUN: iree-opt -allow-unregistered-dialect -split-input-file -iree-flow-outline-dispatch-regions %s | FileCheck %s

//      CHECK: flow.executable private @staticShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.dispatch.entry public @staticShapeDispatch_dispatch_0 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @staticShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG:.+]]: !flow.dispatch.tensor<readonly:8x4xf32>,
// CHECK-SAME:     %[[RET:.+]]: !flow.dispatch.tensor<writeonly:4x8xf32>) {
//  CHECK-DAG:   %[[ARG_VALUE:.+]] = flow.dispatch.tensor.load %[[ARG]], {{.*}} : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
// CHECK-NEXT:   %[[RET_VALUE:.+]] = "test.sink"(%[[ARG_VALUE]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   flow.dispatch.tensor.store %[[RET_VALUE]], %[[RET]], {{.*}} : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK-LABEL: func @staticShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func @staticShapeDispatch(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET:.+]] = flow.dispatch @staticShapeDispatch_dispatch_0::@staticShapeDispatch_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> tensor<4x8xf32> = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
    %ret_value = "test.sink"(%arg_value) : (tensor<8x4xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret,  offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
    flow.return
  }
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_0
// CHECK-NEXT:   flow.dispatch.entry public @dispatchFnMuli_dispatch_0 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @dispatchFnMuli_dispatch_0(

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_1
// CHECK-NEXT:   flow.dispatch.entry public @dispatchFnMuli_dispatch_1 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @dispatchFnMuli_dispatch_1(

// CHECK-LABEL: func @dispatchFnMuli(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func @dispatchFnMuli(%arg0 : tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET0:.+]] = flow.dispatch @dispatchFnMuli_dispatch_0::@dispatchFnMuli_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:8x4xf32> -> tensor<8x4xf32>
    %ret_value = "test.sink1"(%arg_value) : (tensor<8x4xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:4x8xf32>
    flow.return
  }
  // CHECK: %[[RET1:.+]] = flow.dispatch @dispatchFnMuli_dispatch_1::@dispatchFnMuli_dispatch_1[
  // CHECK-SAME: %[[Y]], %[[X]]
  // CHECK-SAME: ](%[[RET0]]) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = flow.dispatch.workgroups[%y, %x](%0) : (tensor<4x8xf32>) -> (tensor<8x4xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:4x8xf32>, %ret: !flow.dispatch.tensor<writeonly:8x4xf32>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : !flow.dispatch.tensor<readonly:4x8xf32> -> tensor<8x4xf32>
    %ret_value = "test.sink2"(%arg_value) : (tensor<8x4xf32>) -> (tensor<8x4xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : tensor<8x4xf32> -> !flow.dispatch.tensor<writeonly:8x4xf32>
    flow.return
  }
  // CHECK-NEXT: return %[[RET1]]
  return %1 : tensor<8x4xf32>
}

// -----

// CHECK: flow.executable private @dispatchFn1_dispatch_0

// CHECK-LABEL: func @dispatchFn1
func @dispatchFn1(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn1_dispatch_0::@dispatchFn1_dispatch_0
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// CHECK: flow.executable private @dispatchFn2_dispatch_0

// CHECK-LABEL: func @dispatchFn2
func @dispatchFn2(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn2_dispatch_0::@dispatchFn2_dispatch_0
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:8x4xf32>, %ret: !flow.dispatch.tensor<writeonly:4x8xf32>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dynamicShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.dispatch.entry public @dynamicShapeDispatch_dispatch_0 attributes {
// CHECK-SAME:       workgroup_rank = 2 : index}
//      CHECK: func @dynamicShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG_TENSOR:.+]]: !flow.dispatch.tensor<readonly:7x?x24x?xf32>,
// CHECK-SAME:     %[[DIM1_CAPTURE:.+]]: index, %[[DIM3_CAPTURE:.+]]: index,
// CHECK-SAME:     %[[RET_TENSOR:.+]]: !flow.dispatch.tensor<writeonly:?x?x1024xf32>) {

//      CHECK: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG_TENSOR]], {{.+}} : !flow.dispatch.tensor<readonly:7x?x24x?xf32>{%[[DIM1_CAPTURE]], %[[DIM3_CAPTURE]]}
// CHECK-NEXT: %[[RET_TILE:.+]] = "test.tile_math"(%[[ARG_TILE]])
// CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET_TENSOR]], {{.+}} -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>{%[[DIM3_CAPTURE]], %[[DIM1_CAPTURE]]}

// CHECK:   return
// CHECK-NEXT: }

// CHECK-LABEL: func @dynamicShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<7x?x24x?xf32>
func @dynamicShapeDispatch(%arg0 : tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %c1
  %dim1 = tensor.dim %arg0, %c1 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[DIM3:.+]] = tensor.dim %[[ARG0]], %c3
  %dim3 = tensor.dim %arg0, %c3 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[X:.+]] = arith.constant 1024
  %x = arith.constant 1024 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 512
  %y = arith.constant 512 : index
  // CHECK-NEXT: %[[RET0:.+]] = flow.dispatch @dynamicShapeDispatch_dispatch_0::@dynamicShapeDispatch_dispatch_0[
  // CHECK-SAME:   %[[X]], %[[Y]]
  // CHECK-SAME: ](%arg0, %[[DIM1]], %[[DIM3]])
  // CHECK-SAME: : (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]}, index, index) -> tensor<?x?x1024xf32>{%[[DIM3]], %[[DIM1]]}
  %ret0 = flow.dispatch.workgroups[%x, %y](%arg0, %dim1, %dim3) : (tensor<7x?x24x?xf32>{%dim1, %dim3}, index, index) -> tensor<?x?x1024xf32>{%dim3, %dim1} = (
    %arg: !flow.dispatch.tensor<readonly:7x?x24x?xf32>,
    %dim1_capture: index, %dim3_capture: index,
    %ret: !flow.dispatch.tensor<writeonly:?x?x1024xf32>
  ) {
    %arg_tile = flow.dispatch.tensor.load %arg, offsets=[0, 0, 0, 0], sizes=[7, %dim1_capture, 24, %dim3_capture], strides=[1, 1, 1, 1] : !flow.dispatch.tensor<readonly:7x?x24x?xf32>{%dim1_capture, %dim3_capture} -> tensor<7x?x24x?xf32>
    %ret_tile = "test.tile_math"(%arg_tile) : (tensor<7x?x24x?xf32>) -> (tensor<?x?x1024xf32>)
    flow.dispatch.tensor.store %ret_tile, %ret, offsets=[0, 0, 0], sizes=[%dim3_capture, %dim1_capture, 1024], strides=[1, 1, 1] : tensor<?x?x1024xf32> -> !flow.dispatch.tensor<writeonly:?x?x1024xf32>{%dim3_capture, %dim1_capture}
    flow.return
  }
  // CHECK-NEXT: return %[[RET0]]
  return %ret0 : tensor<?x?x1024xf32>
}
