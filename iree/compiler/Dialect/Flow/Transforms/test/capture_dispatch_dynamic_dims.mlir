// RUN: iree-opt -iree-flow-capture-dispatch-dynamic-dims %s | FileCheck %s

// Tests that both operands and results get any dims captured that aren't
// already captured.

// CHECK-LABEL: @captureDims
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG0_DIM0:.+]]: index, %[[ARG0_DIM1:.+]]: index, %[[RET0_DIM0:.+]]: index, %[[RET0_DIM1:.+]]: index)
func.func @captureDims(%arg0: tensor<?x?xf32>, %arg0_dim0: index, %arg0_dim1: index, %ret0_dim0: index, %ret0_dim1: index) {
  %c1 = arith.constant 1 : index
  // CHECK: flow.dispatch.workgroups[%c1, %c1, %c1](%[[ARG0]], %[[ARG0_DIM0]], %[[RET0_DIM0]], %[[ARG0_DIM1]], %[[RET0_DIM1]])
  %0 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg0_dim0, %ret0_dim0) : (tensor<?x?xf32>{%arg0_dim0, %arg0_dim1}, index, index) -> tensor<?x?xf32>{%ret0_dim0, %ret0_dim1} =
      // CHECK-NEXT: (%[[ARG0_CAPTURE:.+]]: !flow.dispatch.tensor<readonly:?x?xf32>, %[[ARG0_DIM0_CAPTURE:.+]]: index, %[[RET0_DIM0_CAPTURE:.+]]: index, %[[ARG0_DIM1_CAPTURE:.+]]: index, %[[RET0_DIM1_CAPTURE:.+]]: index, %[[RET0_CAPTURE:.+]]: !flow.dispatch.tensor<writeonly:?x?xf32>)
      (%arg0_capture: !flow.dispatch.tensor<readonly:?x?xf32>, %arg0_dim0_capture: index, %ret0_dim0_capture: index, %ret0_capture: !flow.dispatch.tensor<writeonly:?x?xf32>) {
    // CHECK-DAG: = flow.dispatch.tie_shape %[[ARG0_CAPTURE]] : !flow.dispatch.tensor<readonly:?x?xf32>{%[[ARG0_DIM0_CAPTURE]], %[[ARG0_DIM1_CAPTURE]]}
    // CHECK-DAG: = flow.dispatch.tie_shape %[[RET0_CAPTURE]] : !flow.dispatch.tensor<writeonly:?x?xf32>{%[[RET0_DIM0_CAPTURE]], %[[RET0_DIM1_CAPTURE]]}
    flow.return
  }
  return
}

// -----

// Tests dim capture on tied operands.

// CHECK-LABEL: @capturedTiedDims
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG0_DIM0:.+]]: index, %[[ARG0_DIM1:.+]]: index)
func.func @capturedTiedDims(%arg0: tensor<?x?xf32>, %arg0_dim0: index, %arg0_dim1: index) {
  %c1 = arith.constant 1 : index
  // CHECK: flow.dispatch.workgroups[%c1, %c1, %c1](%[[ARG0]], %[[ARG0_DIM0]], %[[ARG0_DIM1]])
  %0 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg0_dim0) : (tensor<?x?xf32>{%arg0_dim0, %arg0_dim1}, index) -> %arg0{%arg0_dim0, %arg0_dim1} =
      // CHECK-NEXT: (%[[ARG0_CAPTURE:.+]]: !flow.dispatch.tensor<readwrite:?x?xf32>, %[[ARG0_DIM0_CAPTURE:.+]]: index, %[[ARG0_DIM1_CAPTURE:.+]]: index)
      (%arg0_capture: !flow.dispatch.tensor<readwrite:?x?xf32>, %arg0_dim0_capture: index) {
    // CHECK-DAG: = flow.dispatch.tie_shape %[[ARG0_CAPTURE]] : !flow.dispatch.tensor<readwrite:?x?xf32>{%[[ARG0_DIM0_CAPTURE]], %[[ARG0_DIM1_CAPTURE]]}
    flow.return
  }
  return
}
