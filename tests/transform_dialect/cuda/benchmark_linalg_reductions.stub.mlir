// iree-transform-compile   /usr/local/google/home/ntv/github/iree/tests/transform_dialect/cuda/warp_reduction_dispatch.mlir -b cuda -- --iree-hal-benchmark-dispatch-repeat-count=5 | \
// /usr/local/cuda-11.4/bin/nvprof  --print-gpu-trace  iree-run-module --entry_function=warp_reduction_dispatch --device=cuda --function_input="512x10240xf32=1"


!in_tensor_reduction_2d_static_t = tensor<${SZ1}x${SZ2}xf32>
!out_tensor_reduction_2d_static_t = tensor<${SZ1}xf32>

func.func @reduction_2d_static(%arg : !in_tensor_reduction_2d_static_t) -> (!out_tensor_reduction_2d_static_t) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -0.000000e+00 : f32
  
  %d0 = tensor.dim %arg, %c0 : !in_tensor_reduction_2d_static_t
  %0 = tensor.empty() : !out_tensor_reduction_2d_static_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_reduction_2d_static_t) ->   !out_tensor_reduction_2d_static_t
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_reduction_2d_static_t) outs(%1 : !out_tensor_reduction_2d_static_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> !out_tensor_reduction_2d_static_t
  return %2 : !out_tensor_reduction_2d_static_t
}

func.func @reduction_2d_elementwise_static(%arg : !in_tensor_reduction_2d_static_t) -> (!in_tensor_reduction_2d_static_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant -0.000000e+00 : f32
  
  %d0 = tensor.dim %arg, %c0 : !in_tensor_reduction_2d_static_t
  %d1 = tensor.dim %arg, %c1 : !in_tensor_reduction_2d_static_t
  %0 = tensor.empty() : !out_tensor_reduction_2d_static_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_reduction_2d_static_t) ->   !out_tensor_reduction_2d_static_t
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_reduction_2d_static_t) outs(%1 : !out_tensor_reduction_2d_static_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> !out_tensor_reduction_2d_static_t

  %cst_0 = arith.constant 3.840000e+02 : f32
  %i = tensor.empty() : !in_tensor_reduction_2d_static_t
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : !out_tensor_reduction_2d_static_t) outs(%i : !in_tensor_reduction_2d_static_t) {
      ^bb0(%arg0: f32, %arg1: f32):
        %12 = arith.divf %arg0, %cst_0 : f32
        linalg.yield %12 : f32
      } -> !in_tensor_reduction_2d_static_t

  return %3 : !in_tensor_reduction_2d_static_t
}

!in_tensor_reduction_2d_dynamic_t = tensor<?x?xf32>
!out_tensor_reduction_2d_dynamic_t = tensor<?xf32>

func.func @reduction_2d_dynamic(%arg : !in_tensor_reduction_2d_dynamic_t) -> (!out_tensor_reduction_2d_dynamic_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %cst = arith.constant -0.000000e+00 : f32
  
  %d0 = tensor.dim %arg, %c0 : !in_tensor_reduction_2d_dynamic_t
  %d1 = tensor.dim %arg, %c1 : !in_tensor_reduction_2d_dynamic_t
  %0 = tensor.empty(%d0) : !out_tensor_reduction_2d_dynamic_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_reduction_2d_dynamic_t) ->   !out_tensor_reduction_2d_dynamic_t
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_reduction_2d_dynamic_t) outs(%1 : !out_tensor_reduction_2d_dynamic_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> !out_tensor_reduction_2d_dynamic_t
  return %2 : !out_tensor_reduction_2d_dynamic_t
}

func.func @reduction_2d_elementwise_dynamic(%arg : !in_tensor_reduction_2d_dynamic_t) -> (!in_tensor_reduction_2d_dynamic_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant -0.000000e+00 : f32
  
  %d0 = tensor.dim %arg, %c0 : !in_tensor_reduction_2d_dynamic_t
  %d1 = tensor.dim %arg, %c1 : !in_tensor_reduction_2d_dynamic_t
  %0 = tensor.empty(%d0) : !out_tensor_reduction_2d_dynamic_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_reduction_2d_dynamic_t) ->   !out_tensor_reduction_2d_dynamic_t
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_reduction_2d_dynamic_t) outs(%1 : !out_tensor_reduction_2d_dynamic_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> !out_tensor_reduction_2d_dynamic_t

  %cst_0 = arith.constant 3.840000e+02 : f32
  %i = tensor.empty(%d0, %d1) : !in_tensor_reduction_2d_dynamic_t
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%2 : !out_tensor_reduction_2d_dynamic_t) outs(%i : !in_tensor_reduction_2d_dynamic_t) {
      ^bb0(%arg0: f32, %arg1: f32):
        %12 = arith.divf %arg0, %cst_0 : f32
        linalg.yield %12 : f32
      } -> !in_tensor_reduction_2d_dynamic_t

  return %3 : !in_tensor_reduction_2d_dynamic_t
}
