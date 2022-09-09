// TODO(#9510): Enable the test.
// RUN: iree-dialects-opt --transform-dialect-interpreter %s | FileCheck %s

func.func @fun_to_benchmark(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) ->
    tensor<128x128xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  // With scalarization we expect vectorization to still work albeit with a leading
  // `1` dimension.
  // CHECK: vector.contract {{.*}} : vector<1x32xf32>, vector<32x16xf32> into vector<1x16xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                    outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @isa_linalg.matmul : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "linalg.matmul"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
    rewrite %2 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @isa_linalg.matmul in %arg1
    %tiled_linalg_op, %loops:3 = transform.structured.tile %0 [6, 16, 32] {interchange = [1, 0, 2]}
    %1 = transform.loop.peel %loops#0

    %tiled_and_peeled_linalg_op = pdl_match @isa_linalg.matmul in %1
    // This test checks the proper handling of the scalarize dims attribute.
    // The first dimension does not divide but we can always scalarize a `?` into `1`
    // and enable vectorization of a lower-rank op this way.
    %tiled_and_peeled_linalg_op_0 = transform.structured.scalarize %tiled_and_peeled_linalg_op
    %parent = transform.get_closest_isolated_parent %tiled_and_peeled_linalg_op_0
    transform.structured.vectorize %parent {vectorize_padding = false}
  }
}
