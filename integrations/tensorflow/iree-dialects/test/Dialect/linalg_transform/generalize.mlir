// RUN: iree-dialects-opt --linalg-transform-interp %s | FileCheck %s


// CHECK-LABEL: func.func @generalize_unary
func.func @generalize_unary(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  // CHECK-NOT:   linalg.elemwise_unary
  //     CHECK:   linalg.generic
  %0 = linalg.elemwise_unary ins(%arg0 : tensor<?x?xf32>)
                             outs(%arg1: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.elemwise_unary"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute = @generalize_unary
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg1
    transform.structured.generalize %0
  }
}
