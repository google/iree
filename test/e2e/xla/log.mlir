// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --output_types=f | IreeFileCheck %s --check-prefixes=CHECK,INTERP
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv --output_types=f --skip_tests=double %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %input = constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "xla_hlo.log"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=0 0.693147 1.09861 1.38629

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input = constant dense<4.0> : tensor<f32>
  %result = "xla_hlo.log"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=1.38629

// -----

// INTERP-LABEL: EXEC @double
func @double() -> tensor<f64> {
  %input = constant dense<4.0> : tensor<f64>
  %result = "xla_hlo.log"(%input) : (tensor<f64>) -> tensor<f64>
  return %result : tensor<f64>
}
// INTERP: f64=1.38629
