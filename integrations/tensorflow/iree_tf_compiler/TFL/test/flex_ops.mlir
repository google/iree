// RUN: iree-opt-tflite -split-input-file -iree-tflite-import-pipeline %s | IreeFileCheck %s

// This test was generated by importing a TFLite model that contained flex ops.
// The opaque data is a serialized tf-node proto and is not easily handwritten.
// CHECK-LABEL: @test_flex_ops
func @test_flex_ops(%arg0: tensor<?x2x64xf32>, %arg1: tensor<1x1x64xf32>) -> tensor<*xf32> {
  // CHECK: %[[ADD:.+]] = "tosa.add"(%arg0, %arg1) : (tensor<?x2x64xf32>, tensor<1x1x64xf32>)
  // CHECK: %[[CAST:.+]] = tensor.cast %[[ADD]]
  // CHECK: return %[[CAST]]
  %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexAddV2", custom_option = opaque<"tfl", "0x054164645632002E120541646456321A001A002A070A01541202300132180A16726573696475616C5F626C6F636B5F2E5F302F616464000237311414042801"> : tensor<63xi8>} : (tensor<?x2x64xf32>, tensor<1x1x64xf32>) -> tensor<*xf32>
  return %0: tensor<*xf32>
}
