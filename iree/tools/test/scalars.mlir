// iree-run-module
// RUN: (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=scalar --function_inputs='i32=42') | IreeFileCheck %s

// iree-benchmark-module (only checking exit codes).
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla --entry_function=scalar --function_inputs='i32=42' --module_file=${TEST_TMPDIR?}/bc.module

// iree-run-mlir
// RUN: (iree-run-mlir --iree-hal-target-backends=vmla --input-value=i32=42 %s) | IreeFileCheck %s

// CHECK-LABEL: EXEC @scalar
func @scalar(%arg0 : i32) -> i32 attributes { iree.module.export } {
  return %arg0 : i32
}
// CHECK: i32=42
