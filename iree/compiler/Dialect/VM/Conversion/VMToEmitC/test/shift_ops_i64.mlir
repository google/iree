// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: @shl_i64
vm.module @my_module {
  vm.func @shl_i64(%arg0 : i64, %arg1 : i32) -> i64 {
    // CHECK: %0 = emitc.call "vm_shl_i64"(%arg0, %arg1) : (i64, i32) -> i64
    %0 = vm.shl.i64 %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @shr_i64_s
vm.module @my_module {
  vm.func @shr_i64_s(%arg0 : i64, %arg1 : i32) -> i64 {
    // CHECK: %0 = emitc.call "vm_shr_i64s"(%arg0, %arg1) : (i64, i32) -> i64
    %0 = vm.shr.i64.s %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}

// -----

// CHECK-LABEL: @shr_i64_u
vm.module @my_module {
  vm.func @shr_i64_u(%arg0 : i64, %arg1 : i32) -> i64 {
    // CHECK: %0 = emitc.call "vm_shr_i64u"(%arg0, %arg1) : (i64, i32) -> i64
    %0 = vm.shr.i64.u %arg0, %arg1 : i64
    vm.return %0 : i64
  }
}
