// RUN: iree-opt -split-input-file -pass-pipeline='vm.module(iree-convert-vm-to-emitc)' %s | IreeFileCheck %s

// CHECK-LABEL: vm.func @cast
vm.module @my_module {
  vm.func @cast(%arg0 : i32) -> (i32, i32) {
    // CHECK-NEXT: %0 = emitc.call "vm_cast_si32f32"(%arg0) : (i32) -> f32
    %0 = vm.cast.si32.f32 %arg0 : i32 -> f32
    // CHECK-NEXT: %1 = emitc.call "vm_cast_ui32f32"(%arg0) : (i32) -> f32
    %1 = vm.cast.ui32.f32 %arg0 : i32 -> f32
    // CHECK-NEXT: %2 = emitc.call "vm_cast_f32si32"(%0) : (f32) -> i32
    %2 = vm.cast.f32.si32 %0 : f32 -> i32
    // CHECK-NEXT: %3 = emitc.call "vm_cast_f32ui32"(%1) : (f32) -> i32
    %3 = vm.cast.f32.ui32 %1 : f32 -> i32
    vm.return %2, %3 : i32, i32
  }
}
