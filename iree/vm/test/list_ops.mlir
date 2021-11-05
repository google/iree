vm.module @list_ops {

// time cmake --build . -t bytecode_module_test
// ./iree/vm/bytecode_module_test
// iree/tools/iree-translate -split-input-file -iree-vm-ir-to-bytecode-module -iree-vm-bytecode-module-output-format=flatbuffer-text /home/cycheng/iree/iree/vm/test/list_ops.mlir -print-ir-after-all
// iree/tools/iree-translate -iree-vm-ir-to-c-module -o list_ops.h /home/cycheng/iree/iree/vm/test/list_ops.mlir

  vm.export @test_swap_lists
  vm.func @test_swap_lists() {
    %c0 = vm.const.i32 0 : i32
    %c1 = vm.const.i32 1 : i32
    %c2 = vm.const.i32 2 : i32
    %c3 = vm.const.i32 3 : i32
    %c4 = vm.const.i32 4 : i32
    %c27 = vm.const.i32 27 : i32
    %c42 = vm.const.i32 42 : i32

    // These allocs shouldn't be CSE'd.
    %list0 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    %list1 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    vm.list.resize %list0, %c1 : (!vm.list<i8>, i32)
    vm.list.resize %list1, %c4 : (!vm.list<i8>, i32)
    vm.list.set.i32 %list0, %c0, %c27 : (!vm.list<i8>, i32, i32)
    vm.list.set.i32 %list1, %c0, %c42 : (!vm.list<i8>, i32, i32)

    vm.list.swap %list0, %list1 : (!vm.list<i8>, !vm.list<i8>)

    %res0 = vm.list.get.i32 %list0, %c0 : (!vm.list<i8>, i32) -> i32
    %res1 = vm.list.get.i32 %list1, %c0 : (!vm.list<i8>, i32) -> i32
    vm.check.eq %res0, %c42, "list0.get(0)=42" : i32
    vm.check.eq %res1, %c27, "list1.get(0)=27" : i32

    // list0 = 42, 27
    vm.list.set.i32 %list0, %c1, %c27 : (!vm.list<i8>, i32, i32)
    // overlapped copy.
    // list0 = 42, 42, 42, 42
    vm.list.copy %list0, %c0, %list0, %c1, %c3 : (!vm.list<i8>, i32, !vm.list<i8>, i32, i32)

    %res0_2 = vm.list.get.i32 %list0, %c2 : (!vm.list<i8>, i32) -> i32
    %res0_3 = vm.list.get.i32 %list0, %c3 : (!vm.list<i8>, i32) -> i32
    vm.check.eq %res0_2, %c42, "list0.get(2)=42" : i32
    vm.check.eq %res0_3, %c42, "list1.get(3)=42" : i32

    //vm.list.copy %list0, %c0, %list1, %c0, %c1 : (!vm.list<i8>, i32, !vm.list<i8>, i32, i32)

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I8 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i8
  vm.func @test_i8() {
    %c42 = vm.const.i32 42 : i32
    %c100 = vm.const.i32 100 : i32
    %c0 = vm.const.i32 0 : i32
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i8>
    vm.list.reserve %list, %c100 : (!vm.list<i8>, i32)
    %sz = vm.list.size %list : (!vm.list<i8>) -> i32
    %sz_dno = util.do_not_optimize(%sz) : i32
    vm.check.eq %sz_dno, %c0, "list<i8>.empty.size()=0" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I16 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i16
  vm.func @test_i16() {
    %c0 = vm.const.i32 0 : i32
    %c1 = vm.const.i32 1 : i32
    %c27 = vm.const.i32 27 : i32
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i16>
    vm.list.resize %list, %c1 : (!vm.list<i16>, i32)
    vm.list.set.i32 %list, %c0, %c27 : (!vm.list<i16>, i32, i32)
    %v = vm.list.get.i32 %list, %c0 : (!vm.list<i16>, i32) -> i32
    vm.check.eq %v, %c27, "list<i16>.empty.set(0, 27).get(0)=27" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with I32 types
  //===--------------------------------------------------------------------===//

  vm.export @test_i32
  vm.func @test_i32() {
    %c42 = vm.const.i32 42 : i32
    %list = vm.list.alloc %c42 : (i32) -> !vm.list<i32>
    %sz = vm.list.size %list : (!vm.list<i32>) -> i32
    %c100 = vm.const.i32 100 : i32
    %c101 = vm.const.i32 101 : i32
    vm.list.resize %list, %c101 : (!vm.list<i32>, i32)
    vm.list.set.i32 %list, %c100, %c42 : (!vm.list<i32>, i32, i32)
    %v = vm.list.get.i32 %list, %c100 : (!vm.list<i32>, i32) -> i32
    vm.check.eq %v, %c42, "list<i32>.empty.set(100, 42).get(100)=42" : i32
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // vm.list.* with ref types
  //===--------------------------------------------------------------------===//

  vm.export @test_ref
  vm.func @test_ref() {
    // TODO(benvanik): test vm.list with ref types.
    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Multiple lists within the same block
  //===--------------------------------------------------------------------===//

  vm.export @test_multiple_lists
  vm.func @test_multiple_lists() {
    %c0 = vm.const.i32 0 : i32
    %c1 = vm.const.i32 1 : i32
    %c27 = vm.const.i32 27 : i32
    %c42 = vm.const.i32 42 : i32

    // These allocs shouldn't be CSE'd.
    %list0 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    %list1 = vm.list.alloc %c1 : (i32) -> !vm.list<i8>
    vm.list.resize %list0, %c1 : (!vm.list<i8>, i32)
    vm.list.resize %list1, %c1 : (!vm.list<i8>, i32)
    vm.list.set.i32 %list0, %c0, %c27 : (!vm.list<i8>, i32, i32)
    vm.list.set.i32 %list1, %c0, %c42 : (!vm.list<i8>, i32, i32)
    %res0 = vm.list.get.i32 %list0, %c0 : (!vm.list<i8>, i32) -> i32
    %res1 = vm.list.get.i32 %list1, %c0 : (!vm.list<i8>, i32) -> i32
    vm.check.eq %res0, %c27, "list0.get(0)=27" : i32
    vm.check.eq %res1, %c42, "list1.get(0)=42" : i32

    vm.return
  }

  //===--------------------------------------------------------------------===//
  // Failure tests
  //===--------------------------------------------------------------------===//

  vm.export @fail_uninitialized_access
  vm.func @fail_uninitialized_access() {
    %c0 = vm.const.i32 0 : i32
    %c1 = vm.const.i32 1 : i32
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i32>
    vm.list.set.i32 %list, %c0, %c1 : (!vm.list<i32>, i32, i32)
    vm.return
  }

  vm.export @fail_out_of_bounds_read
  vm.func @fail_out_of_bounds_read() {
    %c1 = vm.const.i32 1 : i32
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i32>
    vm.list.resize %list, %c1 : (!vm.list<i32>, i32)
    %v = vm.list.get.i32 %list, %c1 : (!vm.list<i32>, i32) -> i32
    %v_dno = util.do_not_optimize(%v) : i32
    vm.return
  }

  vm.export @fail_out_of_bounds_write
  vm.func @fail_out_of_bounds_write() {
    %c1 = vm.const.i32 1 : i32
    %list = vm.list.alloc %c1 : (i32) -> !vm.list<i32>
    vm.list.resize %list, %c1 : (!vm.list<i32>, i32)
    vm.list.set.i32 %list, %c1, %c1 : (!vm.list<i32>, i32, i32)
    vm.return
  }
}
