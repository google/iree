// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" \
// RUN:   --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy \
// RUN: | FileCheck %s

// Check that setting the command line options affect the transform
// strategy as expected.
// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target{test-lowering-configuration})))" \
// RUN:   --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy \
// RUN:   --td-pad-strategy-blk-sizes=32,32,1 \
// RUN:   --td-pad-strategy-num-threads=8,4,1 \
// RUN:   --td-pad-strategy-vector-size=2,2 \
// RUN:   --td-pad-strategy-use-async-copies=false \
// RUN: | FileCheck --check-prefix=WITH_OPTIONS %s

hal.executable @pad {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @pad ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @pad() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<123x456xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [123, 456], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<123x456xf32>> -> tensor<123x456xf32>

      %pad = arith.constant 0.0 : f32
      %padded = tensor.pad %3 low[0, 0] high[5, 56] {
        ^bb0(%arg1: index, %arg2: index):
          tensor.yield %pad : f32
      } : tensor<123x456xf32> to tensor<128x512xf32>

      flow.dispatch.tensor.store %padded, %2, offsets = [0, 0], sizes = [128, 512], strides = [1, 1] : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: func @pad
//       CHECK:   transform.sequence  failures(propagate) {
//       CHECK:   transform.iree.register_match_callbacks
//       CHECK:   {{.*}} = transform.iree.match_callback failures(propagate) "pad"({{.*}}) : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.structured.tile_to_forall_op {{.*}}   num_threads [] tile_sizes [64, 64](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//       CHECK:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.match ops{["scf.if"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.scf.take_assumed_branch {{.*}} take_else_branch : (!transform.any_op) -> ()
//       CHECK:   transform.iree.populate_workgroup_count_region_using_num_threads_slice {{.*}} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.tile_to_forall_op {{.*}}   num_threads [16, 16] tile_sizes [](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//       CHECK:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.match ops{["scf.if"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.scf.take_assumed_branch {{.*}} take_else_branch : (!transform.any_op) -> ()
//       CHECK:   transform.structured.masked_vectorize {{.*}} vector_sizes [4, 4] : !transform.any_op
//       CHECK:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:     transform.apply_patterns.vector.lower_masked_transfers
//       CHECK:   transform.iree.apply_patterns {{.*}} {rank_reducing_linalg, rank_reducing_vector} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.vectorize {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
//       CHECK:   transform.iree.eliminate_empty_tensors {{.*}} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.iree.bufferize {target_gpu} {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.iree.erase_hal_descriptor_type_from_memref {{.*}} : (!transform.any_op) -> ()
//       CHECK:   transform.iree.apply_buffer_optimizations {{.*}} : (!transform.any_op) -> ()
//       CHECK:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       CHECK:   transform.iree.forall_to_workgroup {{.*}} : (!transform.any_op) -> ()
//       CHECK:   transform.iree.map_nested_forall_to_gpu_threads {{.*}} workgroup_dims = [64, 64, 1] warp_dims = [] : (!transform.any_op) -> ()
//       CHECK:     transform.apply_patterns.vector.lower_masks
//       CHECK:     transform.apply_patterns.vector.materialize_masks
//       CHECK:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization} : (!transform.any_op) -> ()

// WITH_OPTIONS-LABEL: func @pad
//       WITH_OPTIONS:   transform.sequence  failures(propagate) {
//       WITH_OPTIONS:   transform.iree.register_match_callbacks
//       WITH_OPTIONS:   {{.*}} = transform.iree.match_callback failures(propagate) "pad"({{.*}}) : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   transform.structured.tile_to_forall_op {{.*}}   num_threads [] tile_sizes [32, 32](mapping = [#gpu.block<y>, #gpu.block<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//       WITH_OPTIONS:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   {{.*}} = transform.structured.match ops{["scf.if"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   transform.scf.take_assumed_branch {{.*}} take_else_branch : (!transform.any_op) -> ()
//       WITH_OPTIONS:   transform.iree.populate_workgroup_count_region_using_num_threads_slice {{.*}} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   {{.*}} = transform.structured.tile_to_forall_op {{.*}}   num_threads [8, 4] tile_sizes [](mapping = [#gpu.thread<y>, #gpu.thread<x>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//       WITH_OPTIONS:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   {{.*}} = transform.structured.match ops{["scf.if"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   transform.scf.take_assumed_branch {{.*}} take_else_branch : (!transform.any_op) -> ()
//       WITH_OPTIONS:   transform.structured.masked_vectorize {{.*}} vector_sizes [2, 2] : !transform.any_op
//       WITH_OPTIONS:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:     transform.apply_patterns.vector.lower_masked_transfers
//       WITH_OPTIONS:   transform.iree.apply_patterns {{.*}} {rank_reducing_linalg, rank_reducing_vector} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   {{.*}} = transform.structured.vectorize {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   transform.iree.eliminate_empty_tensors {{.*}} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   {{.*}} = transform.iree.bufferize {target_gpu} {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   transform.iree.erase_hal_descriptor_type_from_memref {{.*}} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   transform.iree.apply_buffer_optimizations {{.*}} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   {{.*}} = transform.structured.match ops{["func.func"]} in {{.*}} : (!transform.any_op) -> !transform.any_op
//       WITH_OPTIONS:   transform.iree.forall_to_workgroup {{.*}} : (!transform.any_op) -> ()
//       WITH_OPTIONS:   transform.iree.map_nested_forall_to_gpu_threads {{.*}} workgroup_dims = [32, 32, 1] warp_dims = [] : (!transform.any_op) -> ()
//       WITH_OPTIONS:     transform.apply_patterns.vector.lower_masks
//       WITH_OPTIONS:     transform.apply_patterns.vector.materialize_masks
//       WITH_OPTIONS:   transform.iree.apply_patterns {{.*}} {canonicalization, cse, fold_memref_aliases, licm, tiling_canonicalization} : (!transform.any_op) -> ()
