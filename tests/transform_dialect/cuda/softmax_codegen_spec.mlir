// RUN: iree-opt %s

// Codegen
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    // First level of tiling + fusion parallelizes to blocks.
    // The mapping  to block ids can only happen after bufferization atm
    %root = transform.structured.match interface{LinalgOp}
      attributes{iterator_types = ["parallel", "parallel", "parallel"]} in %variant_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    %red = transform.structured.match interface{LinalgOp}
      attributes{iterator_types = ["parallel", "parallel", "reduction"]} in %variant_op
    %not_root = merge_handles %fill, %red
    %foreach_thread, %tiled_generic =
      transform.structured.tile_to_foreach_thread_op %root tile_sizes [1, 4]
    transform.structured.fuse_into_containing_op %not_root into %foreach_thread

    // Second level of tiling + fusion parallelizes to threads.
    // Leaving the reduction untiled on threadIdx.x makes it sequential on
    // threadIdx.x. After distribution, predication by if (threadIdx.x == 0) is
    // introduced and opportunities for distributing vector ops across warps
    // appear.
    %fill_linalg = transform.structured.match ops{["linalg.fill"]} in %variant_op
    %reduction_linalg = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = ["parallel", "parallel", "reduction"]} in %variant_op
    %parallel_linalg = transform.structured.match ops{["linalg.generic"]}
      attributes{iterator_types = ["parallel", "parallel", "parallel"]} in %variant_op
    %foreach_thread_reduction, %tiled_reduction_generic =
      transform.structured.tile_to_foreach_thread_op %reduction_linalg tile_sizes [1, 1]
        (mapped to dims [2, 1, 0])
    // TODO: this fusion currently does not happen properly, this is related to the clone
    // behavior when fusing into scf.foreach_thread.
    // Once fixed we'll be able to fuse.
    // Fusion will save us one roundtrip to memory.
    // transform.structured.fuse_into_containing_op %fill_linalg into %foreach_thread_reduction
    transform.structured.tile_to_foreach_thread_op %parallel_linalg num_threads [1, 4, 32]
        (mapped to dims [2, 1, 0])


    // Inability to tile reductions to scf.foreach_thread has 2 implications:
    //   1. since no scf.foreach_thread is present, no gpu.barrier is added.
    //      This should be fixed independently: ops that are not nested in an scf.foreach_thread
    //      should have a gpu.barrier. Later needs to be complemented by a barrier
    //      removal pass.
    //   2. Similarly, needs to be predicated under an if threadIx == 0 to avoid
    //      multiple threads updating the buffer inplace once bufferized.
    //
    // Instead, we can vectorize and go to vector SSA values that sidestep these
    // issues.
    // Everyone will race to the write while still computing the same value.
    //
    // That is still not good enough because we need to predicate this in order
    // to enable the parallel reduction on warps.
    %func = transform.structured.match ops{["func.func"]} in %variant_op
    %func_2 = transform.structured.vectorize %func

    // Bufferization is necessary for:
    //   1. lowering scf.foreach_thread to workgroup (block level parallelism)
    //   2. lowering scf.foreach_thread to gpu (thread level parallelism)
    //   3. introducing predication (due to 1. + 2.) which enables rewriting to
    //      warp_execute_on_lane_0 and later vector distribution.
    %variant_op_2 = transform.iree.bufferize { target_gpu } %variant_op

    %func_3 = transform.structured.match ops{["func.func"]} in %variant_op_2
    %func_4 = transform.iree.foreach_thread_to_workgroup %func_3
    transform.iree.foreach_thread_to_gpu_and_translation_info %func_4
      { workgroup_size = [32, 4, 1] }

    %end_func = transform.structured.match ops{["func.func"]} in %variant_op_2
    %end_func_2 = transform.iree.apply_patterns %end_func { rank_reducing }

    // Vector distribution needs to happen on buffers.
    %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_2
    %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 }
    transform.iree.vector.warp_distribute %end_func_2
  }
}


