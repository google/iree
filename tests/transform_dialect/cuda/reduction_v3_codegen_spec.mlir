// RUN: iree-opt %s

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  %reduction = transform.structured.match ops{["linalg.generic"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ===========================================================================
  %forall_grid, %grid_reduction =
    transform.structured.tile_to_forall_op %reduction tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

  transform.structured.fuse_into_containing_op %fill into %forall_grid : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Canonicalizations.
  transform.apply_patterns to %variant_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_licm %variant_op : !transform.any_op
  transform.iree.apply_cse %variant_op : !transform.any_op

  // Step 2. Split the reduction to get meatier parallelism.
  // This also parallelizes to threads.
  // ===========================================================================
  %forall, %block_more_parallel_fill_op_2, %block_more_parallel_op_2, %block_combiner_op_2 =
     transform.structured.tile_reduction_using_forall %grid_reduction
        by num_threads = [0, 1024], tile_sizes = [0, 1], mapping = [#gpu.thread<x>]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Fuse the fill and pointwise to privatize them.
  transform.structured.fuse_into_containing_op %block_more_parallel_fill_op_2
    into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // block_combiner_op_2 op is [parallel, reduction] of 1x384 that cannot fuse.
  // map the 1-dim to threadIdx.y to trigger mapping of the reduction to
  // threadIdx.x via predication via `if (x==0)`.
  transform.structured.tile_to_forall_op %block_combiner_op_2 num_threads [1]
    ( mapping = [#gpu.thread<y>] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Canonicalizations.
  transform.apply_patterns to %variant_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_licm %variant_op : !transform.any_op
  transform.iree.apply_cse %variant_op : !transform.any_op

  // Step 3. Rank-reduce and vectorize.
  // ===========================================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  // TODO: masked vectorization on block_more_parallel_op_2 if we want
  // vector<4> to work as intended.
  transform.apply_patterns to %func {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  %func_3 = transform.structured.vectorize %func : (!transform.any_op) -> !transform.any_op

  // Canonicalizations is necessary to get rid of some tensor.cast that block
  // hoisting.
  transform.apply_patterns to %variant_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_licm %variant_op : !transform.any_op
  transform.iree.apply_cse %variant_op : !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %func_3
    : (!transform.any_op) -> ()


  // Step 4. Bufferize and drop HAL descriptor from memref ops.
  // ===========================================================================
  // Canonicalizations required before bufferization to avoid unnecessary allocs.
  transform.apply_patterns to %variant_op {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_licm %variant_op : !transform.any_op
  transform.iree.apply_cse %variant_op : !transform.any_op
  transform.apply_patterns to %func_3 {
    transform.apply_patterns.tensor.reassociative_reshape_folding
  } : !transform.any_op 
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %func_6 = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func_6 {
    transform.apply_patterns.linalg.erase_unnecessary_inputs
  } : !transform.any_op
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
    : (!transform.any_op) -> !transform.any_op
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3
    : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!transform.any_op) -> ()

  // Step 5. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3
    : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_m : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
      workgroup_dims = [1024, 1, 1] : (!transform.any_op) -> ()

  // Step 6. Post-bufferization vector distribution with rank-reduction.
  // ===========================================================================
  transform.apply_patterns to %func_m {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.memref.fold_memref_alias_ops
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  %if_op = transform.structured.match ops{["scf.if"]} in %variant_op_3
    : (!transform.any_op) -> !transform.any_op
  %warp = transform.iree.vector.to_warp_execute_on_lane_0 %if_op { warp_size = 32 } : (!transform.any_op) -> !transform.any_op
  transform.iree.vector.warp_distribute %func_m
    : (!transform.any_op) -> ()

  // Late canonicalizations.
  transform.apply_patterns to %variant_op_3 {
    transform.apply_patterns.iree.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_licm %variant_op_3 : !transform.any_op
  transform.iree.apply_cse %variant_op_3 : !transform.any_op
}
