// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-spirv-lower-executable-target-pass{test-lowering-configuration=true})))' --verify-diagnostics --split-input-file %s

#config = #iree_codegen.lowering_config<tile_sizes = []>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected 3 levels of tiling sizes, got 0}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = []
    }
    // expected-error @+1 {{expected workgroup size to have three dimensions for SPIR-V pipelines}}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x128xf32>
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x128xf32>)
          outs(%result: memref<64x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 8 : index, 128 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x128xf32>
        // expected-error @+1 {{expected workgroup size dimensions not exceeding [128, 128, 64]}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x128xf32>)
          outs(%result: memref<64x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 2], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [32 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x128xf32>
        // expected-error @+1 {{expected total invocation count in workgroup to be <= 128}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x128xf32>)
          outs(%result: memref<64x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [16, 8], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [8 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x128xf32>
        // expected-error @+1 {{expected total workgroup size to be multiple of 32}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x128xf32>)
          outs(%result: memref<64x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 32768,
        max_compute_workgroup_invocations = 512,
        max_compute_workgroup_size = [512, 512, 512],
        subgroup_size = 16>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [32 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x128xf32>
        // expected-error @+1 {{expected workgroup tile sizes to be the product of thread tile sizes and workgroup sizes}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x128xf32>)
          outs(%result: memref<64x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 60], [4, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [15 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x128xf32>
        // expected-error @+1 {{expected each workgroup size dimension to be power of two}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x128xf32>)
          outs(%result: memref<64x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<48x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x128xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<48x128xf32>
        // expected-error @+1 {{LHS shape is indivisible by first level tile size}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<48x16xf32>, memref<16x128xf32>)
          outs(%result: memref<48x128xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[32, 64], [4, 4], [0, 0, 4]]>
#translation = #iree_codegen.translation_info<SPIRVMatmulPromoteVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      translation_info = #translation,
      workgroup_size = [16 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<64x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x80xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<64x80xf32>
        // expected-error @+1 {{RHS shape is indivisible by first level tile size}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<64x16xf32>, memref<16x80xf32>)
          outs(%result: memref<64x80xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 2, 2, 2], [0, 0, 0, 0, 1, 1, 4]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_2d_nhwc_hwcf {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      workgroup_size = [8: index, 2: index, 2: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @illegal() {
        %c112 = arith.constant 112 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c16 step %8 {
              %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
              %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
              %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
              %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              // expected-error @+1 {{expected 4 levels of tiling sizes, got 3}}
              %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>}
                      ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>)
                      outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
            }
          }
        }
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 6, 6, 16], [0, 3, 3, 2], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_2d_nhwc_hwcf {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      workgroup_size = [8: index, 2: index, 2: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @illegal() {
        %c112 = arith.constant 112 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c16 step %8 {
              %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
              %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
              %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
              %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              // expected-error @+1 {{expected first level tile size divides the output size [OH, OW, OC]}}
              %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>}
                      ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>)
                      outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
            }
          }
        }
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 16], [0, 2, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_2d_nhwc_hwcf {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
      workgroup_size = [8: index, 2: index, 2: index],
      translation_info = #translation
    }
    builtin.module  {
      func.func @illegal() {
        %c112 = arith.constant 112 : index
        %c16 = arith.constant 16 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_z]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_z]
        scf.for %arg0 = %3 to %c112 step %4 {
          %5 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_id_y]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%workgroup_count_y]
          scf.for %arg1 = %5 to %c112 step %6 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_id_x]
            %8 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%workgroup_count_x]
            scf.for %arg2 = %7 to %c16 step %8 {
              %9 = flow.dispatch.tensor.load %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>> -> tensor<1x4x4x16xf32>
              %10 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
              %11 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
              %12 = flow.dispatch.tensor.load %0, offsets = [0, %10, %11, 0], sizes = [1, 9, 9, 8], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x225x225x8xf32>> -> tensor<1x9x9x8xf32>
              %13 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, %arg2], sizes = [3, 3, 8, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x8x16xf32>> -> tensor<3x3x8x16xf32>
              %14 = linalg.fill ins(%cst : f32) outs(%9 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              // expected-error @+1 {{expected workgroup tile sizes to be the product of thread tile size and workgroup size}}
              %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #config, strides = dense<2> : tensor<2xi64>}
                      ins(%12, %13 : tensor<1x9x9x8xf32>, tensor<3x3x8x16xf32>)
                      outs(%14 : tensor<1x4x4x16xf32>) -> tensor<1x4x4x16xf32>
              flow.dispatch.tensor.store %15, %2, offsets = [0, %arg0, %arg1, %arg2], sizes = [1, 4, 4, 16], strides = [1, 1, 1, 1] : tensor<1x4x4x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x16xf32>>
            }
          }
        }
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64], [0, 1, 7, 2], [0, 0, 0, 0, 5, 5], [0, 1, 0, 0]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @depthwise_conv_2d_nhwc_hwc {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
    workgroup_size = [32: index, 1: index, 1: index],
    translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x11x11x576xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5x5x576xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<1x7x7x576xf32>
        // expected-error @+1 {{expected tile sizes for KH and KW to be 1}}
        linalg.depthwise_conv_2d_nhwc_hwc {lowering_config = #config}
          ins(%lhs, %rhs : memref<1x11x11x576xf32>, memref<5x5x576xf32>)
          outs(%result: memref<1x7x7x576xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64], [0, 1, 7, 2], [0, 0, 0, 0, 1, 1], [0, 0, 1, 1]]>
#translation = #iree_codegen.translation_info<SPIRVBaseVectorize>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @depthwise_conv_2d_nhwc_hwc {
  hal.executable.variant public @vulkan_spirv_fb, target = <"vulkan", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader], []>, Unknown:IntegratedGPU, #spirv.resource_limits<
        max_compute_shared_memory_size = 16384,
        max_compute_workgroup_invocations = 128,
        max_compute_workgroup_size = [128, 128, 64],
        subgroup_size = 32>>
    }> {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {
    workgroup_size = [32: index, 1: index, 1: index],
    translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x11x11x576xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5x5x576xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<1x7x7x576xf32>
        // expected-error @+1 {{expected the fourth level of tile size to be [0, 1, 0, 0]}}
        linalg.depthwise_conv_2d_nhwc_hwc {lowering_config = #config}
          ins(%lhs, %rhs : memref<1x11x11x576xf32>, memref<5x5x576xf32>)
          outs(%result: memref<1x7x7x576xf32>)
        return
      }
    }
  }
}