// RUN: iree-dialects-opt --iree-linalg-ext-tile-and-distribute-winograd-input %s | FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1280 = arith.constant 1280 : index
  %c32 = arith.constant 32 : index
  %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
  %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<8x8x1x2x2x1280xf32>) {
    %2 = affine.min #map(%arg1)[%c1, %c1]
    %3 = scf.for %arg3 = %c0 to %c1280 step %c32 iter_args(%arg4 = %arg2) -> (tensor<8x8x1x2x2x1280xf32>) {
      %4 = affine.min #map1(%arg3)[%c32, %c1280]
      %extracted_slice = tensor.extract_slice %arg0[%arg1, 0, 0, %arg3] [%2, 10, 10, %4] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
      %extracted_slice_0 = tensor.extract_slice %0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x?x2x2x?xf32>
      %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) tensor_format("nhwc") ins(%extracted_slice : tensor<?x10x10x?xf32>) outs(%extracted_slice_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into tensor<8x8x1x2x2x1280xf32>
      scf.yield %inserted_slice : tensor<8x8x1x2x2x1280xf32>
    }
    scf.yield %3 : tensor<8x8x1x2x2x1280xf32>
  }
  return %1 : tensor<8x8x1x2x2x1280xf32>
}

// CHECK-DAG:    #map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:    #map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:    #map2 = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:    #map3 = affine_map<(d0) -> (-d0 + 10, 8)>
// CHECK:        module {
// CHECK:          func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// CHECK-SAME:       tensor<8x8x1x2x2x1280xf32> {
// CHECK:            %[[C32:.+]] = arith.constant 32 : index
// CHECK:            %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:            %[[C1:.+]] = arith.constant 1 : index
// CHECK:            %[[C0:.+]] = arith.constant 0 : index
// CHECK:            %[[CST:.+]] = arith.constant dense<
// CHECK:            %[[CST_0:.+]] = arith.constant dense<
// CHECK:            %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:            %[[C2:.+]] = arith.constant 2 : index
// CHECK:            %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// CHECK:            %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:            %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK:              %[[D3:.+]] = affine.min #map(%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:              %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// CHECK-SAME:           iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK:                %[[D5:.+]] = affine.min #map1(%[[ARG3]])[%[[C32]], %[[C1280]]]
// CHECK:                %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]],
// CHECK-SAME:             10, 10, %[[D5]]] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
// CHECK:                %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8,
// CHECK-SAME:             8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// CHECK-SAME:             tensor<8x8x?x2x2x?xf32>
// CHECK:                %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:                  %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:               iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:                    %[[D8:.+]] = affine.apply #map2(%[[ARG7]])
// CHECK:                    %[[D9:.+]] = affine.min #map3(%[[D8]])
// CHECK:                    %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:                 iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:                      %[[D11:.+]] = affine.apply #map2(%[[ARG9]])
// CHECK:                      %[[D12:.+]] = affine.min #map3(%[[D11]])
// CHECK:                      %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// CHECK-SAME:                   iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:                        %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]],
// CHECK-SAME:                     %[[D8]], %[[D11]], %[[ARG11]]] [1, %[[D9]], %[[D12]], 1] [1, 1, 1, 1] :
// CHECK-SAME:                     tensor<?x10x10x?xf32> to tensor<?x?xf32>
// CHECK:                        %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) ->
// CHECK-SAME:                     tensor<8x8xf32>
// CHECK:                        %[[INSERTED_SLICE_4:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE_3]] into %[[D14]][0,
// CHECK-SAME:                     0] [%[[D9]], %[[D12]]] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
// CHECK:                        %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// CHECK-SAME:                     %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// CHECK-SAME:                     tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// CHECK:                        %[[D15:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_5]] :
// CHECK-SAME:                     tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                        %[[D16:.+]] = linalg.matmul {winograd.matmul = "I x B"} ins(%[[INSERTED_SLICE_4]],
// CHECK-SAME:                     %[[CST_0]] : tensor<8x8xf32>, tensor<8x8xf32>) outs(%[[D15]] : tensor<8x8xf32>) ->
// CHECK-SAME:                     tensor<8x8xf32>
// CHECK:                        %[[D17:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_5]] :
// CHECK-SAME:                     tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                        %[[D18:.+]] = linalg.matmul {winograd.matmul = "B' x I x B"} ins(%[[CST]], %[[D16]] :
// CHECK-SAME:                     tensor<8x8xf32>, tensor<8x8xf32>) outs(%[[D17]] : tensor<8x8xf32>) ->
// CHECK-SAME:                     tensor<8x8xf32>
// CHECK:                        %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[D18]] into %[[ARG12]][0, 0,
// CHECK-SAME:                     %[[ARG5]], %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// CHECK-SAME:                     tensor<8x8xf32> into tensor<8x8x?x2x2x?xf32>
// CHECK:                        scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                      }
// CHECK:                      scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                    }
// CHECK:                    scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                  }
// CHECK:                  scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                }
// CHECK:                %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// CHECK-SAME:             %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// CHECK-SAME:             tensor<8x8x1x2x2x1280xf32>
// CHECK:                scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:              }
// CHECK:              scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:            }
// CHECK:            return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:        }
