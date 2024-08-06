module {
  func.func @kernel(%arg0: memref<20xi32>, %arg1: memref<20xi32>) -> memref<20xi32> attributes {itypes = "ss", otypes = "s", top} {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "C"} : memref<20xi32>
    affine.for %arg2 = 0 to 20 {
      affine.store %c0_i32, %alloc[%arg2] : memref<20xi32>
    } {hls.unroll = 1}
    affine.for %arg2 = 0 to 20 {
      %0 = affine.load %arg0[%arg2] {from = "A"} : memref<20xi32>
      %1 = affine.load %arg1[%arg2] {from = "B"} : memref<20xi32>
      %2 = arith.extsi %0 : i32 to i33
      %3 = arith.extsi %1 : i32 to i33
      %4 = arith.addi %2, %3 : i33
      %5 = arith.trunci %4 : i33 to i32
      affine.store %5, %alloc[%arg2] {to = "C"} : memref<20xi32>
    } {loop_name = "i", op_name = "S_i_0", hls.unroll = 1, hls.pipeline}
    return %alloc : memref<20xi32>
  }
}
