module {
  func.func @kernel_matmul(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) -> memref<16x16xi32> attributes {itypes = "ss", otypes = "s", top} {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<16x16xi32>
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 16 {
        affine.store %c0_i32, %alloc[%arg2, %arg3] : memref<16x16xi32>
      }
    }
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 16 {
          %0 = affine.load %arg0[%arg2, %arg4] : memref<16x16xi32>
          %1 = affine.load %arg1[%arg4, %arg3] : memref<16x16xi32>
          %2 = affine.load %alloc[%arg2, %arg3] : memref<16x16xi32>
          %3 = arith.muli %0, %1 : i32
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %alloc[%arg2, %arg3] : memref<16x16xi32>
        } {hls.pipeline}
      }
    }
    return %alloc : memref<16x16xi32>
  }
}
