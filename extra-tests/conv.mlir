module {
  func.func @kernel_conv(%arg0: memref<16x16xi32>, %arg1: memref<2x2xi32>, %alloc: memref<16x16xi32>) {
    %0 = affine.load %alloc[0, 0] {from = "C"} : memref<16x16xi32>
    %1 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %0) -> (i32) {
      %2 = affine.for %arg4 = 0 to 2 iter_args(%arg5 = %arg3) -> (i32) {
        %3 = affine.load %arg0[%arg2, %arg4] {from = "A"} : memref<16x16xi32>
        %4 = affine.load %arg1[%arg2, %arg4] {from = "B"} : memref<2x2xi32>
        %5 = arith.extui %3 {unsigned} : i32 to i64
        %6 = arith.extui %4 {unsigned} : i32 to i64
        %7 = arith.muli %5, %6 : i64
        %8 = arith.trunci %7 {unsigned} : i64 to i32
        %9 = arith.addi %arg5, %8 : i32
        affine.yield %9 : i32
      } {hls.pipeline, hls.unroll = 1 : i64}
      affine.yield %2 : i32
    } {hls.unroll = 1 : i64}
    affine.store %1, %alloc[0, 0] : memref<16x16xi32>
    return
  }
}
