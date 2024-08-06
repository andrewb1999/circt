module {
  func.func @kernel_fib(%arg0: memref<20xi32>) attributes {itypes = "u", otypes = "", top} {                                                                                       %c1_i32 = arith.constant 1 : i32
    affine.store %c1_i32, %arg0[1] {to = "A"} : memref<20xi32>
    affine.for %arg1 = 0 to 18 {
      %0 = affine.load %arg0[%arg1 + 1] : memref<20xi32>
      %1 = affine.load %arg0[%arg1] : memref<20xi32>
      %2 = arith.extui %0 {unsigned} : i32 to i33
      %3 = arith.extui %1 {unsigned} : i32 to i33
      %4 = arith.addi %2, %3 : i33
      %5 = arith.trunci %4 {unsigned} : i33 to i32
      affine.store %5, %arg0[%arg1 + 2] : memref<20xi32>
    } {loop_name = "i", op_name = "S_i_0", hls.pipeline}
    return
  }
}
