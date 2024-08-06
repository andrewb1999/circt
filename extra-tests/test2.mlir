module {
  func.func @kernel_fib(%arg0: memref<20xi32>) attributes {itypes = "u", otypes = "", top} {                                                                                       %c1_i32 = arith.constant 1 : i32
    affine.for %arg2 = 0 to 10 {
    affine.for %arg1 = 0 to 18 {
      affine.store %c1_i32, %arg0[%arg1] {to = "A"} : memref<20xi32>
      %0 = affine.load %arg0[%arg1] : memref<20xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    }
    return
  }
}
