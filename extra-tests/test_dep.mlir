module {
  func.func @kernel_deps(%arg0: memref<20x20x20xi32>, %arg1: i32) attributes {itypes = "ss", otypes = "", top} {
    affine.for %arg2 = 0 to 19 {
      affine.for %arg3 = 0 to 20 {
        affine.for %arg4 = 1 to 20 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] {from = "A"} : memref<20x20x20xi32>
          %1 = arith.extsi %0 : i32 to i33
          %2 = arith.extsi %arg1 : i32 to i33
          %3 = arith.addi %1, %2 : i33
          %4 = arith.trunci %3 : i33 to i32
          affine.store %4, %arg0[%arg2, %arg3, %arg4 - 1] {to = "A"} : memref<20x20x20xi32>
        } {loop_name = "k", op_name = "S_k_0"}
      } {loop_name = "j", op_name = "S_j_0"}
    } {loop_name = "i", op_name = "S_i_0"}
    return
  }
}
