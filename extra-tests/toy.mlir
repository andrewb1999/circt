module {
  func.func @toy_kernel(%arg0: memref<20x25xi32>, %arg1: memref<20x25xi32>) attributes {itypes = "ss", otypes = "", top} {
    %c2_i32 = arith.constant 2 : i32
    affine.for %arg2 = 0 to 20 {
      affine.for %arg3 = 0 to 25 {
        %0 = affine.load %arg0[%arg2, %arg3] {from = "out_AB"} : memref<20x25xi32>
        %1 = arith.addi %0, %c2_i32 : i32
        affine.store %1, %arg0[%arg2, %arg3] {to = "out_AB"} : memref<20x25xi32>
      } {loop_name = "j0"}
    } {loop_name = "i0", op_name = "mm1"}
    affine.for %arg2 = 0 to 20 {
      affine.for %arg3 = 0 to 25 {
        %0 = affine.load %arg0[%arg2, %arg3] {from = "out_AB"} : memref<20x25xi32>
        affine.store %0, %arg1[%arg2, %arg3] {to = "output"} : memref<20x25xi32>
      } {loop_name = "j2"}
    } {loop_name = "i2", op_name = "S_i2_j2_1"}
    return
  }
}
