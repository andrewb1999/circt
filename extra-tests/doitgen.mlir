module {
  func.func @kernel_doitgen(%arg0: memref<16x8x8xi32>, %arg1: memref<8x8xi32>) attributes {itypes = "ss", otypes = "", top} {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "sum_"} : memref<8xi32>
    affine.for %arg2 = 0 to 8 {
      affine.store %c0_i32, %alloc[%arg2] : memref<8xi32>
    }
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 8 {
        affine.for %arg4 = 0 to 8 {
          affine.for %arg5 = 0 to 8 {
            %0 = affine.load %alloc[%arg4] {from = "sum_"} : memref<8xi32>
            %1 = affine.load %arg0[%arg2, %arg3, %arg5] {from = "A"} : memref<16x8x8xi32>
            %2 = affine.load %arg1[%arg5, %arg4] {from = "x"} : memref<8x8xi32>
            %3 = arith.extsi %1 : i32 to i64
            %4 = arith.extsi %2 : i32 to i64
            %5 = arith.muli %3, %4 : i64
            %6 = arith.extsi %0 : i32 to i65
            %7 = arith.extsi %5 : i64 to i65
            %8 = arith.addi %6, %7 : i65
            %9 = arith.trunci %8 : i65 to i32
            affine.store %9, %alloc[%arg4] {to = "sum_"} : memref<8xi32>
          } {loop_name = "s", op_name = "S_s_0"}
        } {loop_name = "p", op_name = "S_p_0"}
        affine.for %arg4 = 0 to 8 {
          %0 = affine.load %alloc[%arg4] {from = "sum_"} : memref<8xi32>
          affine.store %0, %arg0[%arg2, %arg3, %arg4] {to = "A"} : memref<16x8x8xi32>
        } {loop_name = "p1", op_name = "S_p1_2"}
      } {loop_name = "q"}
    } {loop_name = "r", op_name = "S_r_q_0"}
    return
  }
}
