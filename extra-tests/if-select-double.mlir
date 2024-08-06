module {
  func.func @kernel_if_select(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>, %arg3: memref<16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    affine.for %arg4 = 0 to 16 {
      %0 = affine.load %arg2[%arg4] : memref<16xi32>
      %1 = arith.cmpi eq, %0, %c0_i32 : i32
      %3 = arith.cmpi eq, %0, %c1_i32 : i32
      scf.if %1 {
        %2 = affine.load %arg0[%arg4] : memref<16xi32>
        affine.store %2, %arg3[%arg4] : memref<16xi32>
      } else {
        scf.if %3 {
          %2 = affine.load %arg0[%arg4] : memref<16xi32>
          affine.store %2, %arg3[%arg4] : memref<16xi32>
        } else {
          %2 = affine.load %arg1[%arg4] : memref<16xi32>
          affine.store %2, %arg3[%arg4] : memref<16xi32>
        }
        %2 = affine.load %arg1[%arg4] : memref<16xi32>
        affine.store %2, %arg3[%arg4] : memref<16xi32>
      }
    } {hls.pipeline}
    return
  }
}
