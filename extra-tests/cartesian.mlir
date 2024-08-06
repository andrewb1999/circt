module {
  func.func @kernel_cartesian_product(%arg0: memref<16xi32>, %arg1: memref<16xi32>) -> memref<256x2xi32> attributes {itypes = "uu", otypes = "u", top} {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "C"} : memref<256x2xi32>
    affine.for %arg2 = 0 to 256 {
      affine.for %arg3 = 0 to 2 {
        affine.store %c0_i32, %alloc[%arg2, %arg3] : memref<256x2xi32>
      }
    }
    affine.for %arg2 = 0 to 16 {
      affine.for %arg3 = 0 to 16 {
        %0 = affine.load %arg0[%arg2] {from = "A"} : memref<16xi32>
        affine.store %0, %alloc[%arg2 * 16 + %arg3, 0] {to = "C"} : memref<256x2xi32>
        %1 = affine.load %arg1[%arg3] {from = "B"} : memref<16xi32>
        affine.store %1, %alloc[%arg2 * 16 + %arg3, 1] {to = "C"} : memref<256x2xi32>
      } {loop_name = "j", op_name = "S_j_0"}
    } {loop_name = "i", op_name = "S_i_0"}
    return %alloc : memref<256x2xi32>
  }
}
