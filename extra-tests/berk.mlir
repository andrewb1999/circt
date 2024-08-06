module {
  func.func @kernel_index_add(%arg0: memref<16xi32>) -> memref<16xi32> attributes {itypes = "u", otypes = "u"} {
    %alloc = memref.alloc() {name = "C"} : memref<16xi32>
    affine.for %arg1 = 0 to 16 {
      %0 = affine.load %alloc[%arg1] {from = "C"} : memref<16xi32>
      %1 = arith.index_castui %arg1 : index to i33
      %2 = arith.trunci %1 : i33 to i32
      %3 = arith.addi %0, %2 : i32
      affine.store %3, %alloc[%arg1] {to = "C"} : memref<16xi32>
    } {loop_name = "i", op_name = "S_i_0"}
    return %alloc : memref<16xi32>
  }
}
