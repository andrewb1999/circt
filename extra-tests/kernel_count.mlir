module {
  func.func @kernel_counting_sort(%arg0: memref<16xi32>) -> memref<16xi32> attributes {itypes = "u", otypes = "u", top} {
    %c1_i34 = arith.constant 1 : i34
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {name = "count"} : memref<21xi32>
    affine.for %arg1 = 0 to 21 {
      affine.store %c0_i32, %alloc[%arg1] : memref<21xi32>
    }
    affine.for %arg1 = 0 to 16 {
      %0 = affine.load %arg0[%arg1] {from = "A"} : memref<16xi32>
      %1 = arith.index_cast %0 : i32 to index
      %2 = memref.load %alloc[%1] {from = "count"} : memref<21xi32>
      %3 = arith.addi %2, %c1_i32 : i32
      memref.store %3, %alloc[%1] {to = "count"} : memref<21xi32>
    } {loop_name = "i", op_name = "S_i_0", hls.pipeline}
    %alloc_0 = memref.alloc() : memref<21xi32>
    affine.for %arg1 = 0 to 21 {
      %0 = affine.load %alloc[%arg1] : memref<21xi32>
      affine.store %0, %alloc_0[%arg1] : memref<21xi32>
    }
    %alloc_1 = memref.alloc() {name = "count_copy"} : memref<21xi32>
    affine.for %arg1 = 0 to 21 {
      %0 = affine.load %alloc_0[%arg1] : memref<21xi32>
      affine.store %0, %alloc_1[%arg1] : memref<21xi32>
    }
    affine.for %arg1 = 0 to 20 {
      %0 = affine.load %alloc_1[%arg1] : memref<21xi32>
      %1 = affine.load %alloc_1[%arg1 + 1] : memref<21xi32>
      %2 = arith.addi %1, %0 : i32
      affine.store %2, %alloc_1[%arg1 + 1] : memref<21xi32>
    } {loop_name = "i", op_name = "S_i_1"}
    %alloc_2 = memref.alloc() : memref<21xi32>
    affine.for %arg1 = 0 to 21 {
      %0 = affine.load %alloc_1[%arg1] : memref<21xi32>
      affine.store %0, %alloc_2[%arg1] : memref<21xi32>
    }
    %alloc_3 = memref.alloc() {name = "count_copy2"} : memref<21xi32>
    affine.for %arg1 = 0 to 21 {
      %0 = affine.load %alloc_2[%arg1] : memref<21xi32>
      affine.store %0, %alloc_3[%arg1] : memref<21xi32>
    }
    %alloc_4 = memref.alloc() {name = "C"} : memref<16xi32>
    affine.for %arg1 = 0 to 16 {
      affine.store %c0_i32, %alloc_4[%arg1] : memref<16xi32>
    }
    affine.for %arg1 = 0 to 16 {
      %0 = affine.load %arg0[%arg1] {from = "A"} : memref<16xi32>
      %1 = arith.index_cast %0 : i32 to index
      %2 = memref.load %alloc_3[%1] {from = "count_copy2"} : memref<21xi32>
      %3 = arith.extui %2 : i32 to i34
      %4 = arith.subi %3, %c1_i34 : i34
      %5 = arith.index_cast %4 : i34 to index
      memref.store %0, %alloc_4[%5] {to = "C"} : memref<16xi32>
      %6 = affine.load %arg0[%arg1] {from = "A"} : memref<16xi32>
      %7 = arith.index_cast %6 : i32 to index
      %8 = memref.load %alloc_3[%7] {from = "count_copy2"} : memref<21xi32>
      %9 = arith.subi %8, %c1_i32 : i32
      memref.store %9, %alloc_3[%7] {to = "count_copy2"} : memref<21xi32>
    } {loop_name = "i", op_name = "S_i_2", hls.pipeline}
    return %alloc_4 : memref<16xi32>
  }
}
