// module {
//   func.func @kernel_matmul() -> memref<16x16xi32> {
//     %c0_i32 = arith.constant 0 : i32
//     %alloc = memref.alloc() : memref<16x16xi32>
//     affine.for %arg0 = 0 to 16 {
//       affine.for %arg1 = 0 to 16 {
//         affine.store %c0_i32, %alloc[%arg0, %arg1] : memref<16x16xi32>
//       }
//     }
//     return %alloc : memref<16x16xi32>
//   }
// }

module {
  func.func @kernel_matmul() {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0_i4 = arith.constant 0 : i4
    %c15_i4 = arith.constant 15 : i4
    %c1_i4 = arith.constant 1 : i4
    %alloc = memref.alloc() : memref<16x16xi32>
    
    %7:2 = scf.for %arg0 = %c0 to %c256 step %c1 iter_args(%arg1 = %c0_i4, %arg2 = %c0_i4) -> (i4, i4) {
      %5 = arith.index_castui %arg1 : i4 to index 
      %6 = arith.index_castui %arg2 : i4 to index 
      memref.store %c0_i32, %alloc[%5, %6] : memref<16x16xi32>
      %0 = arith.cmpi eq, %arg2, %c15_i4 : i4
      %1 = arith.addi %arg2, %c1_i4 : i4
      %2 = arith.select %0, %c0_i4, %1 : i4
      %3 = arith.addi %arg1, %c1_i4 : i4
      %4 = arith.select %0, %3, %arg1 : i4
      scf.yield %4, %2 : i4, i4
    } {hls.pipeline, loopschedule.trip_count = 256 : i64}
    return
  }
}
