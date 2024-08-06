module {
  func.func @kernel(%arg0: memref<20xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: memref<20xi32>) attributes {itypes = "uuuuu", otypes = ""} {
    affine.for %arg5 = 0 to 20 {
      %0 = affine.load %arg0[%arg5] {from = "vals"} : memref<20xi32>
      %1 = affine.load %arg2[%arg5] {from = "col"} : memref<20xi32>
      %2 = arith.index_cast %1 : i32 to index
      %3 = memref.load %arg3[%2] {from = "denseVec"} : memref<20xi32>
      %4 = arith.extui %0 {unsigned} : i32 to i64
      %5 = arith.extui %3 {unsigned} : i32 to i64
      %6 = arith.muli %4, %5 : i64
      %7 = affine.load %arg1[%arg5] {from = "row"} : memref<20xi32>
      %8 = arith.index_cast %7 : i32 to index
      %9 = memref.load %arg4[%8] {from = "outputVec"} : memref<20xi32>
      %10 = arith.trunci %6 {unsigned} : i64 to i32
      %11 = arith.addi %9, %10 : i32
      %12 = arith.index_cast %7 : i32 to index
      memref.store %11, %arg4[%12] {to = "outputVec"} : memref<20xi32>
    } {hls.pipeline, hls.unroll = 1 : i64}
    return
  }
}
