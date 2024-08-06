func.func @kernel(%arg0: memref<1024xi32>, %arg1: memref<1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %accum = memref.alloc() : memref<1xi32>
  affine.for %arg2 = 0 to 1024 {
    %1 = affine.load %arg0[%arg2] : memref<1024xi32>
    %2 = affine.load %accum[%c0] : memref<1xi32>
    %3 = arith.muli %1, %2 : i32
    affine.store %3, %accum[%c0] : memref<1xi32>
  } {hls.pipeline}
  %0 = affine.load %accum[%c0] : memref<1xi32>
  affine.store %0, %arg1[%c0] : memref<1xi32>
  return
}
