func.func @kernel(%arg0: memref<1024xi32>, %arg1: memref<1xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %accum = memref.alloc() : memref<1xi32>
  %0 = affine.for %arg2 = 0 to 1024 iter_args (%arg3 = %c0_i32) -> i32 {
    %1 = affine.load %arg0[%arg2] : memref<1024xi32>
    %2 = arith.muli %1, %arg3 : i32
    affine.yield %2 : i32
  } {hls.pipeline}
  affine.store %0, %arg1[%c0] : memref<1xi32>
  return
}
