#map = affine_map<() -> (16)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0] -> (d0 mod s0)>
#map3 = affine_map<(d0)[s0] -> (d0 floordiv s0)>
module {
  func.func @kernel(%arg0: memref<16xi32>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 8 {
      affine.for %arg2 = 2 to 16 {
        %0 = affine.load %arg0[%arg2 - 2] : memref<16xi32>
        %1 = arith.addi %0, %arg1 : i32
        affine.store %1, %arg0[%arg2] : memref<16xi32>
      } {hls.unroll = 1 : i64, hls.pipeline}
    }
    return
  }
}
