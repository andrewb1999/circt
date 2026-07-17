// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// Fork-join band: two order-free lane loops communicating with the
// outside world through memory, no results.

// CHECK-LABEL: func.func @par_no_results
// CHECK: loopschedule.par
// CHECK-COUNT-2: affine.for
// CHECK: }
func.func @par_no_results(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  loopschedule.par {
    affine.for %i = 0 to 4 {
      %v = affine.load %arg0[%i * 2] : memref<8xi32>
      affine.store %v, %arg1[%i * 2] : memref<8xi32>
    }
    affine.for %j = 0 to 4 {
      %v = affine.load %arg0[%j * 2 + 1] : memref<8xi32>
      affine.store %v, %arg1[%j * 2 + 1] : memref<8xi32>
    }
  }
  return
}

// Reduction lanes: each child carries its own accumulator; the yield
// hands each lane's result to the join, and the combine lives AFTER the
// op as ordinary arith (emitted by the split pass, consumed by either
// mechanism lowering).

// CHECK-LABEL: func.func @par_reduction_lanes
// CHECK: %[[R:.*]]:2 = loopschedule.par
// CHECK: loopschedule.yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: } : i32, i32
// CHECK: arith.addi %[[R]]#0, %[[R]]#1
func.func @par_reduction_lanes(%arg0: memref<8xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %r:2 = loopschedule.par {
    %r0 = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> (i32) {
      %v = affine.load %arg0[%i * 2] : memref<8xi32>
      %s = arith.addi %acc, %v : i32
      affine.yield %s : i32
    }
    %r1 = affine.for %j = 0 to 4 iter_args(%acc = %zero) -> (i32) {
      %v = affine.load %arg0[%j * 2 + 1] : memref<8xi32>
      %s = arith.addi %acc, %v : i32
      affine.yield %s : i32
    }
    loopschedule.yield %r0, %r1 : i32, i32
  } : i32, i32
  %sum = arith.addi %r#0, %r#1 : i32
  return %sum : i32
}
