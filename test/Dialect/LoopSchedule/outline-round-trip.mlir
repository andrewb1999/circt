// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// A boundary marker with no results: the region only touches memory, so
// nothing crosses out and the yield is empty.

// CHECK-LABEL: func.func @outline_no_results
// CHECK: loopschedule.outline
// CHECK: affine.for
// CHECK: }
func.func @outline_no_results(%arg0: memref<8xi32>, %arg1: memref<8xi32>) {
  loopschedule.outline {
    affine.for %i = 0 to 8 {
      %v = affine.load %arg0[%i] : memref<8xi32>
      affine.store %v, %arg1[%i] : memref<8xi32>
    }
  }
  return
}

// Results: values defined inside the boundary and used outside must be
// yielded — the yield IS the boundary's result interface. Capture the
// other direction is implicit (%init, %arg0 are read directly).

// CHECK-LABEL: func.func @outline_results
// CHECK: %[[R:.*]] = loopschedule.outline
// CHECK: loopschedule.yield %{{.*}} : i32
// CHECK: } : i32
// CHECK: return %[[R]] : i32
func.func @outline_results(%arg0: memref<8xi32>, %init: i32) -> i32 {
  %sum = loopschedule.outline {
    %r = affine.for %i = 0 to 8 iter_args(%acc = %init) -> i32 {
      %v = affine.load %arg0[%i] : memref<8xi32>
      %n = arith.addi %acc, %v : i32
      affine.yield %n : i32
    }
    loopschedule.yield %r : i32
  } : i32
  return %sum : i32
}
