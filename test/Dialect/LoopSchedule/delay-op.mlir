// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// Round-trip a sequential loop whose step body contains a loopschedule.delay
// region wrapping a load that issues 2 cycles after the step starts.
func.func @delay_basic(%arg0: memref<16xi32>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK-LABEL: func.func @delay_basic
  // CHECK: loopschedule.sequential
  %r = loopschedule.sequential iter_args(%i = %c0_i32) : (i32) -> i32 {
    // CHECK: loopschedule.step
    %0:3 = loopschedule.step {
      %cond = arith.cmpi slt, %i, %c10_i32 : i32
      %next = arith.addi %i, %c1_i32 : i32
      // CHECK:      loopschedule.delay 2 {
      // CHECK:        memref.load
      // CHECK:        loopschedule.register %{{.+}} : i32
      // CHECK-NEXT: } -> i32
      %d:1 = loopschedule.delay 2 {
        %idx = arith.index_cast %i : i32 to index
        %v = memref.load %arg0[%idx] : memref<16xi32>
        loopschedule.register %v : i32
      } -> i32
      loopschedule.iter_arg_update %i = %next : i32
      loopschedule.register %next, %d#0, %cond : i32, i32, i1
    } : i32, i32, i1
    loopschedule.terminator condition(%0#2), results(%0#1) : i32
  }
  return %r : i32
}

// Negative tests live in errors.mlir; this file only does positive round-trip.
