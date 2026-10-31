// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// loopschedule.expect round-trip: the handle flows directly from a launch
// in an earlier at-stage to an expect in a later at-stage of the same
// pipeline. Pipeline wrappers (terminator, iter-arg machinery) are kept
// minimal.

// CHECK-LABEL: func.func @expect_same_stage_consumer
func.func @expect_same_stage_consumer(%A: memref<16xi32>, %idx: index) -> i32 {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c15 = arith.constant 15 : i4
  %out = loopschedule.pipeline II = 1 trip_count = 1 iter_args(%iv = %c0) : (i4) -> i32 {
    // CHECK: %[[H:.+]] = loopschedule.at 0 -> !loopschedule.handle
    // CHECK: loopschedule.launch : !loopschedule.handle
    %h = loopschedule.at 0 -> !loopschedule.handle {
      %launched = loopschedule.launch : !loopschedule.handle {
        %val = memref.load %A[%idx] : memref<16xi32>
        loopschedule.yield %val : i32
      }
      loopschedule.yield %launched : !loopschedule.handle
    }
    // CHECK: loopschedule.expect %[[H]] : i32
    %v, %done = loopschedule.at 1 -> (i32, i1) {
      %ld = loopschedule.expect %h : i32
      %cond = arith.cmpi slt, %iv, %c15 : i4
      %nextIv = arith.addi %iv, %c1 : i4
      loopschedule.iter_arg_update %iv = %nextIv : i4
      loopschedule.yield %ld, %cond : i32, i1
    }
    loopschedule.terminator condition(%done), results(%v) : i32
  }
  return %out : i32
}

// A predicated pair (a load under an `if`): the launch issues where its
// condition holds, and the expect, on the same condition carried to its
// stage, waits and consumes only there.

// CHECK-LABEL: func.func @expect_predicated
func.func @expect_predicated(%A: memref<16xi32>, %idx: index, %c: i1) -> i32 {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c15 = arith.constant 15 : i4
  %out = loopschedule.pipeline II = 1 trip_count = 1 iter_args(%iv = %c0) : (i4) -> i32 {
    // CHECK: %[[S:.+]]:2 = loopschedule.at 0 -> (!loopschedule.handle, i1)
    // CHECK: loopschedule.launch if %{{.+}} : !loopschedule.handle
    %h, %p = loopschedule.at 0 -> (!loopschedule.handle, i1) {
      %launched = loopschedule.launch if %c : !loopschedule.handle {
        %val = memref.load %A[%idx] : memref<16xi32>
        loopschedule.yield %val : i32
      }
      loopschedule.yield %launched, %c : !loopschedule.handle, i1
    }
    // CHECK: loopschedule.expect %[[S]]#0 if %[[S]]#1 : i32
    %v, %done = loopschedule.at 1 -> (i32, i1) {
      %ld = loopschedule.expect %h if %p : i32
      %cond = arith.cmpi slt, %iv, %c15 : i4
      %nextIv = arith.addi %iv, %c1 : i4
      loopschedule.iter_arg_update %iv = %nextIv : i4
      loopschedule.yield %ld, %cond : i32, i1
    }
    loopschedule.terminator condition(%done), results(%v) : i32
  }
  return %out : i32
}
