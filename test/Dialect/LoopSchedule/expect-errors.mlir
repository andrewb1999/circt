// RUN: circt-opt %s -split-input-file -verify-diagnostics

// Case 1: expect in an at-stage that is NOT strictly after the launch's at-stage.
func.func @same_stage(%A: memref<16xi32>, %idx: index) -> i32 {
  %c0 = arith.constant 0 : i4
  %c15 = arith.constant 15 : i4
  %c1 = arith.constant 1 : i4
  %out = loopschedule.pipeline II = 1 trip_count = 1 iter_args(%iv = %c0) : (i4) -> i32 {
    %v, %done = loopschedule.at 0 -> (i32, i1) {
      %h = loopschedule.launch : !loopschedule.handle {
        %ld = memref.load %A[%idx] : memref<16xi32>
        loopschedule.yield %ld : i32
      }
      // expected-error @+1 {{expect's `at` offset (0) must be strictly greater than the launch's `at` offset (0)}}
      %val = loopschedule.expect %h : i32
      %cond = arith.cmpi slt, %iv, %c15 : i4
      %n = arith.addi %iv, %c1 : i4
      loopschedule.iter_arg_update %iv = %n : i4
      loopschedule.yield %val, %cond : i32, i1
    }
    loopschedule.terminator condition(%done), results(%v) : i32
  }
  return %out : i32
}

// -----

// Case 2: expect's result type does not match the launch's yield.
func.func @type_mismatch(%A: memref<16xi32>, %idx: index) -> i64 {
  %c0 = arith.constant 0 : i4
  %c15 = arith.constant 15 : i4
  %c1 = arith.constant 1 : i4
  %out = loopschedule.pipeline II = 1 trip_count = 1 iter_args(%iv = %c0) : (i4) -> i64 {
    %h = loopschedule.at 0 -> !loopschedule.handle {
      %launched = loopschedule.launch : !loopschedule.handle {
        %ld = memref.load %A[%idx] : memref<16xi32>
        loopschedule.yield %ld : i32
      }
      loopschedule.yield %launched : !loopschedule.handle
    }
    %v, %done = loopschedule.at 1 -> (i64, i1) {
      // expected-error @+1 {{result #0 type 'i64' does not match launch yield operand type 'i32'}}
      %val = loopschedule.expect %h : i64
      %cond = arith.cmpi slt, %iv, %c15 : i4
      %n = arith.addi %iv, %c1 : i4
      loopschedule.iter_arg_update %iv = %n : i4
      loopschedule.yield %val, %cond : i64, i1
    }
    loopschedule.terminator condition(%done), results(%v) : i64
  }
  return %out : i64
}

// -----

// Case 3: launch body contains more than one non-terminator op.
func.func @launch_multiple_ops(%A: memref<16xi32>, %idx: index) -> i32 {
  %c0 = arith.constant 0 : i4
  %c15 = arith.constant 15 : i4
  %c1 = arith.constant 1 : i4
  %out = loopschedule.pipeline II = 1 trip_count = 1 iter_args(%iv = %c0) : (i4) -> i32 {
    %h = loopschedule.at 0 -> !loopschedule.handle {
      // expected-error @+1 {{body must contain exactly one non-terminator op (found 2)}}
      %launched = loopschedule.launch : !loopschedule.handle {
        %a = memref.load %A[%idx] : memref<16xi32>
        %b = arith.addi %a, %a : i32
        loopschedule.yield %b : i32
      }
      loopschedule.yield %launched : !loopschedule.handle
    }
    %v, %done = loopschedule.at 1 -> (i32, i1) {
      %val = loopschedule.expect %h : i32
      %cond = arith.cmpi slt, %iv, %c15 : i4
      %n = arith.addi %iv, %c1 : i4
      loopschedule.iter_arg_update %iv = %n : i4
      loopschedule.yield %val, %cond : i32, i1
    }
    loopschedule.terminator condition(%done), results(%v) : i32
  }
  return %out : i32
}

// -----

// Case 4: expect on a handle that isn't produced by a loopschedule.launch.
func.func @handle_not_from_launch(%h: !loopschedule.handle) -> i32 {
  %c0 = arith.constant 0 : i4
  %c15 = arith.constant 15 : i4
  %c1 = arith.constant 1 : i4
  %out = loopschedule.pipeline II = 1 trip_count = 1 iter_args(%iv = %c0) : (i4) -> i32 {
    %v, %done = loopschedule.at 0 -> (i32, i1) {
      // expected-error @+1 {{handle must originate from a `loopschedule.launch`}}
      %val = loopschedule.expect %h : i32
      %cond = arith.cmpi slt, %iv, %c15 : i4
      %n = arith.addi %iv, %c1 : i4
      loopschedule.iter_arg_update %iv = %n : i4
      loopschedule.yield %val, %cond : i32, i1
    }
    loopschedule.terminator condition(%done), results(%v) : i32
  }
  return %out : i32
}
