// RUN: circt-opt %s -split-input-file -verify-diagnostics -allow-unregistered-dialect

func.func @no_phases() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op body must contain at least one phase}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    loopschedule.terminator condition(%false), iter_args(%false), results() : (i1) -> ()
  }
  return
}

// -----

func.func @bad_inner_op() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op stages may only contain 'loopschedule.pipeline.stage' or 'loopschedule.terminator' ops}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    "foo"() : () -> ()
    %0 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %arg0 : i1
    } : i1
    loopschedule.terminator condition(%0), iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func.func @cond_not_from_first_phase() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op loop condition must be produced by the first phase of the body}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    %0 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %arg0 : i1
    } : i1
    %1 = loopschedule.pipeline.stage start = 1 end = 2 {
      loopschedule.register %0 : i1
    } : i1
    loopschedule.terminator condition(%1), iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func.func @cond_not_from_phase() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op loop condition must be produced by the first phase of the body}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    %0 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %arg0 : i1
    } : i1
    loopschedule.terminator condition(%false), iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func.func @iter_arg_not_from_phase() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op New iter_args must be produced by a phase}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    %0 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %arg0 : i1
    } : i1
    loopschedule.terminator condition(%0), iter_args(%false), results(%0) : (i1) -> (i1)
  }
  return
}

// -----

func.func @mismatched_result_count() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op TerminatorOp does not produce the expected number of results}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    %0 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %arg0 : i1
    } : i1
    loopschedule.terminator condition(%0), iter_args(%0), results() : (i1) -> ()
  }
  return
}

// -----

func.func @invalid_results() {
  %false = arith.constant 0 : i1
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    %0 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'loopschedule.terminator' op 'results' must be defined by a phase op}}
    loopschedule.terminator condition(%0), iter_args(%0), results(%false) : (i1) -> (i1)
  }
  return
}

// -----

func.func @non_monotonic_start() {
  %false = arith.constant 0 : i1
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    %0 = loopschedule.pipeline.stage start = 1 end = 2 {
      loopschedule.register %arg0 : i1
    } : i1
    // expected-error @+1 {{'loopschedule.pipeline.stage' op 'start' must be after previous 'start' (1)}}
    %1 = loopschedule.pipeline.stage start = 0 end = 1 {
      loopschedule.register %0 : i1
    } : i1
    loopschedule.terminator condition(%0), iter_args(%1), results() : (i1) -> ()
  }
  return
}

// -----

func.func @delay_zero_latency(%arg0: memref<16xi32>) {
  %c0_i32 = arith.constant 0 : i32
  loopschedule.sequential iter_args(%i = %c0_i32) : (i32) -> () {
    %0:2 = loopschedule.step {
      %cond = arith.cmpi slt, %i, %c0_i32 : i32
      // expected-error @+1 {{'loopschedule.delay' op latency must be >= 1}}
      loopschedule.delay 0 {
        loopschedule.register
      }
      loopschedule.register %i, %cond : i32, i1
    } : i32, i1
    loopschedule.terminator condition(%0#1), iter_args(%0#0), results() : (i32) -> ()
  }
  return
}

// -----

func.func @delay_register_type_mismatch(%arg0: memref<16xi32>) {
  %c0_i32 = arith.constant 0 : i32
  loopschedule.sequential iter_args(%i = %c0_i32) : (i32) -> () {
    %0:2 = loopschedule.step {
      %cond = arith.cmpi slt, %i, %c0_i32 : i32
      // expected-error @+1 {{'loopschedule.delay' op delay op result types do not match register op types}}
      %d = loopschedule.delay 1 {
        loopschedule.register %cond : i1
      } -> i32
      loopschedule.register %i, %cond : i32, i1
    } : i32, i1
    loopschedule.terminator condition(%0#1), iter_args(%0#0), results() : (i32) -> ()
  }
  return
}
