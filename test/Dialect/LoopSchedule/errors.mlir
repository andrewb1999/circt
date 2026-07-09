// RUN: circt-opt %s -split-input-file -verify-diagnostics -allow-unregistered-dialect

func.func @no_phases() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op body must contain at least one phase}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    loopschedule.terminator condition(%false), results()
  }
  return
}

// -----

func.func @bad_inner_op() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op stages may only contain 'loopschedule.at' or 'loopschedule.terminator' ops}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    "foo"() : () -> ()
    %0 = loopschedule.at 0 -> i1 {
      loopschedule.iter_arg_update %arg0 = %arg0 : i1
      loopschedule.yield %arg0 : i1
    }
    loopschedule.terminator condition(%0), results()
  }
  return
}

// -----

func.func @cond_not_from_phase() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op loop condition must be produced by a phase in the body}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    %0 = loopschedule.at 0 -> i1 {
      loopschedule.iter_arg_update %arg0 = %arg0 : i1
      loopschedule.yield %arg0 : i1
    }
    loopschedule.terminator condition(%false), results()
  }
  return
}

// -----

func.func @mismatched_result_count() {
  %false = arith.constant 0 : i1
  // expected-error @+1 {{'loopschedule.pipeline' op TerminatorOp does not produce the expected number of results}}
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    %0 = loopschedule.at 0 -> i1 {
      loopschedule.iter_arg_update %arg0 = %arg0 : i1
      loopschedule.yield %arg0 : i1
    }
    loopschedule.terminator condition(%0), results()
  }
  return
}

// -----

func.func @invalid_results() {
  %false = arith.constant 0 : i1
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> (i1) {
    %0 = loopschedule.at 0 -> i1 {
      loopschedule.iter_arg_update %arg0 = %arg0 : i1
      loopschedule.yield %arg0 : i1
    }
    // expected-error @+1 {{'loopschedule.terminator' op 'results' must be defined by a phase op}}
    loopschedule.terminator condition(%0), results(%false) : i1
  }
  return
}

// -----

func.func @non_monotonic_start() {
  %false = arith.constant 0 : i1
  loopschedule.pipeline II = 1 iter_args(%arg0 = %false) : (i1) -> () {
    %0 = loopschedule.at 1 -> i1 {
      loopschedule.iter_arg_update %arg0 = %arg0 : i1
      loopschedule.yield %arg0 : i1
    }
    // expected-error @+1 {{'loopschedule.at' op start time must be after previous start (1)}}
    %1 = loopschedule.at 0 -> i1 {
      loopschedule.yield %0 : i1
    }
    loopschedule.terminator condition(%0), results()
  }
  return
}


// -----

// expected-error @below {{control interface must be 'axil_handshake' or 'handshake', got 'bogus'}}
loopschedule.func_sequential control = bogus @bad_control(%a: memref<4xi32>) {
  loopschedule.return
}
