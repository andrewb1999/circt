// XFAIL: *
// RUN: circt-opt %s -lower-loopschedule-to-calyx -canonicalize -split-input-file | FileCheck %s

// XFAIL until the LoopScheduleToCalyx pass supports the new frame/at surface
// cleanly (currently crashes in BuildIntermediateRegs with
// getUniqueName(phase->getParentOp()) for the inner-at case).

// Pipeline register passed directly to the next stage.
// CHECK: calyx.while
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 20 iter_args(%counter = %const) : (index) -> () {
      %s0:2 = loopschedule.at 0 -> (index, i1) {
        %op = arith.addi %counter, %const : index
        %latch = arith.cmpi ult, %counter, %const : index
        loopschedule.iter_arg_update %counter = %op : index
        loopschedule.yield %op, %latch : index, i1
      }
      %s1 = loopschedule.at 1 -> index {
        loopschedule.yield %s0#0 : index
      }
      loopschedule.terminator condition(%s0#1), results()
    }
    return
  }
}

// -----

// Stage pipeline register passed to the next stage, also used in a computation.
// CHECK: calyx.while
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 20 iter_args(%counter = %const) : (index) -> () {
      %s0:2 = loopschedule.at 0 -> (index, i1) {
        %op = arith.addi %counter, %const : index
        %latch = arith.cmpi ult, %counter, %const : index
        loopschedule.iter_arg_update %counter = %op : index
        loopschedule.yield %op, %latch : index, i1
      }
      %s1 = loopschedule.at 1 -> index {
        %math = arith.addi %s0#0, %const : index
        loopschedule.yield %math : index
      }
      loopschedule.terminator condition(%s0#1), results()
    }
    return
  }
}
