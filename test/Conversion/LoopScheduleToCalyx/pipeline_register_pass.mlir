// RUN: circt-opt %s -lower-loopschedule-to-calyx -canonicalize -split-input-file | FileCheck %s

// Pipeline register passed directly to the next stage. The pipeline has a
// known trip count, so the pass emits `calyx.static_repeat` (not a while
// loop) wrapping the stages.
// CHECK-LABEL: calyx.component @foo
// CHECK: calyx.static_repeat
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
// CHECK-LABEL: calyx.component @foo
// CHECK: calyx.static_repeat
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
