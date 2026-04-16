// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// Simple single-stage II=1 pipeline: accumulates arg0 for 10 iterations.
// Wrapped in a step so it goes through the standard step-based path.
// CHECK-LABEL: hw.module @pipeline_add
// CHECK-SAME: in %arg0 : i32
// CHECK-SAME: in %clk : !seq.clock
// CHECK-SAME: in %rst : i1
// CHECK-SAME: in %start : i1
// CHECK-SAME: out done : i1
// CHECK-SAME: out result0 : i32
// Function-level FSM sequences the single step
// CHECK: fsm.hw_instance
// Pipeline-local active register
// CHECK: seq.compreg sym @loop0_active
// CHECK: hw.output

// CHECK-LABEL: fsm.machine @pipeline_add_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @STEP_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @DONE

func.func @pipeline_add(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %step_result = loopschedule.step {
    %0 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
      %1:3 = loopschedule.at 0 -> (index, i32, i1) {
        %cond = arith.cmpi ult, %i, %c10 : index
        %next_i = arith.addi %i, %c1 : index
        %sum = arith.addi %acc, %arg0 : i32
        loopschedule.iter_arg_update %acc = %sum : i32
        loopschedule.iter_arg_update %i = %next_i : index
        loopschedule.yield %next_i, %sum, %cond : index, i32, i1
      }
      loopschedule.terminator condition(%1#2), results(%1#1) : i32
    }
    loopschedule.register %0 : i32
  } : i32
  return %step_result : i32
}
