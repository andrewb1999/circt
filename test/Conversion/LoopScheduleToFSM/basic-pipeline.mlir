// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{enable-pipeline-prearm=true})" %s | FileCheck %s


// Simple single-stage II=1 pipeline: accumulates arg0 for 10 iterations.
// Wrapped in a frame whose launch hosts the pipeline — the new shape.
// A second frame awaits the launch handle and forwards the pipeline's
// scalar result.
// CHECK-LABEL: hw.module @pipeline_add
// CHECK-SAME: in %arg0 : i32
// CHECK-SAME: in %clk : !seq.clock
// CHECK-SAME: in %rst : i1
// CHECK-SAME: in %start : i1
// CHECK-SAME: out done : i1
// CHECK-SAME: out result0 : i32
// Function-level FSM sequences the two frames
// CHECK: fsm.hw_instance
// Pipeline-local active register
// CHECK: seq.compreg.ce sym @loop0_active
// This pipeline pre-arms (constant inits, solo launch): the issue gate
// ORs the machine's registered issue_arm into `active`, so the first
// iteration issues in the launch cycle instead of one cycle later.
// CHECK: comb.or %loop0_active, %{{.+}}
// CHECK: hw.output

// CHECK-LABEL: fsm.machine @pipeline_add_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @FRAME_1
// CHECK: fsm.state @DONE

loopschedule.func_sequential @pipeline_add(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
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
        loopschedule.yield %pip : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    loopschedule.yield %hp : !loopschedule.handle
  }
  %result = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %r = loopschedule.at 0 -> i32 {
      loopschedule.yield %v : i32
    }
    loopschedule.yield %r : i32
  }
  loopschedule.return %result : i32
}
