// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// Two-stage II=1 pipeline with traveling CE.
// Stage 0: increment counter and compute partial
// Stage 1: finalize result
// CHECK-LABEL: hw.module @pipeline_two_stage
// CHECK: fsm.hw_instance
// CHECK: comb.and
// Traveling CE: stage 1 CE is a register of active_ce
// CHECK: seq.compreg sym @loop0_ce_stage_1
// Stage 0 registers
// CHECK: seq.compreg.ce sym @loop0_s0_r0
// CHECK: seq.compreg.ce sym @loop0_s0_r1
// Stage 1 registers
// CHECK: seq.compreg.ce sym @loop0_s1_r0
// Pipeline done: epilogue delay chain + tail check
// CHECK: comb.and
// CHECK: hw.output

// CHECK-LABEL: fsm.machine @pipeline_two_stage_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @FRAME_1
// CHECK: fsm.state @DONE

func.func @pipeline_two_stage(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
          // Stage 0: compute condition, next_i, and partial; pass to stage 1.
          %1:3 = loopschedule.at 0 -> (index, i32, i1) {
            %cond = arith.cmpi ult, %i, %c10 : index
            %next_i = arith.addi %i, %c1 : index
            %partial = arith.addi %acc, %arg0 : i32
            loopschedule.iter_arg_update %acc = %partial : i32
            loopschedule.iter_arg_update %i = %next_i : index
            loopschedule.yield %next_i, %partial, %cond : index, i32, i1
          }
          // Stage 1: just pass through (accumulate is done in stage 0).
          %2 = loopschedule.at 1 -> i32 {
            loopschedule.yield %1#1 : i32
          }
          loopschedule.terminator condition(%1#2), results(%2) : i32
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
  return %result : i32
}
