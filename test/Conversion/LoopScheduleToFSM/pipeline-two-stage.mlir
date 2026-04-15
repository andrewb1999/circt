// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

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
// CHECK: fsm.state @STEP_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @DONE

func.func @pipeline_two_stage(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %step_result = loopschedule.step {
    %0 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
      // Stage 0: compute condition, next_i, and partial; pass to stage 1.
      %1:3 = loopschedule.pipeline.stage start = 0 end = 1 {
        %cond = arith.cmpi ult, %i, %c10 : index
        %next_i = arith.addi %i, %c1 : index
        %partial = arith.addi %acc, %arg0 : i32
        loopschedule.iter_arg_update %acc = %partial : i32
        loopschedule.iter_arg_update %i = %next_i : index
        loopschedule.register %next_i, %partial, %cond : index, i32, i1
      } : index, i32, i1
      // Stage 1: just pass through (accumulate is done in stage 0).
      %2 = loopschedule.pipeline.stage start = 1 end = 2 {
        loopschedule.register %1#1 : i32
      } : i32
      loopschedule.terminator condition(%1#2), results(%2) : i32
    }
    loopschedule.register %0 : i32
  } : i32
  return %step_result : i32
}
