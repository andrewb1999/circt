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
// Pipeline done: NOT(cond) AND NOT(ce_stage_1)
// CHECK: comb.and
// CHECK: hw.output

// CHECK-LABEL: fsm.machine @pipeline_two_stage_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @loop0_EXECUTE
// CHECK: fsm.state @loop0_DONE

func.func @pipeline_two_stage(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
    %cond = arith.cmpi ult, %i, %c10 : index
    loopschedule.register %cond : i1
  } do {
    // Stage 0: compute next_i and read acc, pass to stage 1.
    %1:2 = loopschedule.pipeline.stage start = 0 end = 1 {
      %next_i = arith.addi %i, %c1 : index
      %partial = arith.addi %acc, %arg0 : i32
      loopschedule.register %next_i, %partial : index, i32
    } : index, i32
    // Stage 1: just pass through (accumulate is done in stage 0).
    %2 = loopschedule.pipeline.stage start = 1 end = 2 {
      loopschedule.register %1#1 : i32
    } : i32
    loopschedule.terminator iter_args(%1#0, %1#1), results(%2) : (index, i32) -> (i32)
  }
  return %0 : i32
}
