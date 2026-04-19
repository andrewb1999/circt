// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation,lower-loopschedule-to-fsm)" %s | FileCheck %s

// Two-stage pipeline with II=2. CE fires every other cycle.
// CHECK-LABEL: hw.module @pipeline_ii2
// Counter for II=2 (1-bit counter cycling 0,1,0,1,...)
// CHECK: seq.compreg sym @loop0_ii_counter
// Counter logic: wrap at II-1
// CHECK: comb.icmp eq
// CHECK: comb.mux
// CE generation: ce_gen = (counter == 0) AND active
// CHECK: comb.icmp eq
// CHECK: comb.and
// active_ce = ce_gen AND cond
// CHECK: comb.and
// Traveling CE for stage 1
// CHECK: seq.compreg sym @loop0_ce_stage_1
// Stage 0 registers
// CHECK: seq.compreg.ce sym @loop0_s0_r0
// CHECK: seq.compreg.ce sym @loop0_s0_r1
// Stage 1 register
// CHECK: seq.compreg.ce sym @loop0_s1_r0
// CHECK: hw.output

// CHECK-LABEL: fsm.machine @pipeline_ii2_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @FRAME_1
// CHECK: fsm.state @DONE

func.func @pipeline_ii2(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.launch at 0 : !loopschedule.handle {
      %pip = loopschedule.pipeline II = 2 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
        %1:3 = loopschedule.at 0 -> (index, i32, i1) {
          %cond = arith.cmpi ult, %i, %c10 : index
          %next_i = arith.addi %i, %c1 : index
          %partial = arith.addi %acc, %arg0 : i32
          loopschedule.iter_arg_update %acc = %partial : i32
          loopschedule.iter_arg_update %i = %next_i : index
          loopschedule.yield %next_i, %partial, %cond : index, i32, i1
        }
        %2 = loopschedule.at 2 -> i32 {
          loopschedule.yield %1#1 : i32
        }
        loopschedule.terminator condition(%1#2), results(%2) : i32
      }
      loopschedule.yield %pip : i32
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
