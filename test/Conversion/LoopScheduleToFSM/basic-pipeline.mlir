// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// Simple single-stage II=1 pipeline: accumulates arg0 for 10 iterations.
// CHECK-LABEL: hw.module @pipeline_add
// CHECK-SAME: in %arg0 : i32
// CHECK-SAME: in %clk : !seq.clock
// CHECK-SAME: in %rst : i1
// CHECK-SAME: in %start : i1
// CHECK-SAME: out done : i1
// CHECK-SAME: out result0 : i32
// CHECK: fsm.hw_instance
// Pipeline-local active register
// CHECK: seq.compreg sym @loop0_active
// CHECK: comb.and
// CHECK: seq.compreg.ce
// CHECK: seq.compreg.ce
// CHECK: comb.mux
// CHECK: comb.mux
// CHECK: hw.output

// CHECK-LABEL: fsm.machine @pipeline_add_fsm
// CHECK: fsm.state @IDLE output
// CHECK: fsm.output %{{.*}} : i1
// CHECK: fsm.transition @loop0_EXECUTE
// CHECK: fsm.state @loop0_EXECUTE output
// CHECK: fsm.output %{{.*}} : i1
// CHECK: fsm.transition @loop0_DONE
// CHECK: fsm.state @loop0_DONE output
// CHECK: fsm.output %{{.*}} : i1
// CHECK: fsm.transition @IDLE

func.func @pipeline_add(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
    %cond = arith.cmpi ult, %i, %c10 : index
    loopschedule.register %cond : i1
  } do {
    %1:2 = loopschedule.pipeline.stage start = 0 end = 1 {
      %next_i = arith.addi %i, %c1 : index
      %sum = arith.addi %acc, %arg0 : i32
      loopschedule.register %next_i, %sum : index, i32
    } : index, i32
    loopschedule.terminator iter_args(%1#0, %1#1), results(%1#1) : (index, i32) -> (i32)
  }
  return %0 : i32
}
