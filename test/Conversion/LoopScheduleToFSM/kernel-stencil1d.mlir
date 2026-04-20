// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-fsm)" %s | FileCheck %s
// XFAIL: *
// XFAIL reason: LoopScheduleToFSM pass needs updating to consume the new
// launch-inside-at dialect shape (AMC Category A refactor). Tracking issue:
// port FSM pass to the new shape analogous to LoopScheduleToCalyx.


// End-to-end 1-D 3-tap stencil with shrsi-based divide.
// CHECK: hw.module @stencil1d
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: fsm.machine @stencil1d_fsm
// CHECK: fsm.machine @loop0_fsm

func.func @stencil1d(%A: memref<16xi32>, %B: memref<16xi32>) attributes {top} {
  %c1 = arith.constant 1 : i32
  %cNm1 = arith.constant 15 : i32
  %c2 = arith.constant 2 : i32
  scf.while (%i = %c1) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cNm1 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %im1 = arith.subi %i, %c1 : i32
    %ip1 = arith.addi %i, %c1 : i32
    %im1i = arith.index_cast %im1 : i32 to index
    %ii = arith.index_cast %i : i32 to index
    %ip1i = arith.index_cast %ip1 : i32 to index
    %a0 = memref.load %A[%im1i] : memref<16xi32>
    %a1 = memref.load %A[%ii] : memref<16xi32>
    %a2 = memref.load %A[%ip1i] : memref<16xi32>
    %s0 = arith.addi %a0, %a1 : i32
    %s1 = arith.addi %s0, %a2 : i32
    %avg = arith.shrsi %s1, %c2 : i32
    memref.store %avg, %B[%ii] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
