// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule,lower-loopschedule-to-fsm)" %s | FileCheck %s

// Two INDEPENDENT copy loops (disjoint memories) schedule as two
// launches of ONE frame, and the function FSM must run them
// CONCURRENTLY: one shared FRAME state pulses both child_starts
// together, and one shared WAIT state holds until the group's combined
// done — the per-child sticky `done_seen` latches OR'd with the live
// dones, so children finishing on different cycles still conjoin.
// (Before the concurrent-sibling grouping, each launch got its own
// FRAME/WAIT pair and the loops ran back to back.)

// CHECK: hw.module @twocopy

// Combined group done: per-child seen latches OR'd with live dones, ANDed.
// CHECK-DAG: %frame0_done_seen_0 = seq.compreg
// CHECK-DAG: %frame0_done_seen_1 = seq.compreg

// Both children instantiated.
// CHECK-DAG: hw.instance "loop0_inst" @loop0
// CHECK-DAG: hw.instance "loop1_inst" @loop1

// One shared FRAME state raises both child_starts; one WAIT waits on the
// combined group done; no second FRAME/WAIT pair exists.
// CHECK: fsm.machine @twocopy_fsm(%arg0: i1, %arg1: i1)
// CHECK-SAME: argNames = ["start", "group_done_0"]
// CHECK-SAME: resNames = ["done", "child_start_0", "child_start_1",
// FRAME_0: done=0, child_start_0=1, child_start_1=1, both runnings=1.
// CHECK: fsm.state @FRAME_0 output {
// CHECK-NEXT: fsm.output %false, %true, %true, %true, %true, %false
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.transition @DONE guard {
// CHECK-NEXT: fsm.return %arg1
// CHECK-NOT: fsm.state @FRAME_1
// CHECK-NOT: fsm.state @WAIT_1

func.func @twocopy(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>, %d: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %v = memref.load %a[%idx] : memref<16xi32>
    memref.store %v, %b[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %v = memref.load %c[%idx] : memref<16xi32>
    memref.store %v, %d[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
