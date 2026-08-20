// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule,lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end vector add: SCF -> LoopSchedule -> FSM + HW.
// The loop inlines into the single per-function machine as prefixed
// loop0_* states; the last body state exits straight to DONE, so no
// separate WAIT state (and no hw.module @loop0) remains.
// CHECK: hw.module @vadd
// CHECK: fsm.hw_instance "vadd_fsm_inst" @vadd_fsm
// CHECK: fsm.machine @vadd_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @loop0_FRAME_0_0
// CHECK: fsm.state @loop0_FRAME_0_2
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0

func.func @vadd(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %av = memref.load %a[%idx] : memref<16xi32>
    %bv = memref.load %b[%idx] : memref<16xi32>
    %sum = arith.addi %av, %bv : i32
    memref.store %sum, %c[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
