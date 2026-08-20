// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule,lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end saxpy. The loop inlines into the single per-function machine
// as loop0_* states (seven cycle states for the pipelined mul); the last
// body state exits straight to DONE, so no separate WAIT state remains.
// CHECK: hw.module @saxpy
// CHECK: fsm.hw_instance "saxpy_fsm_inst" @saxpy_fsm
// CHECK: fsm.machine @saxpy_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @loop0_FRAME_0_0
// CHECK: fsm.state @loop0_FRAME_0_6
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0

func.func @saxpy(%a: i32, %x: memref<16xi32>, %y: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %xv = memref.load %x[%idx] : memref<16xi32>
    %yv = memref.load %y[%idx] : memref<16xi32>
    %mul = arith.muli %a, %xv : i32
    %sum = arith.addi %mul, %yv : i32
    memref.store %sum, %y[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
