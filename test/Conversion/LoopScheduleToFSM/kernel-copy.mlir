// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end memref copy.
// CHECK: hw.module @copy
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: fsm.machine @copy_fsm
// CHECK-DAG: fsm.state @FRAME_0
// CHECK-DAG: fsm.state @DONE
// CHECK: fsm.machine @loop0_fsm

func.func @copy(%a: memref<16xi32>, %b: memref<16xi32>) attributes {top} {
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
  return
}
