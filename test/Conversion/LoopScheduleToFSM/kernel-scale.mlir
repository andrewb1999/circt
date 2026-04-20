// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end scalar-memref multiply, multi-cycle mul.
// CHECK: hw.module @scale
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: fsm.machine @scale_fsm
// CHECK-DAG: fsm.state @FRAME_0
// CHECK-DAG: fsm.state @DONE
// CHECK: fsm.machine @loop0_fsm

func.func @scale(%a: i32, %A: memref<16xi32>, %B: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %av = memref.load %A[%idx] : memref<16xi32>
    %mul = arith.muli %a, %av : i32
    memref.store %mul, %B[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
