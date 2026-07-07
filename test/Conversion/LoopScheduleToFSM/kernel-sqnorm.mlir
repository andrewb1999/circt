// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule,lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end squared-norm reduction. Same failure mode as kernel-dot.mlir.
// CHECK: hw.module @sqnorm
// CHECK: fsm.machine @sqnorm_fsm

func.func @sqnorm(%A: memref<16xi32>) -> i32 attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  %res:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %idx = arith.index_cast %i : i32 to index
    %av = memref.load %A[%idx] : memref<16xi32>
    %p = arith.muli %av, %av : i32
    %ns = arith.addi %s, %p : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %ns : i32, i32
  }
  return %res#1 : i32
}
