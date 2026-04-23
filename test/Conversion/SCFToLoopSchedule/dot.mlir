// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// Reduction kernel: s = sum(A[i] * B[i]) over i, returned as a scalar.
// Currently fails in SCFToLoopSchedule because the scheduling of a sequential
// loop whose iter-arg is returned from the function does not forward the
// reduced value out of the launch region -- the `func.return` ends up using
// an SSA value defined inside the launch.
// CHECK-LABEL: loopschedule.func_sequential @dot
// CHECK: loopschedule.sequential
// CHECK: loopschedule.terminator {{.*}} results

func.func @dot(%A: memref<16xi32>, %B: memref<16xi32>) -> i32 attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  %c0s = arith.constant 0 : i32
  %res:2 = scf.while (%i = %c0, %s = %c0s) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %idx = arith.index_cast %i : i32 to index
    %av = memref.load %A[%idx] : memref<16xi32>
    %bv = memref.load %B[%idx] : memref<16xi32>
    %p = arith.muli %av, %bv : i32
    %ns = arith.addi %s, %p : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %ns : i32, i32
  }
  return %res#1 : i32
}
