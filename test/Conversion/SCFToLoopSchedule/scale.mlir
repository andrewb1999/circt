// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// Scalar-memref product: B[i] = a * A[i]. Multi-cycle muli.
// CHECK-LABEL: loopschedule.func_sequential @scale
// CHECK: loopschedule.frame -> (!loopschedule.handle)
// CHECK: loopschedule.at 0 -> !loopschedule.handle
// CHECK:   loopschedule.launch : !loopschedule.handle
// CHECK-DAG: loopschedule.at 0
// CHECK-DAG: loopschedule.at 1
// CHECK: arith.muli {{.*}} {loopschedule.cycle_latency
// CHECK: loopschedule.store

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
