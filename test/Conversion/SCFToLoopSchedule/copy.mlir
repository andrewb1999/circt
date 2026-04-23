// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// Straight memref-to-memref copy: B[i] = A[i].
// CHECK-LABEL: loopschedule.func_sequential @copy
// CHECK: loopschedule.frame -> (!loopschedule.handle)
// CHECK: loopschedule.at 0 -> !loopschedule.handle
// CHECK:   loopschedule.launch : !loopschedule.handle
// CHECK: loopschedule.sequential
// CHECK: loopschedule.load %{{.+}}[%{{.+}} : i64] : memref<16xi32>
// CHECK: loopschedule.store %{{.+}}, %{{.+}}[%{{.+}} : i64] : memref<16xi32>
// CHECK: loopschedule.await

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
