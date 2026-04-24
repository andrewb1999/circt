// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// A top function that calls vadd once. After SCFToLoopSchedule, both funcs
// become loopschedule.func_sequential and the func.call is rewritten to
// loopschedule.call.

// CHECK-LABEL: loopschedule.func_sequential @vadd
func.func @vadd(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) {
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

// CHECK-LABEL: loopschedule.func_sequential @top
// CHECK-SAME: (%[[A:.+]]: memref<16xi32>, %[[B:.+]]: memref<16xi32>, %[[C:.+]]: memref<16xi32>)
// Dynamic-latency: the call is wrapped in frame -> at -> launch producing a
// handle, and a subsequent frame awaits that handle before the func returns.
// CHECK: %[[H:.+]] = loopschedule.frame -> (!loopschedule.handle)
// CHECK: loopschedule.at 0 -> !loopschedule.handle
// CHECK: loopschedule.launch : !loopschedule.handle
// CHECK: loopschedule.call @vadd(%[[A]], %[[B]], %[[C]]) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
// CHECK: loopschedule.frame
// CHECK: loopschedule.await %[[H]]
// CHECK-NOT: func.call
// CHECK: loopschedule.return
func.func @top(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) attributes {top} {
  func.call @vadd(%a, %b, %c) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
  return
}
