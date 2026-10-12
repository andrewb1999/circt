// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// At function level a launch opens a frame of its own. The FSM lowering runs
// a function-level frame that launches a child as ONE state and starts the
// child in it, so the frame's earlier cycles -- here the load at cycle 1
// whose result seeds the loop's accumulator -- used to run alongside the
// child, which latched its initial value before the load returned.

// CHECK-LABEL: loopschedule.func_sequential @prologue
// CHECK:         %[[PRO:.+]]:2 = loopschedule.frame -> (i32, i32) {
// CHECK:           loopschedule.at 0 -> i32 {
// CHECK:           loopschedule.at 1 -> i32 {
// CHECK-NOT:       loopschedule.launch
// CHECK:         }
// CHECK:         loopschedule.frame -> (i32, !loopschedule.handle) {
// CHECK:           %[[SEED:.+]] = loopschedule.at 0 -> i32 {
// CHECK:             arith.subi %{{.+}}, %[[PRO]]#1
// CHECK:           loopschedule.at 0 -> !loopschedule.handle {
// CHECK-NEXT:        loopschedule.launch
// CHECK-NEXT:          loopschedule.sequential iter_args(%{{.+}} = %{{.+}}, %{{.+}} = %[[SEED]])
func.func @prologue(%r: memref<4xi32>, %y: memref<4xi32>) attributes {top} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %four = arith.constant 4 : i32
  %a = memref.load %r[%c0] : memref<4xi32>
  memref.store %a, %y[%c0] : memref<4xi32>
  %b = memref.load %r[%c1] : memref<4xi32>
  %seed = arith.subi %z, %b : i32
  %res:2 = scf.while (%i = %z, %acc = %seed) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %four : i32
    scf.condition(%cond) %i, %acc : i32, i32
  } do {
  ^bb0(%i: i32, %acc: i32):
    %idx = arith.index_cast %i : i32 to index
    %v = memref.load %r[%idx] : memref<4xi32>
    %nacc = arith.addi %acc, %v : i32
    %ni = arith.addi %i, %one : i32
    scf.yield %ni, %nacc : i32, i32
  }
  memref.store %res#1, %y[%c1] : memref<4xi32>
  return
}
