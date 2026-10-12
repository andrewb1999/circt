// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// Iter args whose next values come from the iteration's edges:
//
// * `p2` takes the OLD `p1` (`scf.yield ..., %p1`). Its update forwards that
//   iter arg; it used to keep `p2` (`iter_arg_update %p2 = %p2`). Nor is the
//   terminator's read of `p1` a hazard on `p1`'s next value: every iter-arg
//   register updates on the same edge. Ordering the terminator before that
//   value's producer, as such a read was, is a cycle (the schedule was
//   declared infeasible).
// * `p1` takes a load issued in the iteration's last cycle, whose data is
//   ready only in the next: that cycle gets an `at` that re-yields the load
//   for the update, instead of the update latching the port's previous read.

// CHECK-LABEL: loopschedule.func_sequential @forward
// CHECK:         loopschedule.sequential iter_args(%[[I:.+]] = %{{.+}}, %[[P1:.+]] = %{{.+}}, %[[P2:.+]] = %{{.+}})
// CHECK:           loopschedule.at 0
// CHECK:             loopschedule.iter_arg_update %[[P2]] = %[[P1]] : i32
// CHECK:           %[[RELOAD:.+]] = loopschedule.at 3 -> i32 {
// CHECK-NEXT:        loopschedule.load %arg1
// CHECK:           loopschedule.at 4 -> i32 {
// CHECK-NEXT:        loopschedule.iter_arg_update %[[P1]] = %[[RELOAD]] : i32
func.func @forward(%A: memref<8xi32>, %B: memref<8xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %res:3 = scf.while (%i = %c0, %p1 = %c0, %p2 = %c0) : (i32, i32, i32) -> (i32, i32, i32) {
    %cond = arith.cmpi slt, %i, %c8 : i32
    scf.condition(%cond) %i, %p1, %p2 : i32, i32, i32
  } do {
  ^bb0(%i: i32, %p1: i32, %p2: i32):
    %idx = arith.index_cast %i : i32 to index
    %a = memref.load %A[%idx] : memref<8xi32>
    %s = arith.addi %a, %p1 : i32
    %t = arith.subi %s, %p2 : i32
    memref.store %t, %B[%idx] : memref<8xi32>
    %x = memref.load %B[%idx] : memref<8xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %x, %p1 : i32, i32, i32
  }
  return
}
