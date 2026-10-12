// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// An `if` without an `else`, followed by a loop, in a loop body whose `if`
// condition is computed outside it (`%low`, from the outer loop). This used
// to assert three ways: the resource walk only closed an `if` at its else
// block's yield, so the loop after an else-less `if` counted as inside it
// ("Loops inside if statements is unsupported"); predicate dependences
// were added for every predicated op the pass had seen, including those of
// loops scheduled earlier; and a condition from outside the loop has no
// producer in its problem. The store under the `if` is ordered before the
// inner loop, which reads what it wrote.

// CHECK-LABEL: loopschedule.func_sequential @if_then_loop
// CHECK:         loopschedule.at {{[0-9]+}} {
// CHECK:           loopschedule.if %{{[^ ]+}} {
// CHECK-NEXT:        loopschedule.store
// CHECK:         loopschedule.at {{[0-9]+}} -> !loopschedule.handle {
// CHECK-NEXT:      loopschedule.launch
func.func @if_then_loop(%A: memref<16xi32>) attributes {top} {
  %z = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %four = arith.constant 4 : i32
  %ro:1 = scf.while (%i = %z) : (i32) -> (i32) {
    %ci = arith.cmpi slt, %i, %four : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %low = arith.cmpi slt, %i, %two : i32
    %row = arith.muli %i, %four : i32
    %rj:1 = scf.while (%j = %z) : (i32) -> (i32) {
      %cj = arith.cmpi slt, %j, %four : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %e = arith.addi %row, %j : i32
      %ei = arith.index_cast %e : i32 to index
      scf.if %low {
        %v = memref.load %A[%ei] : memref<16xi32>
        %w = arith.addi %v, %one : i32
        memref.store %w, %A[%ei] : memref<16xi32>
      }
      %rk:1 = scf.while (%k = %z) : (i32) -> (i32) {
        %ck = arith.cmpi slt, %k, %j : i32
        scf.condition(%ck) %k : i32
      } do {
      ^bb0(%k: i32):
        %f = arith.addi %row, %k : i32
        %fi = arith.index_cast %f : i32 to index
        %a = memref.load %A[%fi] : memref<16xi32>
        %b = memref.load %A[%ei] : memref<16xi32>
        %s = arith.addi %a, %b : i32
        memref.store %s, %A[%ei] : memref<16xi32>
        %nk = arith.addi %k, %one : i32
        scf.yield %nk : i32
      }
      %nj = arith.addi %j, %one : i32
      scf.yield %nj : i32
    }
    %ni = arith.addi %i, %one : i32
    scf.yield %ni : i32
  }
  return
}
