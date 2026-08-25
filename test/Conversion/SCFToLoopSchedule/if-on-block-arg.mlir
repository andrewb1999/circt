// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// An `if` whose condition is a bool function argument. The predicate is a
// block argument -- there is no producer to add a scheduling dependence
// on; it is stable from cycle 0 -- and addPredicateDependencies used to
// assert on its missing defining op.

// CHECK-LABEL: loopschedule.func_sequential @if_on_arg
// CHECK-SAME:    (%[[X:.*]]: i32, %[[ROT:.*]]: i1)
// CHECK:         loopschedule.sequential
// CHECK:         arith.select %[[ROT]]
func.func @if_on_arg(%x: i32, %rot: i1) -> i32 attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 8 : i32
  %c1 = arith.constant 1 : i32
  %res:2 = scf.while (%i = %c0, %v = %x) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i, %v : i32, i32
  } do {
  ^bb0(%i: i32, %v: i32):
    %nv = scf.if %rot -> i32 {
      %a = arith.addi %v, %c1 : i32
      scf.yield %a : i32
    } else {
      %s = arith.subi %v, %c1 : i32
      scf.yield %s : i32
    }
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %nv : i32, i32
  }
  return %res#1 : i32
}
