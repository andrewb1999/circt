// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// An iter arg whose next value is a LOAD that also feeds a reader of the iter
// arg's current value (deriche's `xm1 = imgIn[i][j]`, read beside the
// `a2 * xm1` it replaces once the two loads of imgIn[i][j] are one):
//
//   %a = load A[i];  %s = %a + %p1;  ...;  yield ..., %a   // p1 = A[i]
//
// The update must wait for its readers (`%s` reads the OLD p1), and `%s`
// waits a cycle for the load: with the update at the load, a cycle in the
// schedule, which was declared infeasible. A load is not duplicated, as a
// pure producer is; the yield takes a same-type bitcast of it, a wire,
// whose update is placed after the reader, the load held in a register
// meanwhile.

// CHECK-LABEL: loopschedule.func_sequential @load_feeds_reader
// CHECK:         loopschedule.sequential iter_args(%{{[a-z0-9_]+}} = %{{[a-z0-9_]+}}, %[[P1:[a-z0-9_]+]] = %{{[a-z0-9_]+}})
// CHECK:           %[[AT0:[0-9]+]]:4 = loopschedule.at 0
// CHECK:             loopschedule.load %arg0
// CHECK:           loopschedule.at 1
// CHECK-NEXT:        arith.addi %[[AT0]]#2, %[[P1]] : i32
// CHECK-NEXT:        %[[COPY:[0-9]+]] = arith.bitcast %[[AT0]]#2 : i32 to i32
// CHECK-NEXT:        loopschedule.iter_arg_update %[[P1]] = %[[COPY]] : i32
func.func @load_feeds_reader(%A: memref<8xi32>, %B: memref<8xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %res:2 = scf.while (%i = %c0, %p1 = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c8 : i32
    scf.condition(%cond) %i, %p1 : i32, i32
  } do {
  ^bb0(%i: i32, %p1: i32):
    %idx = arith.index_cast %i : i32 to index
    %a = memref.load %A[%idx] : memref<8xi32>
    %s = arith.addi %a, %p1 : i32
    memref.store %s, %B[%idx] : memref<8xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %a : i32, i32
  }
  return
}
