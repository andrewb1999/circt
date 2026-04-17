// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-calyx)" %s | FileCheck %s

// End-to-end saxpy: scalar mult produces a multi-cycle static_group.
// CHECK: calyx.component @saxpy
// CHECK-DAG: calyx.std_mult_pipe
// CHECK-DAG: calyx.static_group latency<4> @muli_0
// CHECK-DAG: calyx.static_group latency<1> @store_0
// CHECK: calyx.control
// CHECK: calyx.while

func.func @saxpy(%a: i32, %x: memref<16xi32>, %y: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %xv = memref.load %x[%idx] : memref<16xi32>
    %yv = memref.load %y[%idx] : memref<16xi32>
    %mul = arith.muli %a, %xv : i32
    %sum = arith.addi %mul, %yv : i32
    memref.store %sum, %y[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
