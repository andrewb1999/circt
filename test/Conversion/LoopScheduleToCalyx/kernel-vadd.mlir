// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-calyx)" %s | FileCheck %s

// End-to-end vector add through Calyx.
// CHECK: calyx.component @vadd
// CHECK: calyx.seq_mem @ext_mem_0
// CHECK-DAG: calyx.static_group latency<1> @load_0
// CHECK-DAG: calyx.static_group latency<1> @store_0
// CHECK: calyx.control
// CHECK: calyx.while

func.func @vadd(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) attributes {top} {
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
