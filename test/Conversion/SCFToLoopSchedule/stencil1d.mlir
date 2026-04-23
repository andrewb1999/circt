// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// 1-D 3-tap stencil: B[i] = (A[i-1] + A[i] + A[i+1]) >> 2. Multi-cycle reads
// from the same memref at three neighboring indices.
// CHECK-LABEL: loopschedule.func_sequential @stencil1d
// CHECK: loopschedule.sequential
// CHECK-DAG: loopschedule.at 0
// CHECK-DAG: loopschedule.at 1
// CHECK-DAG: loopschedule.at 2
// CHECK: arith.shrsi
// CHECK: loopschedule.store

func.func @stencil1d(%A: memref<16xi32>, %B: memref<16xi32>) attributes {top} {
  %c1 = arith.constant 1 : i32
  %cNm1 = arith.constant 15 : i32
  %c2 = arith.constant 2 : i32
  scf.while (%i = %c1) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cNm1 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %im1 = arith.subi %i, %c1 : i32
    %ip1 = arith.addi %i, %c1 : i32
    %im1i = arith.index_cast %im1 : i32 to index
    %ii = arith.index_cast %i : i32 to index
    %ip1i = arith.index_cast %ip1 : i32 to index
    %a0 = memref.load %A[%im1i] : memref<16xi32>
    %a1 = memref.load %A[%ii] : memref<16xi32>
    %a2 = memref.load %A[%ip1i] : memref<16xi32>
    %s0 = arith.addi %a0, %a1 : i32
    %s1 = arith.addi %s0, %a2 : i32
    %avg = arith.shrsi %s1, %c2 : i32
    memref.store %avg, %B[%ii] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
