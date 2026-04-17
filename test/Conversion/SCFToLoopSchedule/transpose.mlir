// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule))" %s | FileCheck %s

// Matrix transpose: B[j,i] = A[i,j]. 2-D swapped indices on the store.
// CHECK-LABEL: func.func @transpose
// CHECK: loopschedule.sequential
// CHECK: loopschedule.launch at 0 : !loopschedule.handle
// CHECK: loopschedule.sequential
// CHECK: loopschedule.load %{{.+}}[%{{.+}}, %{{.+}} : i64, i64] : memref<4x4xi32>
// CHECK: loopschedule.store %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}} : i64, i64] : memref<4x4xi32>

func.func @transpose(%A: memref<4x4xi32>, %B: memref<4x4xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %condi = arith.cmpi slt, %i, %cN : i32
    scf.condition(%condi) %i : i32
  } do {
  ^bb0(%i: i32):
    scf.while (%j = %c0) : (i32) -> i32 {
      %condj = arith.cmpi slt, %j, %cN : i32
      scf.condition(%condj) %j : i32
    } do {
    ^bb1(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %jj = arith.index_cast %j : i32 to index
      %v = memref.load %A[%ii, %jj] : memref<4x4xi32>
      memref.store %v, %B[%jj, %ii] : memref<4x4xi32>
      %nj = arith.addi %j, %c1 : i32
      scf.yield %nj : i32
    }
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
