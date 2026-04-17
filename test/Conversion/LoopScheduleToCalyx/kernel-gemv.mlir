// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-calyx)" %s | FileCheck %s

// End-to-end gemv. Fails in LoopScheduleToCalyx:
// `unsupported pipeline result type` assertion -- Calyx lowering does not
// currently support `!loopschedule.handle` results on a frame (nested-loop
// launch handle forwarded via the outer frame).
// CHECK: calyx.component @gemv

func.func @gemv(%A: memref<4x4xi32>, %x: memref<4xi32>, %y: memref<4xi32>) attributes {top} {
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
      %av = memref.load %A[%ii, %jj] : memref<4x4xi32>
      %xv = memref.load %x[%jj] : memref<4xi32>
      %yv = memref.load %y[%ii] : memref<4xi32>
      %p = arith.muli %av, %xv : i32
      %ns = arith.addi %yv, %p : i32
      memref.store %ns, %y[%ii] : memref<4xi32>
      %nj = arith.addi %j, %c1 : i32
      scf.yield %nj : i32
    }
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
