// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-calyx)" %s | FileCheck %s

// XFAIL: *
// End-to-end 1-D convolution. Fails in BuildIntermediateRegs with
// `assertion phaseRegs[phase].count(idx) == 0` — a phase register for an
// accumulator iter-arg is added twice because the outer loop reinitializes
// the inner accumulator each iteration. Distinct from the frame
// dynamic-child issue already fixed for gemv/gemm/transpose/outer_product;
// needs targeted handling of reset-each-outer-iteration accumulators.
// CHECK: calyx.component @conv1d

func.func @conv1d(%A: memref<16xi32>, %K: memref<4xi32>, %B: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 12 : i32
  %cK = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %condi = arith.cmpi slt, %i, %cN : i32
    scf.condition(%condi) %i : i32
  } do {
  ^bb0(%i: i32):
    %c0s = arith.constant 0 : i32
    %res:2 = scf.while (%j = %c0, %s = %c0s) : (i32, i32) -> (i32, i32) {
      %condj = arith.cmpi slt, %j, %cK : i32
      scf.condition(%condj) %j, %s : i32, i32
    } do {
    ^bb1(%j: i32, %s: i32):
      %ipj = arith.addi %i, %j : i32
      %ipji = arith.index_cast %ipj : i32 to index
      %jj = arith.index_cast %j : i32 to index
      %av = memref.load %A[%ipji] : memref<16xi32>
      %kv = memref.load %K[%jj] : memref<4xi32>
      %p = arith.muli %av, %kv : i32
      %ns = arith.addi %s, %p : i32
      %nj = arith.addi %j, %c1 : i32
      scf.yield %nj, %ns : i32, i32
    }
    %ii = arith.index_cast %i : i32 to index
    memref.store %res#1, %B[%ii] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
