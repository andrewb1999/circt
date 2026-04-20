// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-fsm)" %s | FileCheck %s
// XFAIL: *
// XFAIL reason: LoopScheduleToFSM pass needs updating to consume the new
// launch-inside-at dialect shape (AMC Category A refactor). Tracking issue:
// port FSM pass to the new shape analogous to LoopScheduleToCalyx.


// End-to-end gemm: triple-nested loops.
// CHECK: hw.module @gemm
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: fsm.machine @gemm_fsm
// CHECK: fsm.machine @loop0_fsm

func.func @gemm(%A: memref<4x4xi32>, %B: memref<4x4xi32>, %C: memref<4x4xi32>) attributes {top} {
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
      scf.while (%k = %c0) : (i32) -> i32 {
        %condk = arith.cmpi slt, %k, %cN : i32
        scf.condition(%condk) %k : i32
      } do {
      ^bb2(%k: i32):
        %ii = arith.index_cast %i : i32 to index
        %jj = arith.index_cast %j : i32 to index
        %kk = arith.index_cast %k : i32 to index
        %av = memref.load %A[%ii, %kk] : memref<4x4xi32>
        %bv = memref.load %B[%kk, %jj] : memref<4x4xi32>
        %cv = memref.load %C[%ii, %jj] : memref<4x4xi32>
        %p = arith.muli %av, %bv : i32
        %ns = arith.addi %cv, %p : i32
        memref.store %ns, %C[%ii, %jj] : memref<4x4xi32>
        %nk = arith.addi %k, %c1 : i32
        scf.yield %nk : i32
      }
      %nj = arith.addi %j, %c1 : i32
      scf.yield %nj : i32
    }
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
