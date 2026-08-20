// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule,lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end transpose: 2-D nested loops, swapped store indices.
// Both loop levels inline into the single per-function machine as
// loop0_* / loop0_loop1_* states; no per-loop hw.modules remain.
// CHECK: hw.module @transpose
// CHECK: fsm.hw_instance "transpose_fsm_inst" @transpose_fsm
// CHECK: fsm.machine @transpose_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @loop0_FRAME_0
// CHECK: fsm.state @loop0_loop1_FRAME_0_0
// CHECK: fsm.state @loop0_loop1_FRAME_0_1
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0

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
