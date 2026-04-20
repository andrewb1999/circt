// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end outer product: C[i,j] = x[i]*y[j].
// CHECK: hw.module @outer_product
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: fsm.machine @outer_product_fsm
// CHECK: fsm.machine @loop0_fsm

func.func @outer_product(%x: memref<4xi32>, %y: memref<4xi32>, %C: memref<4x4xi32>) attributes {top} {
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
      %xv = memref.load %x[%ii] : memref<4xi32>
      %yv = memref.load %y[%jj] : memref<4xi32>
      %p = arith.muli %xv, %yv : i32
      memref.store %p, %C[%ii, %jj] : memref<4x4xi32>
      %nj = arith.addi %j, %c1 : i32
      scf.yield %nj : i32
    }
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
