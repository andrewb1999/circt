// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies))" -split-input-file %s | FileCheck %s

// A 2-D array linearized onto a 1-D memory keeps its logical map and shape
// (allocation records them): fdtd_2d's `ey[i][j] -= ...` as `ey[i*80 + j +
// 80]`, with j below a runtime bound. Linearized, row i + 1 could reach row
// i (j past 80), a dependence carried by the i loop; subscript by subscript
// (each in bounds, as memref semantics require) the accesses meet only in
// the same (i, j). (The untagged rows are the conservative memref path.)

// CHECK-LABEL: loopschedule.dependencies @rows_deps
// CHECK-NOT:     outer<
// CHECK:       func.func @rows
func.func @rows(%A: memref<4800xi32>, %n: index, %m: index) attributes {top} {
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %m {
      %x = affine.load %A[%i * 80 + %j + 80] {loopschedule.logical_map = affine_map<(d0, d1) -> (d0 + 1, d1)>, loopschedule.logical_shape = array<i64: 60, 80>} : memref<4800xi32>
      %y = arith.addi %x, %x : i32
      affine.store %y, %A[%i * 80 + %j + 80] {loopschedule.logical_map = affine_map<(d0, d1) -> (d0 + 1, d1)>, loopschedule.logical_shape = array<i64: 60, 80>} : memref<4800xi32>
    }
  }
  return
}

// -----

// Without the logical form, the same nest carries a dependence on i.

// CHECK-LABEL: loopschedule.dependencies @rows_linear_deps
// CHECK:         outer<1,
func.func @rows_linear(%A: memref<4800xi32>, %n: index, %m: index) attributes {top} {
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %m {
      %x = affine.load %A[%i * 80 + %j + 80] : memref<4800xi32>
      %y = arith.addi %x, %x : i32
      affine.store %y, %A[%i * 80 + %j + 80] : memref<4800xi32>
    }
  }
  return
}

// -----

// A logical map that no longer linearizes to the access's map (a rewrite
// since allocation changed the subscript) is ignored: the dependence stays.

// CHECK-LABEL: loopschedule.dependencies @rows_stale_deps
// CHECK:         outer<1,
func.func @rows_stale(%A: memref<4800xi32>, %n: index, %m: index) attributes {top} {
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %m {
      %x = affine.load %A[%i * 80 + %j + 80] {loopschedule.logical_map = affine_map<(d0, d1) -> (d0, d1)>, loopschedule.logical_shape = array<i64: 60, 80>} : memref<4800xi32>
      %y = arith.addi %x, %x : i32
      affine.store %y, %A[%i * 80 + %j + 80] {loopschedule.logical_map = affine_map<(d0, d1) -> (d0, d1)>, loopschedule.logical_shape = array<i64: 60, 80>} : memref<4800xi32>
    }
  }
  return
}
