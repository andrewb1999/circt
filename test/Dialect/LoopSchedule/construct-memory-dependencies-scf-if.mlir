// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies))" %s | FileCheck %s

// Affine dependence analysis does not model an access under an `scf.if`: it
// FAILS on such a pair, which proves nothing. The pass used to read that as
// "no dependence", so the store under the `if` and the read-modify-write of
// the same element after it were scheduled in either order. A failure now
// counts as a dependence: at distance 1 across iterations, and within one
// iteration only from the access that comes first.

// CHECK-LABEL: loopschedule.dependencies @if_then_rmw_deps
// CHECK:         loopschedule.depends_on "load0" {
// CHECK-NEXT:      loopschedule.access "store0" dist<1>
// CHECK-NEXT:      loopschedule.access "store1" dist<1>
// CHECK-NEXT:    }
// CHECK:         loopschedule.depends_on "load1" {
// CHECK-NEXT:      loopschedule.access "store0"{{$}}
// CHECK:         loopschedule.depends_on "store1" {
// CHECK-DAG:       loopschedule.access "load0"{{$}}
// CHECK-DAG:       loopschedule.access "store0"{{$}}
func.func @if_then_rmw(%A: memref<4xi32>) attributes {top} {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : index
  affine.for %j = 0 to 4 {
    %small = arith.cmpi slt, %j, %c2 : index
    scf.if %small {
      %a = affine.load %A[%j] : memref<4xi32>
      %b = arith.addi %a, %c1 : i32
      affine.store %b, %A[%j] : memref<4xi32>
    }
    %c = affine.load %A[%j] : memref<4xi32>
    %d = arith.addi %c, %c : i32
    affine.store %d, %A[%j] : memref<4xi32>
  }
  return
}
