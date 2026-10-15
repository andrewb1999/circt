// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies))" -split-input-file %s | FileCheck %s

// A dependence carried by an outer loop is recorded as `outer<depth, [...]>`:
// the depth of the loop carrying it, and the distance lower bounds in each
// loop below it. gemm's `for k: for j: C[j] += A[k] * B[k][j]` reads C[j]
// one k iteration after writing it, in the same j: carried by k (depth 2,
// the function's loops counted from 1), 0 apart in j. The pipeline over j
// never sees it, but a flattened k * j loop does. The i loop carries it
// too, from any k (-7 is as far back as k reaches). (The untagged dist<1>
// rows are the conservative memref path, which plain affine accesses also
// take.)

// CHECK-LABEL: loopschedule.dependencies @gemm_row_deps
// CHECK:         loopschedule.depends_on "load2" {
// CHECK-DAG:       loopschedule.access "store0" dist<1> outer<2, [0]>
// CHECK-DAG:       loopschedule.access "store0" dist<1> outer<1, [-7, 0]>
// CHECK:         loopschedule.depends_on "store0" {
// CHECK-DAG:       loopschedule.access "load2"{{$}}
// CHECK-DAG:       loopschedule.access "load2" dist<1> outer<2, [0]>
func.func @gemm_row(%A: memref<8xi32>, %B: memref<8x16xi32>, %C: memref<16xi32>) attributes {top} {
  affine.for %i = 0 to 4 {
    affine.for %k = 0 to 8 {
      affine.for %j = 0 to 16 {
        %a = affine.load %A[%k] : memref<8xi32>
        %b = affine.load %B[%k, %j] : memref<8x16xi32>
        %c = affine.load %C[%j] : memref<16xi32>
        %m = arith.muli %a, %b : i32
        %s = arith.addi %c, %m : i32
        affine.store %s, %C[%j] : memref<16xi32>
      }
    }
  }
  return
}

// -----

// A stencil reads what earlier rows and passes wrote: each carrying loop
// records the bounds below it. A[i][j - 1] is the inner loop's own
// (distance 1); A[i - 1][j + 1] was written one i earlier, one j later (-1);
// A[i + 1][j] one t earlier and one i later (-1 in i, 0 in j).

// CHECK-LABEL: loopschedule.dependencies @stencil_deps
// CHECK:         loopschedule.depends_on "load0" {
// CHECK-DAG:       loopschedule.access "store0" dist<1>{{$}}
// CHECK-DAG:       loopschedule.access "store0" dist<1> outer<1, [0, 1]>
// CHECK:         loopschedule.depends_on "load1" {
// CHECK-DAG:       loopschedule.access "store0" dist<1> outer<2, [-1]>
// CHECK-DAG:       loopschedule.access "store0" dist<1> outer<1, [1, -1]>
// CHECK:         loopschedule.depends_on "load2" {
// CHECK-DAG:       loopschedule.access "store0" dist<1> outer<1, [-1, 0]>
// CHECK:         loopschedule.depends_on "store0" {
func.func @stencil(%A: memref<10x10xi32>) attributes {top} {
  affine.for %t = 0 to 4 {
    affine.for %i = 1 to 9 {
      affine.for %j = 1 to 9 {
        %w = affine.load %A[%i, %j - 1] : memref<10x10xi32>
        %ne = affine.load %A[%i - 1, %j + 1] : memref<10x10xi32>
        %s = affine.load %A[%i + 1, %j] : memref<10x10xi32>
        %a = arith.addi %w, %ne : i32
        %b = arith.addi %a, %s : i32
        affine.store %b, %A[%i, %j] : memref<10x10xi32>
      }
    }
  }
  return
}
