// RUN: circt-opt -split-input-file --flatten-memref --canonicalize --scf-while-loop-flattening %s | FileCheck %s

// End-to-end integration test for the linearized-address promotion feature.
// Runs FlattenMemRefs (which rewrites multi-dim memref accesses to a 1D
// memref + arith-chain index) followed by canonicalization and
// SCFWhileLoopFlattening (which should recognize the arith chain as a linear
// combination of the loop IVs and promote it to an iter-arg that updates
// every iteration via select-driven addi's — no muli/shli in the inner body).

// -----

// 2D copy: load src[i,j], store dst[i,j]. Both addresses are i*8+j.
// Expect: flattened single scf.while, no arith.muli/shli in after region,
// memref.load/store indices come directly from iter-args.

// CHECK-LABEL: func.func @copy2d
// CHECK-SAME:    (%[[SRC:.*]]: memref<32xi32>, %[[DST:.*]]: memref<32xi32>)
// CHECK:         scf.while
// CHECK:         ^bb0(
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK-NOT:       arith.divsi
// CHECK-NOT:       arith.remsi
// CHECK:           memref.load %[[SRC]][%{{.*}}] : memref<32xi32>
// CHECK:           memref.store %{{.*}}, %[[DST]][%{{.*}}] : memref<32xi32>
// CHECK:           scf.yield
func.func @copy2d(%src: memref<4x8xi32>, %dst: memref<4x8xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c8 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %v = memref.load %src[%ii, %ji] : memref<4x8xi32>
      memref.store %v, %dst[%ii, %ji] : memref<4x8xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// 3-deep matmul-shape nest. Three distinct linear address forms (A[i,k],
// B[k,j], C[i,j]) should all be promoted. Inner body must contain no
// arith.muli/shli for the address chains — only the arith.muli for the
// multiply-accumulate on the payload.

// CHECK-LABEL: func.func @matmul_addr
// CHECK-SAME:    (%[[A:.*]]: memref<32xi32>, %[[B:.*]]: memref<32xi32>, %[[C:.*]]: memref<16xi32>)
// CHECK:         scf.while
// CHECK:         ^bb0(
// CHECK:           %[[AV:.*]] = memref.load %[[A]][%{{.*}}] : memref<32xi32>
// CHECK:           %[[BV:.*]] = memref.load %[[B]][%{{.*}}] : memref<32xi32>
// CHECK:           %[[CV:.*]] = memref.load %[[C]][%{{.*}}] : memref<16xi32>
// CHECK:           arith.muli %[[AV]], %[[BV]] : i32
// CHECK:           memref.store %{{.*}}, %[[C]][%{{.*}}] : memref<16xi32>
// CHECK-NOT:       arith.shli
// CHECK:           scf.yield
func.func @matmul_addr(%A: memref<4x8xi32>, %B: memref<8x4xi32>, %C: memref<4x4xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      scf.while (%k = %c0) : (i32) -> i32 {
        %ck = arith.cmpi slt, %k, %c8 : i32
        scf.condition(%ck) %k : i32
      } do {
      ^bb0(%k: i32):
        %ii = arith.index_cast %i : i32 to index
        %ji = arith.index_cast %j : i32 to index
        %ki = arith.index_cast %k : i32 to index
        %a = memref.load %A[%ii, %ki] : memref<4x8xi32>
        %b = memref.load %B[%ki, %ji] : memref<8x4xi32>
        %c = memref.load %C[%ii, %ji] : memref<4x4xi32>
        %ab = arith.muli %a, %b : i32
        %cc = arith.addi %c, %ab : i32
        memref.store %cc, %C[%ii, %ji] : memref<4x4xi32>
        %kn = arith.addi %k, %c1 : i32
        scf.yield %kn : i32
      }
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// 2D stencil read: output[i,j] = input[i,j] + input[i,j+1]. Two distinct
// linear address forms (offset 0 and offset +1 into the same row). Both
// should be promoted; the address arith chains should be gone from the body.

// CHECK-LABEL: func.func @stencil2d
// CHECK-SAME:    (%[[IN:.*]]: memref<32xi32>, %[[OUT:.*]]: memref<32xi32>)
// CHECK:         scf.while
// CHECK:         ^bb0(
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK:           memref.load %[[IN]][%{{.*}}] : memref<32xi32>
// CHECK:           memref.load %[[IN]][%{{.*}}] : memref<32xi32>
// CHECK:           memref.store %{{.*}}, %[[OUT]][%{{.*}}] : memref<32xi32>
// CHECK:           scf.yield
func.func @stencil2d(%input: memref<4x8xi32>, %output: memref<4x8xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c1idx = arith.constant 1 : index
  %c4 = arith.constant 4 : i32
  %c7 = arith.constant 7 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c7 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %jip1 = arith.addi %ji, %c1idx : index
      %a = memref.load %input[%ii, %ji] : memref<4x8xi32>
      %b = memref.load %input[%ii, %jip1] : memref<4x8xi32>
      %s = arith.addi %a, %b : i32
      memref.store %s, %output[%ii, %ji] : memref<4x8xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}
