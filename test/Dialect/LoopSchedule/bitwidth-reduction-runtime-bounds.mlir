// RUN: circt-opt --bitwidth-reduction-for-loopschedule %s | FileCheck %s

// A loop with a RUNTIME upper bound narrows when its IV unconditionally
// indexes a memref every iteration: an iterate at or past the dimension size
// would execute an out-of-bounds access (undefined behavior), so every
// well-defined execution keeps the IV below the dimension. dim 1024, step 1
// -> signed width holding 1024 = 12 bits. The bound is narrowed under a sign
// guard so a negative bound stays a zero-trip loop.

// CHECK-LABEL: func.func @runtime_ub_direct
// CHECK-DAG:     %[[NEG:.+]] = arith.cmpi slt, %arg2, %{{.+}} : i64
// CHECK-DAG:     %[[TR:.+]] = arith.trunci %arg2 : i64 to i12
// CHECK:         %[[UB:.+]] = arith.select %[[NEG]], %{{.+}}, %[[TR]] : i12
// CHECK:         scf.for %{{.+}} = %{{.+}} to %[[UB]] step %{{.+}} : i12
// CHECK:           loopschedule.store %arg1, %arg0[%{{.+}} : i11] : memref<1024xi32>
func.func @runtime_ub_direct(%mem: memref<1024xi32>, %v: i32, %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %n step %c1 : i64 {
    loopschedule.store %v, %mem[%i : i64] : memref<1024xi32>
  }
  return
}

// A tiled shape: the outer IV reaches memory only as `i0 + i1` under a
// constant-trip inner loop. `i1` is provably non-negative, so the access
// licenses the OUTER loop too. dim 128, step 16 -> holds 143 -> 9 bits.

// CHECK-LABEL: func.func @addi_license
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i5
func.func @addi_license(%C: memref<128x128xi32>, %v: i32, %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64
  %c16 = arith.constant 16 : i64
  scf.for %i0 = %c0 to %n step %c16 : i64 {
    scf.for %i1 = %c0 to %c8 step %c1 : i64 {
      %idx = arith.addi %i0, %i1 : i64
      loopschedule.store %v, %C[%idx, %i1 : i64, i64] : memref<128x128xi32>
    }
  }
  return
}

// The post-normalization tiled shape: a unit-step tile counter whose IV
// reaches memory only scaled and linearized — `(t*16 + i1)*128 + k` spelled
// with shifts over a flat memref<16384>. The licensing store (matmul's C
// store) sits in the constant-trip i1 body, OUTSIDE the runtime k loop: an
// access inside the k loop licenses only k itself (k may zero-trip). The
// tile counter's scale is 2048 -> bound (16384-1)/2048+1 = 8 -> i5; k's
// load at scale 128 (B's row stride) bounds it at 128 -> i9.

// CHECK-LABEL: func.func @scaled_license
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i5
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i6
// CHECK:             scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @scaled_license(%A: memref<16384xi32>, %v: i32, %nt: i64, %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %c7 = arith.constant 7 : i64
  %c16 = arith.constant 16 : i64
  scf.for %t = %c0 to %nt step %c1 : i64 {
    scf.for %i1 = %c0 to %c16 step %c1 : i64 {
      %i0 = arith.shli %t, %c4 : i64
      %row = arith.addi %i0, %i1 : i64
      %base = arith.shli %row, %c7 : i64
      scf.for %k = %c0 to %n step %c1 : i64 {
        %krow = arith.shli %k, %c7 : i64
        %kidx = arith.addi %krow, %i1 : i64
        loopschedule.store %v, %A[%kidx : i64] : memref<16384xi32>
      }
      loopschedule.store %v, %A[%base : i64] : memref<16384xi32>
    }
  }
  return
}

// Two stacked runtime-bound loops sharing ONE bound value (matmul's square
// tile counters): if the outer enters, lb < ub held, so the same-bound inner
// (lb 0) enters too — its body is guaranteed and licenses the outer. Outer
// scale 2048 -> bound 8 -> i5; inner scale 16 -> bound 1024 -> i12. The
// match must survive the inner narrowing first (bottom-up), i.e. look
// through the select rewrite.

// CHECK-LABEL: func.func @same_bound_nest
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i5
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i12
func.func @same_bound_nest(%A: memref<16384xi32>, %v: i32, %nt: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  %c11 = arith.constant 11 : i64
  scf.for %t = %c0 to %nt step %c1 : i64 {
    scf.for %j = %c0 to %nt step %c1 : i64 {
      %hi = arith.shli %t, %c11 : i64
      %lo = arith.shli %j, %c4 : i64
      %idx = arith.addi %hi, %lo : i64
      loopschedule.store %v, %A[%idx : i64] : memref<16384xi32>
    }
  }
  return
}

// A loop whose IV never reaches memory still narrows when its BOUND does:
// an access indexed by `%n` in an ANCESTOR block executes whenever the loop
// does, so every well-defined execution keeps n < 1024 — the AXI stream
// shape, where a hoisted burst request carries the trip count as a length.
// n <= 1023 with step 1 -> 11 bits. The body's constant-indexed store
// licenses nothing itself.

// CHECK-LABEL: func.func @ub_licensed
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i11
func.func @ub_licensed(%mem: memref<1024xi32>, %acc: memref<8xi32>, %v: i32,
                       %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  loopschedule.store %v, %mem[%n : i64] : memref<1024xi32>
  scf.for %i = %c0 to %n step %c1 : i64 {
    loopschedule.store %v, %acc[%c0 : i64] : memref<8xi32>
  }
  return
}

// REFUSED: the only access sits under an scf.if, so it may not execute and
// licenses nothing. The loop must stay at its original width.

// CHECK-LABEL: func.func @under_if_refused
// CHECK:         scf.for %{{.+}} = %{{.+}} to %arg2 step %{{.+}} : i64
func.func @under_if_refused(%mem: memref<1024xi32>, %v: i32, %n: i64, %p: i1) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %n step %c1 : i64 {
    scf.if %p {
      loopschedule.store %v, %mem[%i : i64] : memref<1024xi32>
    }
  }
  return
}

// REFUSED: a runtime LOWER bound cannot preserve the zero-trip compare
// (lo >= hi is well-defined and carries no UB license).

// CHECK-LABEL: func.func @nonconst_lb_refused
// CHECK:         scf.for %{{.+}} = %arg2 to %arg3 step %{{.+}} : i64
func.func @nonconst_lb_refused(%mem: memref<1024xi32>, %v: i32, %lo: i64,
                               %hi: i64) {
  %c1 = arith.constant 1 : i64
  scf.for %i = %lo to %hi step %c1 : i64 {
    loopschedule.store %v, %mem[%i : i64] : memref<1024xi32>
  }
  return
}

// A runtime-bound INNER loop is not guaranteed to execute (it may be
// zero-trip), so its accesses do not license the outer loop — the outer IV
// stays wide. The inner loop licenses itself and narrows.

// CHECK-LABEL: func.func @runtime_inner_not_credited
// CHECK:         scf.for %{{.+}} = %{{.+}} to %arg2 step %{{.+}} : i64
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @runtime_inner_not_credited(%A: memref<128x128xi32>, %v: i32,
                                      %rows: i64, %cols: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    scf.for %j = %c0 to %cols step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
  }
  return
}
