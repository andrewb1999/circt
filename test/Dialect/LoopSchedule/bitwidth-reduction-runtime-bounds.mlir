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

// Past step 1 the bound is also saturated at the license: truncating an
// out-of-range size could leave the narrow IV stepping past its type's
// maximum and wrapping, counting forever. Any of bits 63..7 set (2^7 being
// the least power of 2 at or past 100) saturates it at 100; the test is the
// OR of those bits. The bound reaches 127 unsaturated, and the final update
// 126 + 4 = 130 -> 9 bits.

// CHECK-LABEL: func.func @runtime_ub_step4_saturated
// CHECK-DAG:     %[[NEG:.+]] = arith.cmpi slt, %arg2, %{{.+}} : i64
// CHECK-DAG:     %[[HI:.+]] = arith.shrui %arg2, %c7_i64 : i64
// CHECK-DAG:     %[[HIT:.+]] = arith.trunci %[[HI]] : i64 to i57
// CHECK-DAG:     %[[OVER:.+]] = arith.cmpi ne, %[[HIT]], %{{.+}} : i57
// CHECK-DAG:     %[[SAT:.+]] = arith.select %[[OVER]], %c100_i64, %arg2 : i64
// CHECK-DAG:     %[[TR:.+]] = arith.trunci %[[SAT]] : i64 to i9
// CHECK:         %[[UB:.+]] = arith.select %[[NEG]], %{{.+}}, %[[TR]] : i9
// CHECK:         scf.for %{{.+}} = %{{.+}} to %[[UB]] step %{{.+}} : i9
func.func @runtime_ub_step4_saturated(%mem: memref<100xi32>, %v: i32,
                                      %n: i64) {
  %c0 = arith.constant 0 : i64
  %c4 = arith.constant 4 : i64
  scf.for %i = %c0 to %n step %c4 : i64 {
    loopschedule.store %v, %mem[%i : i64] : memref<100xi32>
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

// A TILED loop's bound is the ceil-div chain the affine normalizer leaves
// behind: ub = ceil(n/16), spelled cmpi-sle / two subs / select / shrui /
// sub-add / select with ONE shared guard. The chain itself licenses
// nothing, but its root does: the ancestor store at %n keeps n < 1024, so
// ub <= ceil(1023/16) = 64 -> 8 bits.

// CHECK-LABEL: func.func @ceildiv_licensed
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i8
func.func @ceildiv_licensed(%mem: memref<1024xi32>, %acc: memref<8xi32>,
                            %v: i32, %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  loopschedule.store %v, %mem[%n : i64] : memref<1024xi32>
  %nonpos = arith.cmpi sle, %n, %c0 : i64
  %neg = arith.subi %c0, %n : i64
  %dec = arith.subi %n, %c1 : i64
  %dvd = arith.select %nonpos, %neg, %dec : i64
  %q = arith.shrui %dvd, %c4 : i64
  %negq = arith.subi %c0, %q : i64
  %incq = arith.addi %q, %c1 : i64
  %ub = arith.select %nonpos, %negq, %incq : i64
  scf.for %i = %c0 to %ub step %c1 : i64 {
    loopschedule.store %v, %acc[%c0 : i64] : memref<8xi32>
  }
  return
}

// REFUSED: the two selects carry DIFFERENT guards, so the chain is not the
// ceil-div shape and the root's license must not transfer. The loop narrows
// only as far as its bound's own range allows, exactly: a quotient of a
// shift by 4 is at most 2^60 -> 62 bits (not the root's 1024).

// CHECK-LABEL: func.func @ceildiv_mismatched_guard
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i62
func.func @ceildiv_mismatched_guard(%mem: memref<1024xi32>, %acc: memref<8xi32>,
                                    %v: i32, %n: i64, %m: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4 = arith.constant 4 : i64
  loopschedule.store %v, %mem[%n : i64] : memref<1024xi32>
  %nonpos = arith.cmpi sle, %n, %c0 : i64
  %other = arith.cmpi sle, %m, %c0 : i64
  %neg = arith.subi %c0, %n : i64
  %dec = arith.subi %n, %c1 : i64
  %dvd = arith.select %nonpos, %neg, %dec : i64
  %q = arith.shrui %dvd, %c4 : i64
  %negq = arith.subi %c0, %q : i64
  %incq = arith.addi %q, %c1 : i64
  %ub = arith.select %other, %negq, %incq : i64
  scf.for %i = %c0 to %ub step %c1 : i64 {
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

// A loop with BOTH bounds runtime (spmv's inner loop: k = rowptr[i] to
// rowptr[i+1]) narrows through its OWN accesses: every executed iterate
// indexes the face, so |k| is licensed below the extent — including the
// first iterate, which IS the lower bound. What the license cannot
// decide is EMPTINESS (lo >= hi executes nothing and both bounds are
// unlicensed garbage under truncation), so emptiness is decided ONCE at
// full width and the narrow ub collapses onto the narrow lb for the
// empty case: nonempty = lo <s hi; lb' = trunc(lo);
// ub' = nonempty ? trunc(hi) : lb'. This used to be a refusal; the
// zero-trip property the refusal protected is carried by the select.

// CHECK-LABEL: func.func @nonconst_lb_licensed
// CHECK:         %[[NE:.+]] = arith.cmpi slt, %arg2, %arg3 : i64
// CHECK:         %[[LBN:.+]] = arith.trunci %arg2 : i64 to i12
// CHECK:         %[[UBT:.+]] = arith.trunci %arg3 : i64 to i12
// CHECK:         %[[UBN:.+]] = arith.select %[[NE]], %[[UBT]], %[[LBN]] : i12
// CHECK:         scf.for %{{.+}} = %[[LBN]] to %[[UBN]] step %{{.+}} : i12
func.func @nonconst_lb_licensed(%mem: memref<1024xi32>, %v: i32, %lo: i64,
                                %hi: i64) {
  %c1 = arith.constant 1 : i64
  scf.for %i = %lo to %hi step %c1 : i64 {
    loopschedule.store %v, %mem[%i : i64] : memref<1024xi32>
  }
  return
}

// REFUSED: a runtime lower bound with NO access in the loop's own body —
// nothing licenses the iterate, so it stays wide (the accumulator-only
// shape; the store's index here is a CONSTANT, not the IV).

// CHECK-LABEL: func.func @nonconst_lb_unlicensed_refused
// CHECK:         scf.for %{{.+}} = %arg2 to %arg3 step %{{.+}} : i64
func.func @nonconst_lb_unlicensed_refused(%mem: memref<1024xi32>, %v: i32,
                                          %lo: i64, %hi: i64) {
  %c1 = arith.constant 1 : i64
  %c0 = arith.constant 0 : i64
  scf.for %i = %lo to %hi step %c1 : i64 {
    loopschedule.store %v, %mem[%c0 : i64] : memref<1024xi32>
  }
  return
}

// A runtime-bound INNER loop may be zero-trip, but its bound is invariant in
// the outer loop, so it runs on every outer iteration or on none. Its
// accesses license the outer IV where every write is inside it too: when it
// runs, an outer iterate past the 128 rows stores out of bounds (undefined
// behavior); when it does not, the outer loop has no effect, and how often
// it counts is unobservable. Both narrow to the rows' 9 bits.

// CHECK-LABEL: func.func @runtime_inner_credited
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @runtime_inner_credited(%A: memref<128x128xi32>, %v: i32,
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

// gemm with runtime sizes: each loop reaches memory only inside inner loops
// with invariant runtime bounds, and every write (C's) is inside the j loops.
// i is licensed by C's rows in either j loop (60 -> 7 bits), k by B's rows
// (80 -> 8 bits), the j loops by the flattened arrays (14 bits). The
// narrowed j bounds still root at %nj, so a write under the narrowed j1 and
// k loops keeps the entry conditions of the j0 access that licenses i.

// CHECK-LABEL: func.func @gemm_runtime_nest
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i7
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i14
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i8
// CHECK:             scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i14
func.func @gemm_runtime_nest(%C: memref<4200xf32>, %A: memref<4800xf32>,
                      %B: memref<5600xf32>, %ni32: i32, %nj32: i32,
                      %nk32: i32) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c70 = arith.constant 70 : i64
  %c80 = arith.constant 80 : i64
  %nk = arith.extsi %nk32 : i32 to i64
  %nj = arith.extsi %nj32 : i32 to i64
  %ni = arith.extsi %ni32 : i32 to i64
  scf.for %i = %c0 to %ni step %c1 : i64 {
    scf.for %j0 = %c0 to %nj step %c1 : i64 {
      %r = arith.muli %i, %c70 overflow<nsw> : i64
      %x = arith.addi %r, %j0 : i64
      %c = loopschedule.load %C[%x : i64] : memref<4200xf32>
      loopschedule.store %c, %C[%x : i64] : memref<4200xf32>
    }
    scf.for %k = %c0 to %nk step %c1 : i64 {
      scf.for %j1 = %c0 to %nj step %c1 : i64 {
        %r = arith.muli %i, %c70 overflow<nsw> : i64
        %x = arith.addi %r, %j1 : i64
        %c = loopschedule.load %C[%x : i64] : memref<4200xf32>
        %ra = arith.muli %i, %c80 overflow<nsw> : i64
        %xa = arith.addi %ra, %k : i64
        %a = loopschedule.load %A[%xa : i64] : memref<4800xf32>
        %rb = arith.muli %k, %c70 overflow<nsw> : i64
        %xb = arith.addi %rb, %j1 : i64
        %b = loopschedule.load %B[%xb : i64] : memref<5600xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        loopschedule.store %s, %C[%x : i64] : memref<4200xf32>
      }
    }
  }
  return
}

// An inner bound computed inside the outer body from invariant values
// (`cols - 1`, as the affine lowering emits fdtd_2d's `range(NY - 1)`) is
// invariant too: the outer IV narrows to the rows' 9 bits.

// CHECK-LABEL: func.func @runtime_inner_computed_bound
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @runtime_inner_computed_bound(%A: memref<128x128xi32>, %v: i32,
                                        %rows: i64, %cols: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    %m = arith.subi %cols, %c1 : i64
    scf.for %j = %c0 to %m step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
  }
  return
}

// fdtd_2d's time loop: t indexes only fict[t], read in a `j0 < ny` loop,
// while the writes sit in loops from 1 or to `ny - 1` (entering as `1 < ny`),
// which can only run when the j0 loop does. So t is licensed by fict's 40
// entries: 7 bits.

// CHECK-LABEL: func.func @time_loop_licensed_by_sibling
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i7
func.func @time_loop_licensed_by_sibling(%fict: memref<40xi32>,
                                         %ey: memref<64xi32>,
                                         %ex: memref<64x64xi32>, %v: i32,
                                         %tmax: i64, %nx32: i32,
                                         %ny32: i32) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %cm1 = arith.constant -1 : i64
  %nx = arith.extsi %nx32 : i32 to i64
  %ny = arith.extsi %ny32 : i32 to i64
  scf.for %t = %c0 to %tmax step %c1 : i64 {
    scf.for %j0 = %c0 to %ny step %c1 : i64 {
      %f = loopschedule.load %fict[%t : i64] : memref<40xi32>
      loopschedule.store %f, %ey[%j0 : i64] : memref<64xi32>
    }
    scf.for %i = %c0 to %nx step %c1 : i64 {
      %m = arith.addi %ny, %cm1 : i64
      scf.for %j = %c0 to %m step %c1 : i64 {
        loopschedule.store %v, %ex[%i, %j : i64, i64] : memref<64x64xi32>
      }
    }
  }
  return
}

// REFUSED: a write outside the inner loop happens on every outer iteration
// whether or not the inner loop runs, and nothing licenses the outer IV
// there, so an outer loop past the rows with an empty inner loop is a
// well-defined execution. The outer IV stays wide.

// CHECK-LABEL: func.func @runtime_inner_write_outside_refused
// CHECK:         scf.for %{{.+}} = %{{.+}} to %arg3 step %{{.+}} : i64
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @runtime_inner_write_outside_refused(%A: memref<128x128xi32>,
                                               %acc: memref<8xi32>, %v: i32,
                                               %rows: i64, %cols: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    loopschedule.store %v, %acc[%c0 : i64] : memref<8xi32>
    scf.for %j = %c0 to %cols step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
  }
  return
}

// REFUSED: the write in the second inner loop runs when IT enters, which
// the first one's (licensing) entry does not imply: with %cols empty and
// %n not, the outer loop writes on every iteration past the rows,
// well-defined. The second loop's own access does not mention the outer IV.

// CHECK-LABEL: func.func @runtime_inner_other_write_refused
// CHECK:         scf.for %{{.+}} = %{{.+}} to %arg3 step %{{.+}} : i64
func.func @runtime_inner_other_write_refused(%A: memref<128x128xi32>,
                                             %B: memref<64xi32>, %v: i32,
                                             %rows: i64, %cols: i64,
                                             %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    scf.for %j = %c0 to %cols step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
    scf.for %k = %c0 to %n step %c1 : i64 {
      loopschedule.store %v, %B[%k : i64] : memref<64xi32>
    }
  }
  return
}

// A triangular inner loop, `j < i` (or `j <= i`), is entered on every outer
// iteration from i = 1 on, so on the first one past the rows: its access
// licenses the outer IV like an unconditional one. 9 bits.

// CHECK-LABEL: func.func @runtime_inner_triangular_credited
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @runtime_inner_triangular_credited(%A: memref<128x128xi32>,
                                             %v: i32, %rows: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    scf.for %j = %c0 to %i step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
  }
  return
}

// REFUSED: an inner bound that shrinks as the outer IV grows (`j < n - i`)
// is not entered on the iterations past the rows, so its access licenses
// nothing outside it: the outer loop may count past the rows, well-defined.

// CHECK-LABEL: func.func @runtime_inner_variant_bound_refused
// CHECK:         scf.for %{{.+}} = %{{.+}} to %arg2 step %{{.+}} : i64
func.func @runtime_inner_variant_bound_refused(%A: memref<128x128xi32>,
                                               %v: i32, %rows: i64,
                                               %n: i64) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    %m = arith.subi %n, %i : i64
    scf.for %j = %c0 to %m step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
  }
  return
}

// A runtime bound with a known range narrows the loop EXACTLY, with no
// license: an int32 size sign-extended to i64 keeps the IV below 2^31, so 32
// bits hold the loop (the store's constant index licenses nothing). The
// bounds are truncated and the loop runs as before.

// CHECK-LABEL: func.func @int32_size_exact
// CHECK:         scf.for %{{.+}} = %{{.+}} to %arg2 step %{{.+}} : i32
func.func @int32_size_exact(%acc: memref<8xi32>, %v: i32, %n32: i32) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %n = arith.extsi %n32 : i32 to i64
  scf.for %i = %c0 to %n step %c1 : i64 {
    loopschedule.store %v, %acc[%c0 : i64] : memref<8xi32>
  }
  return
}

// gemm's shape with runtime sizes: the outer IV reaches memory only inside
// an inner loop with its own invariant runtime bound, which holds every
// write. The license (the 128 rows, 9 bits) is narrower than the bound's
// range (32 bits), so both loops take it.

// CHECK-LABEL: func.func @int32_sizes_nest
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
// CHECK:           scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : i9
func.func @int32_sizes_nest(%A: memref<128x128xi32>, %v: i32, %rows32: i32,
                            %cols32: i32) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %rows = arith.extsi %rows32 : i32 to i64
  %cols = arith.extsi %cols32 : i32 to i64
  scf.for %i = %c0 to %rows step %c1 : i64 {
    scf.for %j = %c0 to %cols step %c1 : i64 {
      loopschedule.store %v, %A[%i, %j : i64, i64] : memref<128x128xi32>
    }
  }
  return
}

// A runtime LOWER bound narrows by range too (ludcmp's `range(i, n)`): the
// outer IV and an int32 size bound it, 32 bits, where nothing licenses it.

// CHECK-LABEL: func.func @int32_runtime_lb
// CHECK:         scf.for %[[I:.+]] = %{{.+}} to %arg2 step %{{.+}} : i32
// CHECK:           scf.for %{{.+}} = %[[I]] to %arg2 step %{{.+}} : i32
func.func @int32_runtime_lb(%acc: memref<8xi32>, %v: i32, %n32: i32) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %n = arith.extsi %n32 : i32 to i64
  scf.for %i = %c0 to %n step %c1 : i64 {
    scf.for %j = %i to %n step %c1 : i64 {
      loopschedule.store %v, %acc[%c0 : i64] : memref<8xi32>
    }
  }
  return
}
