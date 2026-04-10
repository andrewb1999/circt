// RUN: circt-opt -split-input-file --scf-while-loop-flattening %s | FileCheck %s

// Positive: depth-2, unit steps, slt predicates. Body uses both IVs.

// CHECK-LABEL: func.func @depth2_unit
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK:         scf.while (%[[I:.*]] = %{{.*}}, %[[J:.*]] = %{{.*}}) : (i32, i32) -> (i32, i32) {
// CHECK:           arith.cmpi slt, %[[I]], %{{.*}} : i32
// CHECK:           scf.condition(%{{.*}}) %[[I]], %[[J]] : i32, i32
// CHECK:         } do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32):
// CHECK:           arith.addi %[[BI]], %[[BJ]] : i32
// CHECK:           %[[JN:.*]] = arith.addi %[[BJ]], %{{.*}} : i32
// CHECK:           %[[DJ:.*]] = arith.cmpi sge, %[[JN]], %{{.*}} : i32
// CHECK:           %[[JOUT:.*]] = arith.select %[[DJ]], %{{.*}}, %[[JN]] : i32
// CHECK:           %[[IN:.*]] = arith.addi %[[BI]], %{{.*}} : i32
// CHECK:           %[[IOUT:.*]] = arith.select %[[DJ]], %[[IN]], %[[BI]] : i32
// CHECK:           scf.yield %[[IOUT]], %[[JOUT]] : i32, i32
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @depth2_unit(%arg: memref<12xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %inner = scf.while (%j = %c0) : (i32) -> i32 {
      %cond2 = arith.cmpi slt, %j, %c3 : i32
      scf.condition(%cond2) %j : i32
    } do {
    ^bb0(%j: i32):
      %use = arith.addi %i, %j : i32
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Positive: depth-3 unit steps. Expect three iter-args and two done signals.

// CHECK-LABEL: func.func @depth3_unit
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32, i32) -> (i32, i32, i32)
// CHECK:         do {
// CHECK:         ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
// CHECK:           arith.cmpi sge
// CHECK:           arith.select
// CHECK:           arith.cmpi sge
// CHECK:           arith.andi
// CHECK:           arith.select
// CHECK:           arith.select
// CHECK:           scf.yield %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @depth3_unit() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c3 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %rk = scf.while (%k = %c0) : (i32) -> i32 {
        %ck = arith.cmpi slt, %k, %c2 : i32
        scf.condition(%ck) %k : i32
      } do {
      ^bb0(%k: i32):
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

// Positive: non-unit inner step. Inner 0..8 step 2 -> trip count 4.

// CHECK-LABEL: func.func @nonunit_inner_step
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32) -> (i32, i32)
// CHECK:         do {
// CHECK:         ^bb0(%[[I:.*]]: i32, %[[J:.*]]: i32):
// CHECK:           %[[STEP:.*]] = arith.constant 2 : i32
// CHECK:           %[[JN:.*]] = arith.addi %[[J]], %[[STEP]] : i32
// CHECK:           %[[DJ:.*]] = arith.cmpi sge, %[[JN]], %{{.*}} : i32
// CHECK:           arith.select %[[DJ]]
// CHECK-NOT:     scf.while
func.func @nonunit_inner_step() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c8 = arith.constant 8 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c8 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c2 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Positive: `ne` predicate at the inner level. Wrap check should use `eq`.

// CHECK-LABEL: func.func @ne_inner
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32) -> (i32, i32)
// CHECK:         do {
// CHECK:           arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @ne_inner() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi ne, %j, %c3 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Positive: `sle` predicate at the inner level. Wrap check should use `sge`
// against the normalized exclusive upper bound (inner ub 3 inclusive -> 4).

// CHECK-LABEL: func.func @sle_inner
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32) -> (i32, i32)
// CHECK:         do {
// CHECK:           %[[UB:.*]] = arith.constant 4 : i32
// CHECK:           arith.cmpi sge, %{{.*}}, %[[UB]] : i32
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @sle_inner() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi sle, %j, %c3 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Negative: depth 1. Leaves IR unchanged (the nest must be ≥ 2 levels).

// CHECK-LABEL: func.func @depth1_noop
// CHECK:         scf.while
// CHECK:         do {
// CHECK-NOT:       scf.while
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
func.func @depth1_noop() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Negative: inner upper bound is not a constant (comes from function arg).

// CHECK-LABEL: func.func @noncanonical_inner
// CHECK:         scf.while
// CHECK:         do {
// CHECK:           scf.while
// CHECK:         }
func.func @noncanonical_inner(%ubDyn: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %ubDyn : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Negative: extra op in the outer after region (a stray addi that isn't the
// IV update). The intermediate-level shape check rejects this.

// CHECK-LABEL: func.func @outer_extra_op
// CHECK:         scf.while
// CHECK:         do {
// CHECK:           scf.while
// CHECK:         }
func.func @outer_extra_op(%arg: memref<4xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c3 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %stray = arith.addi %i, %i : i32
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Negative: outer yield IV operand is the inner while result instead of the
// addi update. The pass should leave this alone.

// CHECK-LABEL: func.func @outer_yield_from_inner
// CHECK:         scf.while
// CHECK:         do {
// CHECK:           scf.while
// CHECK:         }
func.func @outer_yield_from_inner() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c3 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    scf.yield %rj : i32
  }
  return
}

// -----

// Negative: zero trip count at a level (lb == ub). Canonicalization handles
// zero-trip loops; the flattening pass bails.

// CHECK-LABEL: func.func @zero_trip_inner
// CHECK:         scf.while
// CHECK:         do {
// CHECK:           scf.while
// CHECK:         }
func.func @zero_trip_inner() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c4 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c0 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Positive: 6-deep perfect nest modelling a 2D convolution
// (oc x ic x oh x ow x kh x kw). All levels have constant slt bounds with
// unit step. The body loads input and weight, multiplies, accumulates
// into output. Checks that the entire nest collapses to a single scf.while
// with six iter-args, that the cloned body still references all six IVs,
// and that five cascading done signals (one cmpi per non-outer level
// plus four andi reductions) drive the odometer.

// CHECK-LABEL: func.func @conv2d
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK:         scf.while (%[[OC:.*]] = %{{.*}}, %[[IC:.*]] = %{{.*}}, %[[OH:.*]] = %{{.*}}, %[[OW:.*]] = %{{.*}}, %[[KH:.*]] = %{{.*}}, %[[KW:.*]] = %{{.*}}) : (i32, i32, i32, i32, i32, i32) -> (i32, i32, i32, i32, i32, i32) {
// CHECK:           arith.cmpi slt, %[[OC]], %{{.*}} : i32
// CHECK:           scf.condition(%{{.*}}) %[[OC]], %[[IC]], %[[OH]], %[[OW]], %[[KH]], %[[KW]] : i32, i32, i32, i32, i32, i32
// CHECK:         } do {
// CHECK:         ^bb0(%[[BOC:.*]]: i32, %[[BIC:.*]]: i32, %[[BOH:.*]]: i32, %[[BOW:.*]]: i32, %[[BKH:.*]]: i32, %[[BKW:.*]]: i32):
// Body references every IV through the cloned index_cast chain.
// CHECK:           arith.addi %[[BOH]], %[[BKH]] : i32
// CHECK:           arith.addi %[[BOW]], %[[BKW]] : i32
// CHECK:           memref.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<8x8x3xi32>
// CHECK:           memref.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<3x3x3x4xi32>
// CHECK:           arith.muli
// CHECK:           memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<6x6x4xi32>
// Odometer: kw level (innermost) -- unconditional advance, wrap, reset.
// CHECK:           %[[KWN:.*]] = arith.addi %[[BKW]], %{{.*}} : i32
// CHECK:           %[[DKW:.*]] = arith.cmpi sge, %[[KWN]], %{{.*}} : i32
// CHECK:           arith.select %[[DKW]], %{{.*}}, %[[KWN]] : i32
// kh level.
// CHECK:           %[[KHT:.*]] = arith.addi %[[BKH]], %{{.*}} : i32
// CHECK:           %[[KHN:.*]] = arith.select %[[DKW]], %[[KHT]], %[[BKH]] : i32
// CHECK:           %[[KHR:.*]] = arith.cmpi sge, %[[KHN]], %{{.*}} : i32
// CHECK:           %[[DKH:.*]] = arith.andi %[[DKW]], %[[KHR]] : i1
// CHECK:           arith.select %[[DKH]], %{{.*}}, %[[KHN]] : i32
// ow level.
// CHECK:           %[[OWT:.*]] = arith.addi %[[BOW]], %{{.*}} : i32
// CHECK:           %[[OWN:.*]] = arith.select %[[DKH]], %[[OWT]], %[[BOW]] : i32
// CHECK:           %[[OWR:.*]] = arith.cmpi sge, %[[OWN]], %{{.*}} : i32
// CHECK:           %[[DOW:.*]] = arith.andi %[[DKH]], %[[OWR]] : i1
// CHECK:           arith.select %[[DOW]], %{{.*}}, %[[OWN]] : i32
// oh level.
// CHECK:           %[[OHT:.*]] = arith.addi %[[BOH]], %{{.*}} : i32
// CHECK:           %[[OHN:.*]] = arith.select %[[DOW]], %[[OHT]], %[[BOH]] : i32
// CHECK:           %[[OHR:.*]] = arith.cmpi sge, %[[OHN]], %{{.*}} : i32
// CHECK:           %[[DOH:.*]] = arith.andi %[[DOW]], %[[OHR]] : i1
// CHECK:           arith.select %[[DOH]], %{{.*}}, %[[OHN]] : i32
// ic level.
// CHECK:           %[[ICT:.*]] = arith.addi %[[BIC]], %{{.*}} : i32
// CHECK:           %[[ICN:.*]] = arith.select %[[DOH]], %[[ICT]], %[[BIC]] : i32
// CHECK:           %[[ICR:.*]] = arith.cmpi sge, %[[ICN]], %{{.*}} : i32
// CHECK:           %[[DIC:.*]] = arith.andi %[[DOH]], %[[ICR]] : i1
// CHECK:           arith.select %[[DIC]], %{{.*}}, %[[ICN]] : i32
// oc level (outermost) -- advance conditionally, NEVER reset.
// CHECK:           %[[OCT:.*]] = arith.addi %[[BOC]], %{{.*}} : i32
// CHECK:           %[[OCO:.*]] = arith.select %[[DIC]], %[[OCT]], %[[BOC]] : i32
// CHECK:           scf.yield %[[OCO]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32, i32, i32, i32
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @conv2d(%in: memref<8x8x3xi32>,
                  %w:  memref<3x3x3x4xi32>,
                  %out: memref<6x6x4xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c6 = arith.constant 6 : i32
  %0 = scf.while (%oc = %c0) : (i32) -> i32 {
    %c = arith.cmpi slt, %oc, %c4 : i32
    scf.condition(%c) %oc : i32
  } do {
  ^bb0(%oc: i32):
    %1 = scf.while (%ic = %c0) : (i32) -> i32 {
      %c = arith.cmpi slt, %ic, %c3 : i32
      scf.condition(%c) %ic : i32
    } do {
    ^bb0(%ic: i32):
      %2 = scf.while (%oh = %c0) : (i32) -> i32 {
        %c = arith.cmpi slt, %oh, %c6 : i32
        scf.condition(%c) %oh : i32
      } do {
      ^bb0(%oh: i32):
        %3 = scf.while (%ow = %c0) : (i32) -> i32 {
          %c = arith.cmpi slt, %ow, %c6 : i32
          scf.condition(%c) %ow : i32
        } do {
        ^bb0(%ow: i32):
          %4 = scf.while (%kh = %c0) : (i32) -> i32 {
            %c = arith.cmpi slt, %kh, %c3 : i32
            scf.condition(%c) %kh : i32
          } do {
          ^bb0(%kh: i32):
            %5 = scf.while (%kw = %c0) : (i32) -> i32 {
              %c = arith.cmpi slt, %kw, %c3 : i32
              scf.condition(%c) %kw : i32
            } do {
            ^bb0(%kw: i32):
              %ih = arith.addi %oh, %kh : i32
              %iw = arith.addi %ow, %kw : i32
              %ih_idx = arith.index_cast %ih : i32 to index
              %iw_idx = arith.index_cast %iw : i32 to index
              %ic_idx = arith.index_cast %ic : i32 to index
              %kh_idx = arith.index_cast %kh : i32 to index
              %kw_idx = arith.index_cast %kw : i32 to index
              %oc_idx = arith.index_cast %oc : i32 to index
              %oh_idx = arith.index_cast %oh : i32 to index
              %ow_idx = arith.index_cast %ow : i32 to index
              %iv = memref.load %in[%ih_idx, %iw_idx, %ic_idx] : memref<8x8x3xi32>
              %wv = memref.load %w[%kh_idx, %kw_idx, %ic_idx, %oc_idx] : memref<3x3x3x4xi32>
              %prod = arith.muli %iv, %wv : i32
              %acc = memref.load %out[%oh_idx, %ow_idx, %oc_idx] : memref<6x6x4xi32>
              %sum = arith.addi %acc, %prod : i32
              memref.store %sum, %out[%oh_idx, %ow_idx, %oc_idx] : memref<6x6x4xi32>
              %kwn = arith.addi %kw, %c1 : i32
              scf.yield %kwn : i32
            }
            %khn = arith.addi %kh, %c1 : i32
            scf.yield %khn : i32
          }
          %own = arith.addi %ow, %c1 : i32
          scf.yield %own : i32
        }
        %ohn = arith.addi %oh, %c1 : i32
        scf.yield %ohn : i32
      }
      %icn = arith.addi %ic, %c1 : i32
      scf.yield %icn : i32
    }
    %ocn = arith.addi %oc, %c1 : i32
    scf.yield %ocn : i32
  }
  return
}

// -----

// Linearized address promotion (packed): a 2D loop nest whose inner body
// loads from memref<12xi32> via the FlattenMemRefs-style chain `i*4 + j`.
// With i in [0,3) and j in [0,4), the per-iteration delta folds to +1
// unconditionally and no select is needed for the address iter-arg.

// CHECK-LABEL: func.func @linearized_addr_packed
// CHECK:         %[[INIT:.*]] = arith.constant 0 : index
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A:.*]] = %[[INIT]]) : (i32, i32, index) -> (i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BA:.*]]: index):
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<12xi32>
// CHECK:           %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           %[[AN:.*]] = arith.addi %[[BA]], %[[ONE]] : index
// CHECK:           scf.yield %{{.*}}, %{{.*}}, %[[AN]] : i32, i32, index
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @linearized_addr_packed(%m: memref<12xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c4i = arith.constant 4 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %mul = arith.muli %ii, %c4i : index
      %addr = arith.addi %mul, %ji : index
      %v = memref.load %m[%addr] : memref<12xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Linearized address promotion with carry: i in [0,3), j in [0,2), but the
// linearized stride for i is still 4 (so the access skips 2 addresses each
// time the inner loop wraps). The delta becomes `1 + select(done_j, 2, 0)`.

// CHECK-LABEL: func.func @linearized_addr_carry
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A:.*]] = %{{.*}}) : (i32, i32, index) -> (i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BA:.*]]: index):
// CHECK-NOT:       arith.muli
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<12xi32>
// CHECK:           %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           %[[A1:.*]] = arith.addi %[[BA]], %[[ONE]] : index
// CHECK:           %[[TWO:.*]] = arith.constant 2 : index
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[SEL:.*]] = arith.select %{{.*}}, %[[TWO]], %[[ZERO]] : index
// CHECK:           %[[A2:.*]] = arith.addi %[[A1]], %[[SEL]] : index
// CHECK:           scf.yield %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, index
// CHECK:         }
// CHECK-NOT:     scf.while
func.func @linearized_addr_carry(%m: memref<12xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4i = arith.constant 4 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c2 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %mul = arith.muli %ii, %c4i : index
      %addr = arith.addi %mul, %ji : index
      %v = memref.load %m[%addr] : memref<12xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// 3-level perfectly-packed linearization: i*M*K + j*K + k for memref<24xi32>
// with N=2, M=3, K=4. Both wrap-correction terms cancel exactly, so the
// address iter-arg increments by +1 every iteration with no selects.

// CHECK-LABEL: func.func @linearized_addr_3d_packed
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A:.*]] = %{{.*}}) : (i32, i32, i32, index) -> (i32, i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BK:.*]]: i32, %[[BA:.*]]: index):
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.shli
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<24xi32>
// CHECK:           %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           %[[AN:.*]] = arith.addi %[[BA]], %[[ONE]] : index
// CHECK-NOT:       arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK:           scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %[[AN]] : i32, i32, i32, index
func.func @linearized_addr_3d_packed(%m: memref<24xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c4i = arith.constant 4 : index
  %c12i = arith.constant 12 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c2 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c3 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %rk = scf.while (%k = %c0) : (i32) -> i32 {
        %ck = arith.cmpi slt, %k, %c4 : i32
        scf.condition(%ck) %k : i32
      } do {
      ^bb0(%k: i32):
        %ii = arith.index_cast %i : i32 to index
        %ji = arith.index_cast %j : i32 to index
        %ki = arith.index_cast %k : i32 to index
        %m1 = arith.muli %ii, %c12i : index
        %m2 = arith.muli %ji, %c4i : index
        %s1 = arith.addi %m1, %m2 : index
        %addr = arith.addi %s1, %ki : index
        %v = memref.load %m[%addr] : memref<24xi32>
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

// FlattenMemRefs uses arith.shli when the stride is a power of two. The
// matcher must recognize shli-by-constant as multiplication.

// CHECK-LABEL: func.func @linearized_addr_shli
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A:.*]] = %{{.*}}) : (i32, i32, index) -> (i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BA:.*]]: index):
// CHECK-NOT:       arith.shli
// CHECK-NOT:       arith.muli
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<8xi32>
// CHECK:           %[[ONE:.*]] = arith.constant 1 : index
// CHECK:           arith.addi %[[BA]], %[[ONE]] : index
func.func @linearized_addr_shli(%m: memref<8xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c2i = arith.constant 2 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c2 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      // i * 4 expressed as (i shl 2)
      %sh = arith.shli %ii, %c2i : index
      %addr = arith.addi %sh, %ji : index
      %v = memref.load %m[%addr] : memref<8xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// A load and a store sharing the same address SSA value should be
// de-duplicated to a single address iter-arg, not two.

// CHECK-LABEL: func.func @linearized_addr_dedup
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A:.*]] = %{{.*}}) : (i32, i32, index) -> (i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%{{.*}}: i32, %{{.*}}: i32, %[[BA:.*]]: index):
// CHECK:           %[[V:.*]] = memref.load %{{.*}}[%[[BA]]] : memref<12xi32>
// CHECK:           memref.store %[[V]], %{{.*}}[%[[BA]]] : memref<12xi32>
func.func @linearized_addr_dedup(%a: memref<12xi32>, %b: memref<12xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c4i = arith.constant 4 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %mul = arith.muli %ii, %c4i : index
      %addr = arith.addi %mul, %ji : index
      %v = memref.load %a[%addr] : memref<12xi32>
      memref.store %v, %b[%addr] : memref<12xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Two distinct linearized addresses each get their own iter-arg.
// addrA = i*4 + j (row-major); addrB = j*3 + i (transposed).

// CHECK-LABEL: func.func @linearized_addr_two_distinct
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32, index, index) -> (i32, i32, index, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BA:.*]]: index, %[[BB:.*]]: index):
// CHECK-NOT:       arith.muli
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<12xi32>
// CHECK:           memref.load %{{.*}}[%[[BB]]] : memref<12xi32>
func.func @linearized_addr_two_distinct(%a: memref<12xi32>, %b: memref<12xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c3i = arith.constant 3 : index
  %c4i = arith.constant 4 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %m1 = arith.muli %ii, %c4i : index
      %addrA = arith.addi %m1, %ji : index
      %m2 = arith.muli %ji, %c3i : index
      %addrB = arith.addi %m2, %ii : index
      %va = memref.load %a[%addrA] : memref<12xi32>
      %vb = memref.load %b[%addrB] : memref<12xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Non-unit innermost step: j steps by 2 from 0..8. Per-iter base becomes
// stride[inner]*step[inner] = 1*2 = 2; carry term is 4*1 - 1*8 = -4.

// CHECK-LABEL: func.func @linearized_addr_nonunit_step
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %[[A:.*]] = %{{.*}}) : (i32, i32, index) -> (i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BA:.*]]: index):
// CHECK-NOT:       arith.muli
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<12xi32>
// CHECK:           %[[TWO:.*]] = arith.constant 2 : index
// CHECK:           %[[A1:.*]] = arith.addi %[[BA]], %[[TWO]] : index
// CHECK:           %[[NEG4:.*]] = arith.constant -4 : index
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[SEL:.*]] = arith.select %{{.*}}, %[[NEG4]], %[[ZERO]] : index
// CHECK:           %[[A2:.*]] = arith.addi %[[A1]], %[[SEL]] : index
func.func @linearized_addr_nonunit_step(%m: memref<12xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c8 = arith.constant 8 : i32
  %c4i = arith.constant 4 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c8 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %mul = arith.muli %ii, %c4i : index
      %addr = arith.addi %mul, %ji : index
      %v = memref.load %m[%addr] : memref<12xi32>
      %jn = arith.addi %j, %c2 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Negative case: the address depends on a value loaded from memory, which
// is not a constant-coefficient linear combination of the IVs. The loop
// nest must still flatten, and the indirect access keeps its dynamic
// address. The first load — whose address IS linear in the IVs — is
// still promoted to an iter-arg.

// CHECK-LABEL: func.func @linearized_addr_dynamic_indirect
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32, index) -> (i32, i32, index)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32, %[[BA:.*]]: index):
// CHECK:           memref.load %{{.*}}[%[[BA]]] : memref<12xi32>
// CHECK:           memref.load %{{.*}}[%{{.*}}] : memref<16xi32>
// CHECK-NOT:     scf.while
func.func @linearized_addr_dynamic_indirect(%idx: memref<12xi32>, %m: memref<16xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c4i = arith.constant 4 : index
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ii = arith.index_cast %i : i32 to index
      %ji = arith.index_cast %j : i32 to index
      %mul = arith.muli %ii, %c4i : index
      %lin = arith.addi %mul, %ji : index
      %perm = memref.load %idx[%lin] : memref<12xi32>
      %permIdx = arith.index_cast %perm : i32 to index
      %v = memref.load %m[%permIdx] : memref<16xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Negative case: a bare IV used as an index (after a unit index_cast)
// must NOT be promoted — the per-level iter-arg already tracks it. The
// loop still flattens; the load just consumes the cast.

// CHECK-LABEL: func.func @linearized_addr_bare_iv
// CHECK:         scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (i32, i32) -> (i32, i32)
// CHECK:         do {
// CHECK:         ^bb0(%[[BI:.*]]: i32, %[[BJ:.*]]: i32):
// CHECK:           %[[CAST:.*]] = arith.index_cast %[[BJ]] : i32 to index
// CHECK:           memref.load %{{.*}}[%[[CAST]]] : memref<4xi32>
// CHECK-NOT:     scf.while
func.func @linearized_addr_bare_iv(%m: memref<4xi32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %r = scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    %rj = scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %ji = arith.index_cast %j : i32 to index
      %v = memref.load %m[%ji] : memref<4xi32>
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    }
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Positive: hls.pipeline (unit attr) on inner loop is preserved on flattened loop.

// CHECK-LABEL: func.func @pipeline_attr_unit
// CHECK:         scf.while
// CHECK:         } attributes {hls.pipeline}
func.func @pipeline_attr_unit() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %s = arith.addi %i, %j : i32
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    } attributes {hls.pipeline}
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}

// -----

// Positive: hls.pipeline with explicit II value is preserved on flattened loop.

// CHECK-LABEL: func.func @pipeline_attr_ii
// CHECK:         scf.while
// CHECK:         } attributes {hls.pipeline = 2 : i64}
func.func @pipeline_attr_ii() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %ci = arith.cmpi slt, %i, %c3 : i32
    scf.condition(%ci) %i : i32
  } do {
  ^bb0(%i: i32):
    scf.while (%j = %c0) : (i32) -> i32 {
      %cj = arith.cmpi slt, %j, %c4 : i32
      scf.condition(%cj) %j : i32
    } do {
    ^bb0(%j: i32):
      %s = arith.addi %i, %j : i32
      %jn = arith.addi %j, %c1 : i32
      scf.yield %jn : i32
    } attributes {hls.pipeline = 2 : i64}
    %in = arith.addi %i, %c1 : i32
    scf.yield %in : i32
  }
  return
}
