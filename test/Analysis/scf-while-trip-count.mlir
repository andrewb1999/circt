// RUN: circt-opt %s -test-scf-while-trip-count -verify-diagnostics -o /dev/null

// Canonical slt + unit step starting at 0: 10 iterations.
func.func @slt_unit_step() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 10}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Unsigned ult with step 2 from 0 to 10: 5 iterations.
func.func @ult_step_two() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c2 = arith.constant 2 : i32
  // expected-remark @below {{trip_count = 5}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi ult, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c2 : i32
    scf.yield %next : i32
  }
  return
}

// sle (inclusive): 0..=9 with unit step -> 10 iterations.
func.func @sle_inclusive() {
  %c0 = arith.constant 0 : i32
  %c9 = arith.constant 9 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 10}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi sle, %iv, %c9 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// ne with unit step from 0 to 4: 4 iterations.
func.func @ne_unit_step() {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 4}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi ne, %iv, %c4 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Non-zero start: 3 to 10 step 1 = 7 iterations.
func.func @nonzero_start() {
  %c3 = arith.constant 3 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 7}}
  %res = scf.while (%iv = %c3) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Inexact division: 0 to 10 step 3 -> ceilDiv(10,3) = 4.
func.func @inexact_step() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c3 = arith.constant 3 : i32
  // expected-remark @below {{trip_count = 4}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c3 : i32
    scf.yield %next : i32
  }
  return
}

// Trip count of 0: lb == ub under slt.
func.func @zero_trips() {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 0}}
  %res = scf.while (%iv = %c5) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c5 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Trip count of 1.
func.func @one_trip() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 1}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c1 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Commutative addi: constant on the left.
func.func @commutative_add() {
  %c0 = arith.constant 0 : i32
  %c8 = arith.constant 8 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = 8}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c8 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %c1, %arg : i32
    scf.yield %next : i32
  }
  return
}

// ----- Negative cases -----

// Unsupported predicate (sgt).
func.func @unsupported_pred(%arg0: i32) {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi sgt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Non-constant upper bound.
func.func @nonconst_ub(%arg0: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %arg0 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Non-constant initial value.
func.func @nonconst_init(%arg0: i32) {
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %arg0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Non-constant step.
func.func @nonconst_step(%arg0: i32) {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %arg0 : i32
    scf.yield %next : i32
  }
  return
}

// Update is a subi, not addi.
func.func @not_addi() {
  %c10 = arith.constant 10 : i32
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %c10) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c0 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.subi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}

// Zero step.
func.func @zero_step() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %iv, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c0 : i32
    scf.yield %next : i32
  }
  return
}

// IV lhs of cmpi is not a block argument (it's the result of an op).
func.func @iv_lhs_not_block_arg() {
  %c0 = arith.constant 0 : i32
  %c10 = arith.constant 10 : i32
  %c1 = arith.constant 1 : i32
  // expected-remark @below {{trip_count = none}}
  %res = scf.while (%iv = %c0) : (i32) -> i32 {
    %shifted = arith.addi %iv, %c1 : i32
    %cond = arith.cmpi slt, %shifted, %c10 : i32
    scf.condition(%cond) %iv : i32
  } do {
  ^bb0(%arg: i32):
    %next = arith.addi %arg, %c1 : i32
    scf.yield %next : i32
  }
  return
}
