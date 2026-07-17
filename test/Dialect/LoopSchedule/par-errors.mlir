// RUN: circt-opt %s -split-input-file -verify-diagnostics

// Cross-child SSA is forbidden: children are order-free and communicate
// through memory only.
func.func @cross_lane_ssa(%arg0: memref<8xi32>) {
  %zero = arith.constant 0 : i32
  loopschedule.par {
    %r0 = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> (i32) {
      %v = affine.load %arg0[%i * 2] : memref<8xi32>
      %s = arith.addi %acc, %v : i32
      affine.yield %s : i32
    }
    affine.for %j = 0 to 4 {
      // expected-error @below {{uses a value defined by a sibling child}}
      %s = arith.addi %r0, %r0 : i32
      %idx = arith.index_cast %s : i32 to index
      memref.store %s, %arg0[%idx] : memref<8xi32>
    }
  }
  return
}

// -----

// Yielded values must be results of direct children.
func.func @yield_outside_value(%arg0: memref<8xi32>, %ext: i32) -> i32 {
  %r = loopschedule.par {
    affine.for %i = 0 to 4 {
      %v = affine.load %arg0[%i] : memref<8xi32>
      affine.store %v, %arg0[%i] : memref<8xi32>
    }
    // expected-error @below {{operands must be results of the par op's direct children}}
    loopschedule.yield %ext : i32
  } : i32
  return %r : i32
}

// -----

// Each yielded value must come from a distinct child.
func.func @yield_same_child(%arg0: memref<8xi32>) -> (i32, i32) {
  %zero = arith.constant 0 : i32
  %r:2 = loopschedule.par {
    %r0:2 = affine.for %i = 0 to 4 iter_args(%a = %zero, %b = %zero) -> (i32, i32) {
      %v = affine.load %arg0[%i] : memref<8xi32>
      %s = arith.addi %a, %v : i32
      affine.yield %s, %b : i32, i32
    }
    // expected-error @below {{each yielded value must come from a distinct child}}
    loopschedule.yield %r0#0, %r0#1 : i32, i32
  } : i32, i32
  return %r#0, %r#1 : i32, i32
}

// -----

// Yield types must match the par results.
func.func @yield_type_mismatch(%arg0: memref<8xi32>) -> i32 {
  %zero = arith.constant 0 : i16
  %r = loopschedule.par {
    %r0 = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> (i16) {
      %s = arith.addi %acc, %acc : i16
      affine.yield %s : i16
    }
    // expected-error @below {{yielded types must match parent loopschedule.par result types}}
    loopschedule.yield %r0 : i16
  } : i32
  return %r : i32
}
