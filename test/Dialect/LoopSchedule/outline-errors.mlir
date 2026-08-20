// RUN: circt-opt %s -split-input-file -verify-diagnostics

// A value already visible outside the region needs no forwarding: yielding
// it would alias an enclosing value through a module port for nothing.
func.func @yield_outside_value(%arg0: i32) -> i32 {
  %r = loopschedule.outline {
    // expected-error @below {{yielded values must be defined inside the outline region}}
    loopschedule.yield %arg0 : i32
  } : i32
  return %r : i32
}

// -----

// The yield is the result interface; its types must match the op's.
func.func @yield_type_mismatch() -> i64 {
  %r = loopschedule.outline {
    %c = arith.constant 1 : i32
    // expected-error @below {{yielded types must match parent loopschedule.outline result types}}
    loopschedule.yield %c : i32
  } : i64
  return %r : i64
}
