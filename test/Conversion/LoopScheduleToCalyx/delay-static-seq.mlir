// RUN: circt-opt --lower-loopschedule-to-calyx %s | FileCheck %s

// Hand-crafted sequential loop whose step contains a `loopschedule.delay 2`
// region wrapping a store. The Calyx lowering should emit the step's body as
// a static_par containing a static_seq { delay_pad ; static_par { body } } so
// the inner store fires two cycles into the enclosing step.

module {
  func.func @delay_step(%arg0: memref<16xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    loopschedule.step {
    loopschedule.sequential trip_count = 10 iter_args(%i = %c0_i32) : (i32) -> () {
      %0:2 = loopschedule.step {
        %cond = arith.cmpi slt, %i, %c10_i32 : i32
        %next = arith.addi %i, %c1_i32 : i32
        loopschedule.delay 2 {
          loopschedule.store %next, %arg0[%i : i32] : memref<16xi32>
          loopschedule.register
        }
        loopschedule.register %next, %cond : i32, i1
      } : i32, i1
      loopschedule.terminator condition(%0#1), iter_args(%0#0), results() : (i32) -> ()
    }
    }
    return
  }
}

// CHECK: calyx.component @delay_step
// CHECK: calyx.wires
// A padding static group of latency 2 should exist for the delay region.
// CHECK-DAG: calyx.static_group latency<2> @delay_pad
// CHECK: calyx.control
// The step body should contain a static_seq enabling the pad followed by a
// static_par for the delay's inner schedulables.
// CHECK: calyx.static_par
// CHECK: calyx.static_seq
// CHECK: calyx.enable @delay_pad
// CHECK: calyx.static_par
