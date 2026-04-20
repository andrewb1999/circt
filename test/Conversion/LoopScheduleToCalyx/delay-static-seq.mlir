// RUN: circt-opt --lower-loopschedule-to-calyx %s | FileCheck %s

// Sequential loop whose body-frame contains an `at 2` region wrapping a
// store. The Calyx lowering should emit the frame's body as a static_par
// containing a static_seq { at_pad; static_par { body } } so the inner
// store fires two cycles into the enclosing frame.
//
// The outer sequential loop lives inside a `loopschedule.launch at 0` in a
// top-level frame (new surface). A trailing `await` frame consumes the
// launch's handle.

module {
  func.func @delay_step(%arg0: memref<16xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 10 iter_args(%i = %c0_i32) : (i32) -> () {
            %cond, %next = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c10_i32 : i32
                %n = arith.addi %i, %c1_i32 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield %c, %n : i1, i32
              }
              loopschedule.at 2 {
                loopschedule.store %r#1, %arg0[%i : i32] : memref<16xi32>
                loopschedule.yield
              }
              loopschedule.yield %r#0, %r#1 : i1, i32
            }
            loopschedule.terminator condition(%cond), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %lh_launch : !loopschedule.handle
      }
      loopschedule.yield %lh : !loopschedule.handle
    }
    loopschedule.frame {
      loopschedule.await %h
      loopschedule.yield
    } do {
      loopschedule.yield
    }
    return
  }
}

// CHECK: calyx.component @delay_step
// CHECK: calyx.wires
// A padding static group of latency 2 should exist for the `at 2` region.
// CHECK-DAG: calyx.static_group latency<2> @at_pad
// CHECK: calyx.control
// The frame body should contain a static_seq enabling the pad followed by a
// static_par for the at-2's inner schedulables.
// CHECK: calyx.static_par
// CHECK: calyx.static_seq
// CHECK: calyx.enable @at_pad
// CHECK: calyx.static_par
