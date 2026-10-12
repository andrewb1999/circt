// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A sequential frame holds no registers between its cycles, so a result
// from an operator instance is captured in the cycle it is ready (issue +
// latency): the unit keeps computing on whatever its inputs show next (and a
// shared one switches to another user's operands). Here the latency-2
// multiply issued at cycle 0 is ready at 2 and read at 3, from its capture.
// The second multiply, issued at 3, is ready in the frame's last cycle (5),
// where the iter_arg update and the frame's result register sample it: they
// read it directly, as its capture would only latch at that same edge.

// CHECK-LABEL: hw.module @hold
// CHECK: %[[M0:.*]] = hw.instance "mul_0" @mul_pipe({{.*}}rhs: %c8_i32: i32) -> (out: i32)
// CHECK: %[[HOLD:frame_hold_[0-9]+]] = seq.compreg.ce %[[M0]], %clk, %{{.*}} reset %rst
// CHECK: %[[SUM:.*]] = arith.addi %[[HOLD]], %{{.*}} : i32
// CHECK: %[[M1:.*]] = hw.instance "mul_1" @mul_pipe({{.*}}lhs: %[[SUM]]: i32, rhs: %c1_i32: i32) -> (out: i32)
// CHECK: %loop0_frame0_r1 = seq.compreg.ce sym @loop0_frame0_r1 %[[M1]], %clk
// CHECK: comb.mux %{{.*}}, %c0_i32, %[[M1]] : i32

oplib.library @lib {
  oplib.operator @mul_i32_l2 latency<2>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul" @mul_pipe(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @mul_pipe(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

loopschedule.func_sequential @hold(%m: memref<8xi32>)
      attributes {oplib.library = @lib, top} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8 iter_args(%i = %c0, %acc = %c0) : (i32, i32) -> () {
            %f:2 = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c8 : i32
                %p = arith.muli %i, %c8 {loopschedule.operator = @mul_i32_l2} : i32
                loopschedule.yield %c, %p : i1, i32
              }
              %s = loopschedule.at 3 -> i32 {
                %a = arith.addi %r#1, %acc : i32
                %q = arith.muli %a, %c1 {loopschedule.operator = @mul_i32_l2} : i32
                %n = arith.addi %i, %c1 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield %q : i32
              }
              %t = loopschedule.at 5 -> i32 {
                loopschedule.iter_arg_update %acc = %s : i32
                loopschedule.yield %s : i32
              }
              loopschedule.yield %r#0, %t : i1, i32
            }
            loopschedule.terminator condition(%f#0), results()
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
    loopschedule.return
  }
