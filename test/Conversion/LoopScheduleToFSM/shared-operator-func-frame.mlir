// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s --implicit-check-not=mul_pipe_i32_hw_inst_1

// A function-level frame that both computes and launches a child: its
// multiply is bound to the same instance (operator, `loopschedule.binding`)
// as a multiply of the pipeline it launches, which the binding pass found
// never issues in the same cycle. The lowering builds one multiplier, the
// frame's operands muxed in under its `at`'s gate (it built a second one,
// as k2mm's flattened loop showed: the chain init `D * beta` in the frame
// between its two loops).
//
// The frame reads the product again five cycles in, while the child has
// the shared multiplier: the result is held from the cycle it is ready
// (offset 0 + latency 4), and read from the hold after.

oplib.library @lib {
  oplib.operator @mul_i32_l4 latency<4>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_pipe_i32_hw_inst" @int_mul_pipe_i32_l4(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @int_mul_pipe_i32_l4(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32}) attributes {verilogName = "int_mul_pipe_i32_l4"}

// CHECK-LABEL: hw.module @frame_shared
// CHECK:       hw.instance "mul_pipe_i32_hw_inst_0" @int_mul_pipe_i32_l4({{.*}}lhs: %[[LHS:[0-9]+]]: i32, rhs: %[[RHS:[0-9]+]]: i32)
// The frame's product, held from cycle 4, read through the hold after.
// CHECK:       %[[HOLD:frame[0-9]+_hold_[0-9]+]] = seq.compreg.ce sym @{{[a-z0-9_]+}} %mul_pipe_i32_hw_inst_0.out, %clk, %[[READY:[a-z0-9_]+]]
// CHECK:       %[[HELD:[0-9]+]] = comb.mux %[[READY]], %mul_pipe_i32_hw_inst_0.out, %[[HOLD]] : i32
// CHECK:       arith.addi %[[HELD]], %{{.*}} : i32
// CHECK-DAG:   %[[LHS]] = comb.mux
// CHECK-DAG:   %[[RHS]] = comb.mux
loopschedule.func_sequential @frame_shared(%x: i32, %n: i32) -> i32
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %h:2 = loopschedule.frame -> (i32, !loopschedule.handle) {
    %p = loopschedule.at 0 -> i32 {
      %v = arith.muli %x, %x {loopschedule.operator = @mul_i32_l4, loopschedule.binding = 0 : i64} : i32
      loopschedule.yield %v : i32
    }
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip:2 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0) : (i32, i32) -> (i32, i32) {
          %r:3 = loopschedule.at 0 -> (i32, i32, i1) {
            %cond = arith.cmpi slt, %i, %n : i32
            %next = arith.addi %i, %c1 : i32
            loopschedule.iter_arg_update %i = %next : i32
            loopschedule.yield %next, %i, %cond : i32, i32, i1
          }
          %s = loopschedule.at 2 -> i32 {
            %m = arith.muli %r#1, %r#1 {loopschedule.operator = @mul_i32_l4, loopschedule.binding = 0 : i64} : i32
            loopschedule.yield %m : i32
          }
          %t = loopschedule.at 6 -> i32 {
            %a = arith.addi %acc, %s : i32
            loopschedule.iter_arg_update %acc = %a : i32
            loopschedule.yield %a : i32
          }
          loopschedule.terminator condition(%r#2), results(%r#0, %t) : i32, i32
        }
        loopschedule.yield %pip#1 : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    %w = loopschedule.at 5 -> i32 {
      %u = arith.addi %p, %c1 : i32
      loopschedule.yield %u : i32
    }
    loopschedule.yield %w, %hp : i32, !loopschedule.handle
  }
  %result = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h#1 -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %r = loopschedule.at 0 -> i32 {
      %sum = arith.addi %v, %h#0 : i32
      loopschedule.yield %sum : i32
    }
    loopschedule.yield %r : i32
  }
  loopschedule.return %result : i32
}
