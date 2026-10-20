// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s --implicit-check-not=mul_i32_hw_inst_1

// A function-level frame computes `x * x` on a combinational (latency-0)
// multiplier in its header cycle, and the pipeline it launches multiplies by
// that product on the same instance (one `loopschedule.binding`). The
// product is held from the header cycle; the pipeline, which starts the
// cycle after, reads it from the hold. Read through a bypass of the
// multiplier's output (`child_start ? out : hold`), it would route the
// instance's output back into its own operands: never selected together,
// but a combinational loop to the timer (deriche: 75 ns through one
// multiply, 32 times).

oplib.library @lib {
  oplib.operator @mul_i32_l0 latency<0>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_i32_hw_inst" @int_mul_i32_l0(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @int_mul_i32_l0(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32}) attributes {verilogName = "int_mul_i32_l0"}

// CHECK-LABEL: hw.module @frame_shared_comb
// CHECK:       hw.instance "mul_i32_hw_inst_0" @int_mul_i32_l0({{.*}}lhs: %[[LHS:[0-9]+]]: i32, rhs: %[[RHS:[0-9]+]]: i32)
// CHECK:       %[[HOLD:frame[0-9]+_hold_[0-9]+]] = seq.compreg.ce sym @{{[a-z0-9_]+}} %mul_i32_hw_inst_0.out, %clk
// CHECK-NOT:   comb.mux {{.*}}%mul_i32_hw_inst_0.out
// The pipeline's multiply takes the product from the hold.
// CHECK:       %[[LHS]] = comb.mux %{{.+}}, %[[HOLD]], %{{.+}} : i32
loopschedule.func_sequential @frame_shared_comb(%x: i32, %n: i32) -> i32
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %h:2 = loopschedule.frame -> (i32, !loopschedule.handle) {
    %p = loopschedule.at 0 -> i32 {
      %v = arith.muli %x, %x {loopschedule.operator = @mul_i32_l0, loopschedule.binding = 0 : i64} : i32
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
          %s = loopschedule.at 1 -> i32 {
            %m = arith.muli %p, %r#1 {loopschedule.operator = @mul_i32_l0, loopschedule.binding = 0 : i64} : i32
            %a = arith.addi %acc, %m : i32
            loopschedule.iter_arg_update %acc = %a : i32
            loopschedule.yield %a : i32
          }
          loopschedule.terminator condition(%r#2), results(%r#0, %s) : i32, i32
        }
        loopschedule.yield %pip#1 : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    loopschedule.yield %p, %hp : i32, !loopschedule.handle
  }
  %result = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h#1 -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %w = loopschedule.at 0 -> i32 {
      %u = arith.addi %v, %h#0 : i32
      loopschedule.yield %u : i32
    }
    loopschedule.yield %w : i32
  }
  loopschedule.return %result : i32
}
