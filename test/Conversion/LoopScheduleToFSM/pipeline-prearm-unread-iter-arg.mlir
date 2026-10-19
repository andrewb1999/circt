// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A pipeline issues in its launch cycle (pre-arm) when every iter-arg it
// reads has its init in place by then. One that nothing reads needs none:
// here %acc, the running value of a multi-cycle operator, which (like a
// multiply-accumulate's, kept in the core) is carried only to leave the
// loop. It takes no part in the decision, and no first-iteration mux.
//
// Its result is the operator's output, latched when the last iteration
// reaches the operator's latency; that latch is not a feedback register,
// so it takes no pre-arm hold: it captures on the drain edge, where a hold
// would sample the previous invocation's value.

// CHECK-LABEL: hw.module @product
// CHECK-DAG: %[[MUL:.+]] = hw.instance "mul_0" @mul_pipe
// CHECK-DAG: comb.or %loop0_active, %{{.+}}
// CHECK-DAG: fsm.variable "issue_arm_0"
// CHECK-DAG: %loop0_result_r0 = seq.compreg.ce sym @loop0_result_r0 %[[MUL]],
// CHECK-NOT: first_iter_s
// CHECK-NOT: result_hold

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

loopschedule.func_sequential @product(%x: i32, %y: i32) -> i32
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip:2 = loopschedule.pipeline II = 1 trip_count = 16 latency = 2 iter_args(%i = %c0, %acc = %c1) : (i32, i32) -> (i32, i32) {
          %r:3 = loopschedule.at 0 -> (i32, i32, i1) {
            %cond = arith.cmpi slt, %i, %c16 : i32
            %next = arith.addi %i, %c1 : i32
            %m = arith.muli %x, %y {loopschedule.operator = @mul_i32_l2} : i32
            loopschedule.iter_arg_update %i = %next : i32
            loopschedule.iter_arg_update %acc = %m : i32
            loopschedule.yield %next, %m, %cond : i32, i32, i1
          }
          %s1 = loopschedule.at 1 -> i32 {
            loopschedule.yield %r#0 : i32
          }
          loopschedule.terminator condition(%r#2), results(%s1, %r#1) : i32, i32
        }
        loopschedule.yield %pip#1 : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    loopschedule.yield %hp : !loopschedule.handle
  }
  %result = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %r = loopschedule.at 0 -> i32 {
      loopschedule.yield %v : i32
    }
    loopschedule.yield %r : i32
  }
  loopschedule.return %result : i32
}
