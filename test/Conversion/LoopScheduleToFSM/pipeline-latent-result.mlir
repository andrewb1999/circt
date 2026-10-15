// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A pipeline result a multi-cycle operator produces is the operator's
// output, valid on the one cycle its iteration reaches issue + latency (here
// stage 0 + 2). The operator keeps running after the drain on whatever its
// inputs hold; this accumulator's multiply would apply x again every cycle
// until the result is read. The result latches each time a live iteration
// reaches that cycle (the stage-CE chain continued past the last stage).
//
// The trip count is a runtime value, so the loop can run no iteration at
// all and must then return its iter-args' inits: `ran` records that an
// iteration got past the condition since the launch, and each result falls
// back to its init without one.

// CHECK-LABEL: hw.module @power
// CHECK:     %[[MUL:.+]] = hw.instance "mul_0" @mul_pipe
// CHECK-DAG: %loop0_ran = seq.compreg sym @loop0_ran
// CHECK-DAG: comb.mux %loop0_ran, %loop0_s0_r0, %c0_i32 : i32
// CHECK-DAG: %loop0_live_s1 = seq.compreg.ce sym @loop0_live_s1
// CHECK-DAG: %loop0_live_s2 = seq.compreg.ce sym @loop0_live_s2 %loop0_live_s1
// CHECK-DAG: %[[READY:.+]] = comb.and %loop0_live_s2, %{{.+}} : i1
// CHECK-DAG: %loop0_result_r0 = seq.compreg.ce sym @loop0_result_r0 %[[MUL]], %clk, %[[READY]]
// CHECK-DAG: comb.mux %loop0_ran, %loop0_result_r0, %c1_i32 : i32

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

loopschedule.func_sequential @power(%x: i32, %n: i32) -> i32
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip:2 = loopschedule.pipeline II = 2 iter_args(%i = %c0, %acc = %c1) : (i32, i32) -> (i32, i32) {
          %r:3 = loopschedule.at 0 -> (i32, i32, i1) {
            %cond = arith.cmpi slt, %i, %n : i32
            %next = arith.addi %i, %c1 : i32
            %m = arith.muli %acc, %x {loopschedule.operator = @mul_i32_l2} : i32
            loopschedule.iter_arg_update %i = %next : i32
            loopschedule.iter_arg_update %acc = %m : i32
            loopschedule.yield %next, %m, %cond : i32, i32, i1
          }
          loopschedule.terminator condition(%r#2), results(%r#0, %r#1) : i32, i32
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
