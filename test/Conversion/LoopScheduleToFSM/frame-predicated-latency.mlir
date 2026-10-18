// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A frame lasts until each op of its `at` bodies has its result, consumed in
// the frame or not, so the op's next execution (the next iteration's) comes
// no sooner than its latency: an operator that takes an operand only every
// so many cycles relies on that. The ops a `loopschedule.if` predicates
// count too. Here the loop body's only multi-cycle op is a predicated
// latency-14 divide whose result the frame does not use: the frame has 14
// cycles (it had one, the `if` counting as a single-cycle op).

// CHECK: fsm.machine @predicated_fsm
// CHECK-SAME: "loop0_frame_cycle_0_13"
// CHECK-NOT: "loop0_frame_cycle_0_14"
oplib.library @lib {
  oplib.operator @div_i32_l14 latency<14> {
    oplib.target @arith_divui_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.divui"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_divui_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "div" @div_pipe(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @div_pipe(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

loopschedule.func_sequential @predicated(%x: i32, %y: i32, %n: i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0) : (i32) -> i32 {
          %f:2 = loopschedule.frame -> (i1, i32) {
            %s0:2 = loopschedule.at 0 -> (i1, i32) {
              %cond = arith.cmpi slt, %i, %n : i32
              %inext = arith.addi %i, %c1 : i32
              %small = arith.cmpi slt, %i, %c2 : i32
              loopschedule.if %small {
                %q = arith.divui %x, %y {loopschedule.operator = @div_i32_l14, loopschedule.cycle_latency = 14 : i64} : i32
              }
              loopschedule.iter_arg_update %i = %inext : i32
              loopschedule.yield %cond, %inext : i1, i32
            }
            loopschedule.yield %s0#0, %s0#1 : i1, i32
          }
          loopschedule.terminator condition(%f#0), results(%f#1) : i32
        }
        loopschedule.yield %seq : i32
      }
      loopschedule.yield %launch : !loopschedule.handle
    }
    loopschedule.yield %outer : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}
