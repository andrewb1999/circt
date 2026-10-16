// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// An operator with state, an accumulator, must see each execution of its op
// once: its `valid` port (`oplib.valid`) is high in the cycles the op
// issues, the gate its stage drives stores with (live, not stalled).

// CHECK-LABEL: hw.module @sum
// CHECK: %loop0_ce_stage_1 = seq.compreg.ce sym @loop0_ce_stage_1 %{{.+}}, %clk, %[[NOTSTALL:[^ ]+]] reset
// CHECK: %[[S1:.+]] = comb.and %loop0_ce_stage_1, %[[NOTSTALL]] : i1
// CHECK: hw.instance "acc_0" @acc_core({{.*}}ce: %[[NOTSTALL]]: i1, x: %loop0_s0_r1: i32, y: %{{.+}}: i32, valid: %[[S1]]: i1)

oplib.library @lib {
  oplib.operator @acc_i32 latency<1>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_addi_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32, valid %v : i1) {
      %0 = oplib.hw_instance "acc" @acc_core(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, x: %arg0 : i32, y: %arg1 : i32, valid: %v : i1) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @acc_core(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %x : i32 {oplib.operand = 0 : i32}, in %y : i32 {oplib.operand = 1 : i32}, in %valid : i1 {oplib.valid}, out out : i32 {oplib.result = 0 : i32})

loopschedule.func_sequential @sum(%n: i32) -> i32
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip:2 = loopschedule.pipeline II = 1 iter_args(%i = %c0) : (i32) -> (i32, i32) {
          %r:3 = loopschedule.at 0 -> (i32, i32, i1) {
            %cond = arith.cmpi slt, %i, %n : i32
            %next = arith.addi %i, %c1 : i32
            loopschedule.iter_arg_update %i = %next : i32
            loopschedule.yield %next, %i, %cond : i32, i32, i1
          }
          %s = loopschedule.at 1 -> i32 {
            %a = arith.addi %r#1, %c1 {loopschedule.operator = @acc_i32} : i32
            loopschedule.yield %a : i32
          }
          loopschedule.terminator condition(%r#2), results(%r#0, %s) : i32, i32
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
