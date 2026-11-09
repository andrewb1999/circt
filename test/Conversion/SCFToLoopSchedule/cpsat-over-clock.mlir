// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 scheduler=cpsat cpsat-workers=1})" %s 2>&1 | FileCheck %s

// Operator allocation gives a pipelined loop's recurrence the latency its
// II needs, past the clock if it must (the II takes precedence there), and
// the scheduler clamps such an operator's delays to the cycle, so in the
// CP-SAT model it would seem to fit. This scheduler keeps the clock and
// chooses the II: it takes an operator slower than the cycle only where
// none of the op's candidates fits. Here the accumulation's add was given
// a combinational core of 3.5 ns at a 3 ns cycle, two DSPs; its 1-cycle
// candidate, three, closes the recurrence at II 1 too, and is taken (the
// cheaper slow one was).

// CHECK: warning: operator 'add_slow_l0' delays (incoming 3.500000e+00, outgoing 3.500000e+00 ns) exceed the cycle time
// CHECK-LABEL: loopschedule.func_sequential @accumulate
// CHECK:         loopschedule.pipeline II = 1
// CHECK:           arith.addi %{{.+}}, %arg0 {{{.*}}loopschedule.operator = @add_l1}

oplib.library @lib {
  oplib.operator @add_slow_l0 latency<0>, incDelay<3.5>, outDelay<3.5> attributes {amc.area_dsp = 2 : i64} {
    oplib.target @arith_addi_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "add" @add_slow_l0(lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @add_l1 latency<1>, incDelay<1.0>, outDelay<0.5> attributes {amc.area_dsp = 3 : i64} {
    oplib.target @arith_addi_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "add" @add_l1(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @add_slow_l0(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @add_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

func.func @accumulate(%x: i32) -> i32 attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %res:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %ns = arith.addi %s, %x {loopschedule.operator = @add_slow_l0, loopschedule.candidates = [@add_slow_l0, @add_l1]} : i32
    %ni = arith.addi %i, %c1 {loopschedule.operator = @add_l1} : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline}
  return %res#1 : i32
}
