// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 share-mux-delays=0.3})" %s | FileCheck %s --check-prefixes=CHECK,SIMPLEX
// RUN: %if or-tools %{ circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule-cpsat{cycle-time-ns=3.0 cpsat-workers=1 cpsat-cycle-slack=0 share-mux-delays=0.3})" %s | FileCheck %s %}

// Both schedulers time what the FSM lowering puts in front of a region's
// ops beyond the ops before them in it (the delay model, InputDelays): the
// simplex in its problem's earliest starts in a cycle, the CP-SAT model in
// its constraints. One LUT level is the two-user mux's delay (0.3 ns).

oplib.library @lib {
  oplib.operator @add_comb latency<0>, incDelay<1.0>, outDelay<1.0> {
    oplib.target @arith_addi_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "add" @add_comb(lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_l1 latency<1>, incDelay<2.0>, outDelay<0.5> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_muli_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "mul" @mul_l1(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_shared latency<1>, incDelay<2.0>, outDelay<0.5>, limit<1> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_muli_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "mul" @mul_l1(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @add_comb(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @mul_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

// A function-level frame's operator result, read in a loop through its
// hold's bypass (`ready ? instance : capture`): a LUT past the multiply's
// output delay, 0.8 ns into the cycle. Three 1 ns adds chained on it take
// two cycles (from a register they would take one).
// CHECK-LABEL: loopschedule.func_sequential @held
// CHECK:         loopschedule.sequential
// CHECK:           loopschedule.frame
// CHECK:             loopschedule.at 0
// CHECK-COUNT-2:       @add_comb
// CHECK-NOT:           @add_comb
// CHECK:             loopschedule.at 1
// CHECK-NEXT:          @add_comb
func.func @held(%x: i32, %y: i32) -> i32 attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i16
  %c1 = arith.constant 1 : i16
  %c16 = arith.constant 16 : i16
  %s0 = arith.constant 0 : i32
  %m = arith.muli %x, %y {loopschedule.operator = @mul_l1} : i32
  %r:2 = scf.while (%i = %c0, %s = %s0) : (i16, i32) -> (i16, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i16
    scf.condition(%cond) %i, %s : i16, i32
  } do {
  ^bb0(%i: i16, %s: i32):
    %a = arith.addi %s, %m {loopschedule.operator = @add_comb} : i32
    %b = arith.addi %a, %m {loopschedule.operator = @add_comb} : i32
    %d = arith.addi %b, %m {loopschedule.operator = @add_comb} : i32
    %ni = arith.addi %i, %c1 : i16
    scf.yield %ni, %d : i16, i32
  }
  return %r#1 : i32
}

// A pipeline's cycle-0 op on a shared instance (an operator with a limit,
// allocation's shared variant) selects with the stage-0 enable, decided in
// the cycle: here 1.2 ns in, behind an add and a compare. A LUT after it,
// the multiply's 2 ns input path ends past the cycle, so the simplex keeps
// it out of cycle 0, a cycle past the condition.
// SIMPLEX-LABEL: loopschedule.func_sequential @late_select
// SIMPLEX:         loopschedule.pipeline II = 2
// SIMPLEX:           loopschedule.at 0
// SIMPLEX-NOT:         arith.muli
// SIMPLEX:           loopschedule.at 1
// SIMPLEX-NEXT:        arith.muli
func.func @late_select(%x: i32, %y: i32) -> i32 attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %r:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %n = arith.addi %i, %c1 {loopschedule.operator = @add_comb} : i32
    %cond = arith.cmpi slt, %n, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %p = arith.muli %x, %i {loopschedule.operator = @mul_shared} : i32
    %q = arith.muli %p, %y {loopschedule.operator = @mul_shared} : i32
    %ns = arith.addi %s, %q {loopschedule.operator = @add_comb} : i32
    %ni = arith.addi %i, %c1 {loopschedule.operator = @add_comb} : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline = 2}
  return %r#1 : i32
}
