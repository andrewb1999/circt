// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0})" %s | FileCheck %s --check-prefix=SIMPLEX
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 scheduler=cpsat share-mux-delays=0.3,0.3,0.6})" %s | FileCheck %s --check-prefix=CPSAT

// A unit `hls.pipeline` asks for the least II. The accumulation is a
// recurrence through its add, whose operator allocation chose takes 2
// cycles: the simplex, scheduling that operator, takes II 2. The CP-SAT
// scheduler's II is a variable, from the resources' bound up: at II 1 the
// add takes its 1-cycle candidate (the same area) and the recurrence closes,
// in about half the cycles.

oplib.library @lib {
  oplib.operator @add_l1 latency<1>, incDelay<1.0>, outDelay<0.5> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_addi_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "add" @add_l1(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @add_l2 latency<2>, incDelay<0.5>, outDelay<0.5> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_addi_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "add" @add_l2(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @add_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @add_l2(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

// SIMPLEX-LABEL: loopschedule.func_sequential @accumulate
// SIMPLEX:         loopschedule.pipeline II = 2
// SIMPLEX:           arith.addi %{{.*}} {{.*}}loopschedule.operator = @add_l2
// CPSAT-LABEL:   loopschedule.func_sequential @accumulate
// CPSAT:           loopschedule.pipeline II = 1
// CPSAT:             arith.addi %{{.*}} {{.*}}loopschedule.operator = @add_l1
func.func @accumulate(%x: i32) -> i32 attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %res:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %ns = arith.addi %s, %x {loopschedule.operator = @add_l2, loopschedule.candidates = [@add_l2, @add_l1]} : i32
    %ni = arith.addi %i, %c1 {loopschedule.operator = @add_l1} : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline}
  return %res#1 : i32
}
