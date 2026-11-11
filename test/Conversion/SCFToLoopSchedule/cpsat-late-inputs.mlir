// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 scheduler=cpsat cpsat-workers=1 cpsat-cycle-slack=0})" %s | FileCheck %s
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 scheduler=cpsat cpsat-workers=1 cpsat-cycle-slack=0.5})" %s | FileCheck %s --check-prefix=SLACK

// The CP-SAT model times what the lowering puts in front of an op, not only
// the ops before it in its region.

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
}

hw.module.extern @add_comb(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @mul_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

// A sequential loop's iter arg is a register in its frames: the inits
// latch as the loop is entered, and the frames read the register, not
// `first_iter ? init : reg`. Three 1 ns adds chained on it take one cycle.
// CHECK-LABEL: loopschedule.func_sequential @iter_arg
// CHECK:         loopschedule.sequential
// CHECK:           loopschedule.frame
// CHECK:             loopschedule.at 0
// CHECK-COUNT-3:       arith.addi %{{.+}}, %arg0 {{.*}}@add_comb
// CHECK-NOT:         loopschedule.at 1
func.func @iter_arg(%x: i32) -> i32 attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %r:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %a = arith.addi %s, %x {loopschedule.operator = @add_comb} : i32
    %b = arith.addi %a, %x {loopschedule.operator = @add_comb} : i32
    %d = arith.addi %b, %x {loopschedule.operator = @add_comb} : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %d : i32, i32
  }
  return %r#1 : i32
}

// A pipeline's cycle-0 op selects a shared instance with its stage-0
// enable, `active` and the continuation decided in that cycle: here behind
// a 1 ns add. In cycle 0 the first multiply's mux would start after the
// condition, 1.19 ns in, and its 2 ns input path end past the cycle: the
// fastest design (slack 0) keeps the multiplies at cycles 0 and 1 on two
// instances, and with a slack they share one at cycles 1 and 2, a cycle
// deeper (with
// the condition on the counter itself, they share at cycles 0 and 1).
// CHECK-LABEL: loopschedule.func_sequential @late_select
// CHECK:         loopschedule.pipeline II = 2 latency = 2
// CHECK:           loopschedule.at 0
// CHECK:             arith.muli {{.*}}loopschedule.binding = [[B:[0-9]+]] : i64
// CHECK:           loopschedule.at 1
// CHECK-NOT:         loopschedule.binding = [[B]] : i64
// CHECK:             arith.muli
// SLACK-LABEL: loopschedule.func_sequential @late_select
// SLACK:         loopschedule.pipeline II = 2 latency = 3
// SLACK:           loopschedule.at 0
// SLACK-NOT:         arith.muli
// SLACK:           loopschedule.at 1
// SLACK-NEXT:        arith.muli {{.*}}loopschedule.binding = [[B:[0-9]+]] : i64
// SLACK:           loopschedule.at 2
// SLACK-NEXT:        arith.muli {{.*}}loopschedule.binding = [[B]] : i64
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
    %p = arith.muli %x, %i {loopschedule.operator = @mul_l1} : i32
    %q = arith.muli %p, %y {loopschedule.operator = @mul_l1} : i32
    %ns = arith.addi %s, %q {loopschedule.operator = @add_comb} : i32
    %ni = arith.addi %i, %c1 {loopschedule.operator = @add_comb} : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline = 2}
  return %r#1 : i32
}
