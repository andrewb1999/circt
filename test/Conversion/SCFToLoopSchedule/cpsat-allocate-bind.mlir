// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 scheduler=cpsat cpsat-workers=1 share-mux-delays=0.3,0.3,0.6})" %s | FileCheck %s

// The CP-SAT scheduler chooses each op's operator among its candidates and
// binds the ops of a shareable operator (an extern instance) to instances,
// the mux an instance's users put on its operands in their chains (0.3 ns
// for two users here). It trades the function's area against its cycles,
// area^(1 - t) x cycles^t (t = 0.5 by default: their product).

oplib.library @lib {
  oplib.operator @mul_l0 latency<0>, incDelay<2.5>, outDelay<2.5> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_muli_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "mul" @mul_l0(lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_slow_l0 latency<0>, incDelay<2.8>, outDelay<2.8> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_muli_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "mul" @mul_slow_l0(lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_l1 latency<1>, incDelay<1.0>, outDelay<0.5> attributes {amc.area_dsp = 1 : i64} {
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

hw.module.extern @mul_l0(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @mul_slow_l0(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @mul_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

// The combinational multiply takes the mux (2.5 + 0.3 <= 3.0): the two,
// chained through a register in any case, share one instance.
// CHECK-LABEL: loopschedule.func_sequential @fits
// CHECK:         arith.muli {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l0}
// CHECK:         arith.muli {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l0}
func.func @fits(%a: i32, %b: i32, %c: i32) -> i32 attributes {oplib.library = @lib} {
  %x = arith.muli %a, %b {loopschedule.operator = @mul_l0, loopschedule.candidates = [@mul_l0, @mul_l1]} : i32
  %y = arith.muli %x, %c {loopschedule.operator = @mul_l0, loopschedule.candidates = [@mul_l0, @mul_l1]} : i32
  return %y : i32
}

// A slower combinational multiply leaves no room for the mux (2.8 + 0.3 >
// 3.0): shared, the multiplies take the pipelined core (its input path 1.0
// + 0.3), one instance where the combinational core needed two, for a cycle
// more (half the area for a third more cycles).
// CHECK-LABEL: loopschedule.func_sequential @slow
// CHECK:         arith.muli {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l1}
// CHECK:         arith.muli {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l1}
func.func @slow(%a: i32, %b: i32, %c: i32) -> i32 attributes {oplib.library = @lib} {
  %x = arith.muli %a, %b {loopschedule.operator = @mul_slow_l0, loopschedule.candidates = [@mul_slow_l0, @mul_l1]} : i32
  %y = arith.muli %x, %c {loopschedule.operator = @mul_slow_l0, loopschedule.candidates = [@mul_slow_l0, @mul_l1]} : i32
  return %y : i32
}
