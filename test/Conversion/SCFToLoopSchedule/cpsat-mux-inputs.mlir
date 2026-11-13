// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule-cpsat{cycle-time-ns=3.0 cpsat-workers=1 share-mux-delays=0.3 cpsat-mux-input-luts=20})" %s | FileCheck %s

// A shared instance's operand mux costs `cpsat-mux-input-luts` per input
// past the first, per 32 bits of operand, an input being a value its users
// put on the operand: users with one value there (a constant, a function
// argument, one op's result) feed it one input. Here an instance takes 30
// LUTs and its mux 20 per extra input; the two adds are a cycle apart in
// any case, so area alone decides whether they share.

oplib.library @lib {
  oplib.operator @add_l1 latency<1>, incDelay<1.0>, outDelay<0.5> attributes {amc.area_lut = 30 : i64} {
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

hw.module.extern @add_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

// Distinct values on both operands: the mux would take 40 LUTs, more than
// the instance it saves.
// CHECK-LABEL: loopschedule.func_sequential @distinct
// CHECK:         arith.addi {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @add_l1}
// CHECK:         arith.addi {{.*}} {loopschedule.binding = 1 : i64, loopschedule.operator = @add_l1}
func.func @distinct(%a: i32, %b: i32, %c: i32) -> i32 attributes {oplib.library = @lib} {
  %x = arith.addi %a, %b {loopschedule.operator = @add_l1, loopschedule.candidates = [@add_l1]} : i32
  %y = arith.addi %x, %c {loopschedule.operator = @add_l1, loopschedule.candidates = [@add_l1]} : i32
  return %y : i32
}

// One argument on the second operand of both: 20 LUTs, and they share.
// CHECK-LABEL: loopschedule.func_sequential @common_argument
// CHECK:         arith.addi {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @add_l1}
// CHECK:         arith.addi {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @add_l1}
func.func @common_argument(%a: i32, %c: i32) -> i32 attributes {oplib.library = @lib} {
  %x = arith.addi %a, %c {loopschedule.operator = @add_l1, loopschedule.candidates = [@add_l1]} : i32
  %y = arith.addi %x, %c {loopschedule.operator = @add_l1, loopschedule.candidates = [@add_l1]} : i32
  return %y : i32
}

// Two constants of one value are one input too.
// CHECK-LABEL: loopschedule.func_sequential @common_constant
// CHECK:         arith.addi {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @add_l1}
// CHECK:         arith.addi {{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @add_l1}
func.func @common_constant(%a: i32) -> i32 attributes {oplib.library = @lib} {
  %k0 = arith.constant 7 : i32
  %k1 = arith.constant 7 : i32
  %x = arith.addi %a, %k0 {loopschedule.operator = @add_l1, loopschedule.candidates = [@add_l1]} : i32
  %y = arith.addi %x, %k1 {loopschedule.operator = @add_l1, loopschedule.candidates = [@add_l1]} : i32
  return %y : i32
}
