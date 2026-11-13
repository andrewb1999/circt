// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule-cpsat{cycle-time-ns=3.0 cpsat-workers=1 share-mux-delays=0.3})" %s 2>&1 | FileCheck %s --check-prefixes=CHECK,CPSAT
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule{cycle-time-ns=3.0 share-mux-delays=0.3})" %s 2>&1 | FileCheck %s

// A requested II is a strict constraint under either scheduler, and takes
// precedence over the clock. Here the sum's recurrence runs through two 2 ns
// adds, which II 1 closes only in one cycle, past the 3 ns clock: no
// schedule within the clock exists, and the CP-SAT scheduler keeps the
// simplex's, pinned in its function's model, as the simplex takes it (the
// warning that it misses the clock is the simplex's too).

oplib.library @lib {
  oplib.operator @add_slow latency<0>, incDelay<2.0>, outDelay<2.0> {
    oplib.target @arith_addi_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "add" @add_slow(lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_slow latency<0>, incDelay<2.8>, outDelay<2.8> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_muli_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "mul" @mul_slow(lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @mul_slow(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})
hw.module.extern @add_slow(in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

// CHECK: warning: the pipelined schedule (II = 1) misses the clock here
// CHECK-LABEL: loopschedule.func_sequential @strict_ii
// CHECK:         loopschedule.pipeline II = 1
func.func @strict_ii(%x: i32, %y: i32) -> i32 attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i16
  %c1 = arith.constant 1 : i16
  %c16 = arith.constant 16 : i16
  %s0 = arith.constant 0 : i32
  %r:2 = scf.while (%i = %c0, %s = %s0) : (i16, i32) -> (i16, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i16
    scf.condition(%cond) %i, %s : i16, i32
  } do {
  ^bb0(%i: i16, %s: i32):
    %a = arith.addi %s, %x {loopschedule.operator = @add_slow} : i32
    %b = arith.addi %a, %y {loopschedule.operator = @add_slow} : i32
    %ni = arith.addi %i, %c1 : i16
    scf.yield %ni, %b : i16, i32
  } attributes {hls.pipeline = 1 : i64}
  return %r#1 : i32
}

// A pinned region's chains are the simplex's, which binding shares an
// instance within: the clock does not bind the region in the CP-SAT model,
// but a shared instance's mux stays within each user's slack there. The two
// multiplies (2.8 ns, in cycles 0 and 1 of the II-2 pipeline, whose
// recurrence through three adds closes only past the clock) would share an
// instance for its DSP, but the mux (0.3 ns) does not fit their 0.2 ns.
// CHECK-LABEL: loopschedule.func_sequential @pinned_mux
// CHECK:         loopschedule.pipeline II = 2
// CPSAT:           arith.muli {{.*}} {loopschedule.binding = [[B:[01]]] : i64, loopschedule.operator = @mul_slow}
// CPSAT-NOT:       loopschedule.binding = [[B]] : i64, loopschedule.operator = @mul_slow
// CPSAT:           arith.muli {{.*}} {loopschedule.binding = {{[01]}} : i64, loopschedule.operator = @mul_slow}
func.func @pinned_mux(%x: i32, %y: i32) -> (i32, i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %r:3 = scf.while (%i = %c0, %s = %c0, %t = %c0) : (i32, i32, i32) -> (i32, i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s, %t : i32, i32, i32
  } do {
  ^bb0(%i: i32, %s: i32, %t: i32):
    %a = arith.addi %s, %x {loopschedule.operator = @add_slow} : i32
    %b = arith.addi %a, %y {loopschedule.operator = @add_slow} : i32
    %d = arith.addi %b, %i {loopschedule.operator = @add_slow} : i32
    %m = arith.muli %i, %x {loopschedule.operator = @mul_slow} : i32
    %n = arith.muli %m, %y {loopschedule.operator = @mul_slow} : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %d, %n : i32, i32, i32
  } attributes {hls.pipeline = 2 : i64}
  return %r#1, %r#2 : i32, i32
}
