// RUN: circt-opt --loopschedule-binding %s | FileCheck %s

// Three concurrent adds tagged for an oplib operator capped at 2 instances.
// The two cycle-0 adds conflict and split onto instances 0 and 1; the
// cycle-1 add is temporally disjoint and shares instance 0.
//
// Operators without a `limit` attribute on their `oplib.operator` entry
// aren't bound (the binder has no cap to enforce) — the plain-mul op
// defined alongside demonstrates this: it's invoked but no
// `loopschedule.binding` attr is stamped.

oplib.library @lib {
  oplib.operator @add_i32_l0 latency<0>, limit<2> {
    oplib.target @arith_addi_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addi_i32 : (i32, i32) -> i32) produce (in %arg0 : i32, in %arg1 : i32) {
      %0 = comb.add %arg0, %arg1 : i32
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_i32_l0 latency<0> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (in %arg0 : i32, in %arg1 : i32) {
      %0 = comb.mul %arg0, %arg1 : i32
      oplib.hw_return %0 : i32
    }
  }
}

// CHECK-LABEL: loopschedule.func_sequential @three_adds
loopschedule.func_sequential @three_adds(%x: i32, %y: i32, %z: i32) -> i32
    attributes {oplib.library = @lib, top} {
  loopschedule.frame {
    loopschedule.at 0 {
      // Two cycle-0 adds: distinct instances.
      // CHECK: arith.addi
      // CHECK-SAME: loopschedule.binding = 0
      %a = arith.addi %x, %y {loopschedule.operator = @add_i32_l0} : i32
      // CHECK: arith.addi
      // CHECK-SAME: loopschedule.binding = 1
      %b = arith.addi %x, %z {loopschedule.operator = @add_i32_l0} : i32
      // A multiply with no declared limit — not stamped with a binding.
      // CHECK: arith.muli
      // CHECK-NOT: loopschedule.binding
      %m = arith.muli %x, %y {loopschedule.operator = @mul_i32_l0} : i32
    }
    loopschedule.at 1 {
      // Cycle-1 add: reuses instance 0 (disjoint from cycle-0 adds).
      // CHECK: arith.addi
      // CHECK-SAME: loopschedule.binding = 0
      %c = arith.addi %x, %y {loopschedule.operator = @add_i32_l0} : i32
    }
  }
  loopschedule.return %x : i32
}
