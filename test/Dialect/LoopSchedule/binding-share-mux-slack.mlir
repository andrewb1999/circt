// RUN: circt-opt --loopschedule-binding="share-mux-delays=0.3,0.3,0.6" %s | FileCheck %s
// RUN: circt-opt --loopschedule-binding %s | FileCheck %s --check-prefix=ANY

// Sharing an instance puts a one-hot mux on each user's operands, on the
// chain the scheduler placed the op on. With share-mux-delays (by users,
// from 2: here 0.3 ns for 2 and 3 users, one level of AND-OR, and 0.6 for
// 4, two), an op shares an instance only if its chain slack
// (`loopschedule.chain_slack`, ns, from the scheduler) takes the mux the
// instance's users need; the tightest user that does not leaves, and the
// rest are bound again. The slack is consumed: no op keeps it. Without
// share-mux-delays every op that can share does.

oplib.library @lib {
  oplib.operator @fadd_l0 latency<0>, incDelay<2.0>, outDelay<2.0> {
    oplib.target @arith_addf(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.addi"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_addf : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "fadd_hw_inst" @fadd_l0(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, din0: %arg0 : i32, din1: %arg1 : i32) -> (dout: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @fadd_l0(in %clk: i1, in %reset: i1, in %ce: i1, in %din0: i32, in %din1: i32, out dout: i32)

// Three serialized adds: the two with 0.5 ns to spare share one instance
// (one level of mux, 0.3 ns); the one with 0.1 keeps its own, an instance
// id past the shared ones (every share-mode op is bound).
// CHECK-LABEL: loopschedule.func_sequential @three
// CHECK:         arith.addi %{{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @fadd_l0}
// CHECK:         arith.addi %{{.*}} {loopschedule.binding = 1 : i64, loopschedule.operator = @fadd_l0}
// CHECK:         arith.addi %{{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @fadd_l0}
// CHECK-NOT:     chain_slack
// ANY-LABEL:   loopschedule.func_sequential @three
// ANY-COUNT-3:   loopschedule.binding = 0
// ANY-NOT:       chain_slack
loopschedule.func_sequential @three(%x: i32, %y: i32) -> i32
    attributes {oplib.library = @lib, top} {
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %x, %y {loopschedule.chain_slack = 0.5 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %x, %x {loopschedule.chain_slack = 0.1 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %y, %y {loopschedule.chain_slack = 0.5 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.return %x : i32
}

// Four adds with 0.5 ns each: four users need 0.6 ns, so one
// leaves and the other three share an instance behind one level.
// CHECK-LABEL: loopschedule.func_sequential @four
// CHECK:         arith.addi %{{.*}} {loopschedule.binding = 1 : i64, loopschedule.operator = @fadd_l0}
// CHECK-COUNT-3: loopschedule.binding = 0 : i64, loopschedule.operator = @fadd_l0
loopschedule.func_sequential @four(%x: i32, %y: i32) -> i32
    attributes {oplib.library = @lib, top} {
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %x, %y {loopschedule.chain_slack = 0.5 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %x, %x {loopschedule.chain_slack = 0.5 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %y, %y {loopschedule.chain_slack = 0.5 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      %v = arith.addi %y, %x {loopschedule.chain_slack = 0.5 : f32, loopschedule.operator = @fadd_l0} : i32
      loopschedule.yield
    }
  }
  loopschedule.return %x : i32
}
