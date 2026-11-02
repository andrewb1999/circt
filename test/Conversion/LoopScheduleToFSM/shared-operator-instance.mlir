// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// Two multiplies stamped with the same (operator, loopschedule.binding)
// pair by the binding pass, in two SEQUENTIAL frames of one loop body —
// the FSM lowering must materialize a single extern multiplier instance
// whose operand ports are muxed between the users under their activation
// gates, instead of one instance per op.

oplib.library @lib {
  oplib.operator @mul_i32_l4 latency<4>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_pipe_i32_hw_inst" @int_mul_pipe_i32_l4(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @int_mul_pipe_i32_l4(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32}) attributes {verilogName = "int_mul_pipe_i32_l4"}

loopschedule.func_sequential @shared(%m: memref<8xi32>)
      attributes {oplib.library = @lib, top} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8 iter_args(%i = %c0) : (i32) -> () {
            %cond:2 = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c8 : i32
                // First user of the shared multiplier.
                %p = arith.muli %i, %c8 {loopschedule.operator = @mul_i32_l4, loopschedule.binding = 0 : i64} : i32
                loopschedule.yield %c, %p : i1, i32
              }
              loopschedule.yield %r#0, %r#1 : i1, i32
            }
            %f2 = loopschedule.frame -> (i32) {
              loopschedule.await
            } do {
              %r = loopschedule.at 0 -> i32 {
                // Second user: a later frame of the same loop body —
                // guaranteed non-concurrent, same instance.
                %q = arith.muli %cond#1, %c1 {loopschedule.operator = @mul_i32_l4, loopschedule.binding = 0 : i64} : i32
                %n = arith.addi %i, %c1 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield %q : i32
              }
              loopschedule.yield %r : i32
            }
            loopschedule.terminator condition(%cond#0), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %lh_launch : !loopschedule.handle
      }
      loopschedule.yield %lh : !loopschedule.handle
    }
    loopschedule.frame {
      loopschedule.await %h
      loopschedule.yield
    } do {
      loopschedule.yield
    }
    loopschedule.return
  }

// One shared instance, now living in the single function module (loops no
// longer get their own hw.module): each operand port is the one-hot AND-OR
// of the users' operands under their frame gates, the first user's standing
// while no other gate is high, `(~g1 & a0) | (g1 & a1)`, with a simulation
// assertion that the gates are one-hot, and only ONE multiplier is
// instantiated.
// CHECK: hw.module @shared
// CHECK: hw.instance "mul_pipe_i32_hw_inst_0" @int_mul_pipe_i32_l4({{.*}}lhs: %[[LHS:[0-9]+]]: i32, rhs: %[[RHS:[0-9]+]]: i32)
// CHECK-DAG: %[[A1:.+]] = comb.and %[[G1:.+]], %loop0_frame0_r1 : i32
// CHECK-DAG: %[[G1]] = comb.replicate %[[S1:.+]] : (i1) -> i32
// CHECK-DAG: %[[LATER:.+]] = comb.or %[[S1]] : i1
// CHECK-DAG: %[[NOT1:.+]] = comb.xor %[[LATER]], %true{{.*}} : i1
// CHECK-DAG: %[[G0:.+]] = comb.replicate %[[NOT1]] : (i1) -> i32
// CHECK-DAG: %[[A0:.+]] = comb.and %[[G0]], %{{.+}} : i32
// CHECK-DAG: %[[LHS]] = comb.or %[[A0]], %[[A1]] : i32
// CHECK-DAG: %[[RHS]] = comb.or %{{.+}}, %{{.+}} : i32
// CHECK-DAG: verif.clocked_assert {{.+}} label "shared_operator_one_user"
// CHECK-NOT: hw.instance "mul_pipe_i32_hw_inst
