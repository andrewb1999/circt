// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// Iter-arg init delivery uses a feedback-register preload instead of a
// combinational first_iter mux: the stage register that carries each
// iter_arg's feedback is loaded with the init value on (a delayed copy
// of) the start pulse, and every consumer reads the register output
// directly. This keeps the 1-bit control select off the datapath — the
// old first_iter mux fanned out across every consumer bit of every
// iter_arg and sat in front of the accumulator carry chain on the
// post-synthesis critical path (doitgen/atax/gesummv at 2 ns).

// CHECK-LABEL: hw.module @preload

// The accumulator add reads the feedback registers directly (no mux
// between the register and the adder).
// CHECK: %[[NEXTI:.+]] = arith.addi %loop0_s0_r0, %c1
// CHECK: %[[PARTIAL:.+]] = arith.addi %loop0_s0_r1, %arg0

// Both feedback registers are preloaded: D input is a mux whose
// true-arm is the init constant, and the clock enable ORs in the start
// pulse. The induction arg feeds the loop condition combinationally, so
// its preload depth is 0 (the raw start pulse, no delay register).
// CHECK-DAG: %loop0_s0_r0 = seq.compreg.ce sym @loop0_s0_r0 %[[D0:.+]], %clk, %[[CE0:.+]] reset %rst
// CHECK-DAG: %[[D0]] = comb.mux %[[START:.+]], %c0{{.*}}, %[[NEXTI]]
// CHECK-DAG: %[[CE0]] = comb.or %{{.+}}, %[[START]]
// CHECK-DAG: %loop0_s0_r1 = seq.compreg.ce sym @loop0_s0_r1 %[[D1:.+]], %clk, %[[CE1:.+]] reset %rst
// CHECK-DAG: %[[D1]] = comb.mux %[[START]], %c0_i32{{.*}}, %[[PARTIAL]]
// CHECK-DAG: %[[CE1]] = comb.or %{{.+}}, %[[START]]

// No per-stage first_iter flops remain.
// CHECK-NOT: first_iter_s

loopschedule.func_sequential @preload(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
          %1:3 = loopschedule.at 0 -> (index, i32, i1) {
            %cond = arith.cmpi ult, %i, %c10 : index
            %next_i = arith.addi %i, %c1 : index
            %partial = arith.addi %acc, %arg0 : i32
            loopschedule.iter_arg_update %acc = %partial : i32
            loopschedule.iter_arg_update %i = %next_i : index
            loopschedule.yield %next_i, %partial, %cond : index, i32, i1
          }
          %2 = loopschedule.at 1 -> i32 {
            loopschedule.yield %1#1 : i32
          }
          loopschedule.terminator condition(%1#2), results(%2) : i32
        }
        loopschedule.yield %pip : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    loopschedule.yield %hp : !loopschedule.handle
  }
  %result = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %r = loopschedule.at 0 -> i32 {
      loopschedule.yield %v : i32
    }
    loopschedule.yield %r : i32
  }
  loopschedule.return %result : i32
}
