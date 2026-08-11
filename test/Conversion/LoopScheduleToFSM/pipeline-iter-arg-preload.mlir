// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{enable-pipeline-prearm=true})" %s | FileCheck %s

// Iter-arg init delivery pre-arms the feedback registers at the
// PREVIOUS invocation's drain instead of the launch edge: the D-mux
// select is the drain pulse (doneComb gated off the launch cycle), the
// clock enable ORs that pulse in, and the register's RESET VALUE is the
// init constant so the first-ever invocation is pre-armed out of reset.
// Every consumer reads the register output directly (no combinational
// first_iter mux on the datapath — the old mux sat in front of the
// accumulator carry chain on the doitgen/atax/gesummv 2 ns paths), and
// because the inits already sit in the registers through the idle
// window, the first issue fires IN the launch cycle: stage-0 CE is
// (active | child_start) & cond, closing the historical dead launch
// state.

// CHECK-LABEL: hw.module @preload

// The accumulator add reads the feedback registers directly (no mux
// between the register and the adder).
// CHECK: %[[NEXTI:.+]] = arith.addi %loop0_s0_r0, %c1
// CHECK: %[[PARTIAL:.+]] = arith.addi %loop0_s0_r1, %arg0

// Both feedback registers pre-arm on ONE shared drain pulse and reset
// to their init constants.
// CHECK-DAG: %loop0_s0_r0 = seq.compreg.ce sym @loop0_s0_r0 %[[D0:.+]], %clk, %[[CE0:.+]] reset %rst, %c0
// CHECK-DAG: %[[D0]] = comb.mux %[[PULSE:.+]], %c0, %[[NEXTI]]
// CHECK-DAG: %[[CE0]] = comb.or %{{.+}}, %[[PULSE]]
// CHECK-DAG: %loop0_s0_r1 = seq.compreg.ce sym @loop0_s0_r1 %[[D1:.+]], %clk, %[[CE1:.+]] reset %rst, %c0_i32
// CHECK-DAG: %[[D1]] = comb.mux %[[PULSE]], %c0_i32, %[[PARTIAL]]
// CHECK-DAG: %[[CE1]] = comb.or %{{.+}}, %[[PULSE]]

// The drain pulse is a ONE-SHOT (`done_prev` limits it to the first
// drain cycle), and the final iteration's result latches into a HOLD
// register on that same edge — consumers read the hold, never the
// re-armed feedback register.
// CHECK-DAG: %loop0_done_prev = seq.compreg sym @loop0_done_prev
// CHECK-DAG: %loop0_result_hold_0 = seq.compreg.ce sym @loop0_result_hold_0 %loop0_s1_r0, %clk, %[[PULSE]] reset %rst

// The issue gate ORs the machine's REGISTERED issue_arm (an
// fsm.variable equal to the child_start decode but read as a flop Q —
// the combinational decode on this cone measured −164 ps on the vadd
// class) into `active`, so the launch cycle issues stage 0. Safe under
// stall: every stall term is stage-occupancy-gated, so stall is 0 while
// all stage CEs are 0.
// CHECK-DAG: comb.or %loop0_active, %{{.+}}
// CHECK-DAG: fsm.variable "issue_arm_0"

// No per-stage first_iter flops and no start-pulse delay chain remain.
// CHECK-NOT: first_iter_s
// CHECK-NOT: preload_start_d

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
