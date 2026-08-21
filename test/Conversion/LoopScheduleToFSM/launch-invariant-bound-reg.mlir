// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A runtime loop bound COMPUTED from a scalar argument — the matmul
// `ceil(n/16)` shape: (n-1)>>4 + 1 is pure arithmetic over a
// transaction-stable input, but left combinational it rides the
// per-iteration continue-compare straight into the machine's transition
// guard and the replicated one-hot state bit (the measured
// 10xLOOKAHEAD8 post-route worst path). The lowering registers the
// LAUNCH-INVARIANT FRONTIER of the condition cone: the computed bound
// lands in a free-running `_bound_reg` and the guard compare reads the
// flop. Zero latency: the register's D settles during the start cycle
// and its first consumer is a FRAME-state guard, one cycle later.

// The bound chain feeds a free-running register and the guard compares
// read the FLOP; cond_next reaches the machine as the registered exit
// flag (loaded on the same edges the iter-args move).
// CHECK: arith.subi %arg0
// CHECK: fsm.hw_instance "bounded_fsm_inst" @bounded_fsm(%start, %{{.+}}, %loop0_cond_next_reg, %{{.+}})
// CHECK: arith.cmpi slt, %{{.+}}, %loop0_bound_reg_0 : i32
// CHECK: %loop0_bound_reg_0 = seq.compreg sym @loop0_bound_reg_0 %{{.+}}, %clk reset %rst
// CHECK: %loop0_cond_next_reg = seq.compreg.ce sym @loop0_cond_next_reg
// CHECK: fsm.machine @bounded_fsm
// CHECK: fsm.state @loop0_FRAME_0

module {
  loopschedule.func_sequential @bounded(%n: i32) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    // ceil(n/16) for positive n: depth-3 stable chain over the scalar.
    %nm1 = arith.subi %n, %c1 : i32
    %sh = arith.shrsi %nm1, %c4 : i32
    %bound = arith.addi %sh, %c1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential iter_args(%i = %c0) : (i32) -> () {
            %cond, %next = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %bound : i32
                %nx = arith.addi %i, %c1 : i32
                loopschedule.iter_arg_update %i = %nx : i32
                loopschedule.yield %c, %nx : i1, i32
              }
              loopschedule.yield %r#0, %r#1 : i1, i32
            }
            loopschedule.terminator condition(%cond), results()
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
}
