// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A pipeline whose terminator condition is produced by STAGE 1 (the
// find_first shape: the exit predicate needs the loaded data, so the
// condition of iteration k resolves one cycle after its stage 0 ran).
//
// The late-condition contract this pins, all three clauses of which were
// confirmed broken against the pre-kill-chain lowering:
//
//   * LAUNCH GATE IDLE-DEFAULT — `activeCE`/`holdActive` gate on
//     `!stageCE[condStage] | cond`, not on `cond` alone: while no live
//     iteration occupies the condition stage the cone is reading idle stage
//     registers (garbage for any nonzero init), so launches proceed and the
//     epilogue must not trigger. Before the fix both gated on the raw cone
//     and survived only when reset values coincided with the inits.
//   * KILL CHAIN — the store's enable at the condition stage carries the
//     condition combinationally, so the one speculative iteration launched
//     past the loop's first decided-false (the "ghost") cannot commit.
//     Before the fix the enable was `stageCE & predicate` alone and a
//     double match returned the SECOND index (the wrong-answer scenario).
//   * A LATE-UPDATED ITER_ARG (found: updated and read at stage 1) resolves
//     to a real feedback register, not a combinational self-loop.
//
// The behavioral proof (all exit positions, ghost squash, zero-trip) is
// test/integration/while-early-exit-pipeline.py in the AMC tree.

// CHECK-LABEL: hw.module @ff_while

// The launch gate: condEffective = !condStageOccupied | cond, ANDed with
// active for both holdActive and activeCE.
// CHECK:      %[[NOTOCC:.+]] = comb.xor %loop0_ce_stage_1, %true
// CHECK-NEXT: %[[CONDEFF:.+]] = comb.or %[[NOTOCC]], %[[COND:.+]] :
// CHECK-NEXT: %[[EFFGATE:.+]] = comb.and %[[GCE1:.+]], %[[COND]] :

// The condition cone: (i < n) via the stage-0 register, (found == 0) via the
// found FEEDBACK REGISTER — a compreg, not a self-referential mux.
// CHECK:      %[[LT:.+]] = arith.cmpi slt, %loop0_s0_r2, %arg2
// CHECK-NEXT: %[[NF:.+]] = arith.cmpi eq, %loop0_s1_r1, %c0_i32
// CHECK-NEXT: %[[COND]] = arith.andi %[[LT]], %[[NF]]

// The ghost squash: the store enable is effectGate (stage-1 CE AND cond)
// AND the match predicate.
// CHECK:      %[[MATCH:.+]] = arith.cmpi eq, %mem0_rd_data, %arg3
// CHECK:      comb.and %[[EFFGATE]], %[[MATCH]]

// The found feedback register exists (late-updated iter_arg).
// CHECK: seq.compreg.ce sym @loop0_s1_r1

module {
  loopschedule.func_sequential @ff_while(%A: memref<16xi32>, %out: memref<1xi32>, %n: i32, %target: i32) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.pipeline II = 1 latency = 2 iter_args(%i = %c0_i32, %found = %c0_i32) : (i32, i32) -> () {
            // Stage 0: issue the read, advance i. No observable effects.
            %s0:3 = loopschedule.at 0 -> (i32, i32, i32) {
              %v = loopschedule.load %A[%i : i32] : memref<16xi32>
              %ni = arith.addi %i, %c1_i32 : i32
              loopschedule.iter_arg_update %i = %ni : i32
              loopschedule.yield %ni, %v, %i : i32, i32, i32
            }
            // Stage 1: the condition of THIS iteration on its entry values
            // (i travels via the stage register, found reads the feedback
            // wire it updates this same stage), the match, and the
            // predicated store.
            %s1:2 = loopschedule.at 1 -> (i1, i32) {
              %lt = arith.cmpi slt, %s0#2, %n : i32
              %fz = arith.cmpi eq, %found, %c0_i32 : i32
              %cond = arith.andi %lt, %fz : i1
              %match = arith.cmpi eq, %s0#1, %target : i32
              %nf = arith.select %match, %c1_i32, %found : i32
              loopschedule.iter_arg_update %found = %nf : i32
              loopschedule.if %match {
                loopschedule.store %s0#2, %out[%c0_i32 : i32] : memref<1xi32>
              }
              loopschedule.yield %cond, %nf : i1, i32
            }
            loopschedule.terminator condition(%s1#0), results()
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
