// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// Exit-tail compression for a SOLO SEQUENTIAL last child of a
// void-returning function: module done cuts through on the child's done
// (advance edge) instead of waiting for the Moore DONE state, and the
// Moore pulse one cycle later is masked through `early_done_sent` so done
// is a SINGLE pulse per invocation. The testbench attributes memory
// writes by counting done cycles, so an unmasked second pulse mis-files
// every transaction after the first in a back-to-back run.
module {
  loopschedule.func_sequential @seq_tail_void(%arg0: memref<32xi32>) attributes {top} {
    %c42 = arith.constant 42 : i32
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %c32_i6 = arith.constant 32 : i6
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 32 iter_args(%i = %c0_i6) : (i6) -> () {
            %cond, %next = loopschedule.frame -> (i1, i6) {
              %r:2 = loopschedule.at 0 -> (i1, i6) {
                %c = arith.cmpi ult, %i, %c32_i6 : i6
                loopschedule.store %c42, %arg0[%i : i6] : memref<32xi32>
                %n = arith.addi %i, %c1_i6 : i6
                loopschedule.iter_arg_update %i = %n : i6
                loopschedule.yield %c, %n : i1, i6
              }
              loopschedule.yield %r#0, %r#1 : i1, i6
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

// CHECK-LABEL: hw.module @seq_tail_void
// tx_in_flight (and so `ready`) clears on the REGISTERED Moore done, NOT
// the cut-through: the early term carries the child-done chain, and
// putting that cone in front of tx_in_flight's register measurably cost
// Fmax on deep nests. The early term reaches only the done PORT.
// CHECK: %[[NMOORE:.+]] = comb.xor %{{.+}}#0, %true{{.*}} : i1
// CHECK: %[[HOLD:.+]] = comb.and %tx_in_flight, %[[NMOORE]] : i1
// The early term: frame_running & ~child_start & child done.
// CHECK: %[[NSTART:.+]] = comb.xor %{{.+}}#1, %true{{.*}} : i1
// CHECK: %[[INWAIT:.+]] = comb.and %{{.+}}#2, %[[NSTART]] : i1
// CHECK: %[[EARLY:.+]] = comb.and %[[INWAIT]], %loop0_inst.done : i1
// The mask register and the single-pulse done:
// CHECK: %early_done_sent = seq.compreg sym @early_done_sent %[[EARLY]]
// CHECK: %[[NPREV:.+]] = comb.xor %early_done_sent, %true{{.*}} : i1
// CHECK: %[[MOORE:.+]] = comb.and %{{.+}}#0, %[[NPREV]] : i1
// CHECK: %[[DONE:.+]] = comb.or %[[MOORE]], %[[EARLY]] : i1
// CHECK: hw.output {{.+}}, %[[DONE]] :

// The function FSM keeps its historical shape — no start arc in DONE:
// ready rises the cycle after DONE (Moore-timed clear), when the machine
// is back in IDLE.
// CHECK-LABEL: fsm.machine @seq_tail_void_fsm
// CHECK: fsm.state @DONE
// CHECK-NOT: fsm.transition @FRAME_0
// CHECK: fsm.transition @IDLE
