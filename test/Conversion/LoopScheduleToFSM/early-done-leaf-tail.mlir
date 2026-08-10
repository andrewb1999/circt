// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// Exit-tail compression when the LAST stateful entry is a LEAF FRAME
// (filter_compact's shape: a pipeline, then a final frame that stores a
// scalar). The frame's last cycle gate IS the completion, so done cuts
// through on it — sound because the frame's results are dead (there are
// none) and the function returns nothing. No ~child_start staleness term
// is needed: a leaf has no child.
module {
  loopschedule.func_sequential @leaf_tail(%arg0: memref<32xi32>) attributes {top} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0_i32 = arith.constant 0 : i32
    %c7 = arith.constant 7 : i32
    %c0_i6 = arith.constant 0 : i6
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %hp = loopschedule.at 0 -> !loopschedule.handle {
        %hp_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.pipeline II = 1 iter_args(%i = %c0) : (index) -> () {
            %1:2 = loopschedule.at 0 -> (index, i1) {
              %cond = arith.cmpi ult, %i, %c10 : index
              %next_i = arith.addi %i, %c1 : index
              loopschedule.iter_arg_update %i = %next_i : index
              loopschedule.yield %next_i, %cond : index, i1
            }
            loopschedule.terminator condition(%1#1), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %hp_launch : !loopschedule.handle
      }
      loopschedule.yield %hp : !loopschedule.handle
    }
    loopschedule.frame {
      loopschedule.await %h
      loopschedule.yield
    } do {
      loopschedule.at 0 {
        loopschedule.store %c7, %arg0[%c0_i6 : i6] : memref<32xi32>
        loopschedule.yield
      }
      loopschedule.yield
    }
    loopschedule.return
  }
}

// CHECK-LABEL: hw.module @leaf_tail
// The early term is the leaf frame's own cycle gate (an FSM output, not a
// child done), masked the same way:
// CHECK: %early_done_sent = seq.compreg sym @early_done_sent %[[EARLY:.+]], %clk
// CHECK: %[[NPREV:.+]] = comb.xor %early_done_sent, %true{{.*}} : i1
// CHECK: %[[MOORE:.+]] = comb.and %{{.+}}#0, %[[NPREV]] : i1
// CHECK: %[[DONE:.+]] = comb.or %[[MOORE]], %[[EARLY]] : i1
// CHECK: hw.output {{.+}}, %[[DONE]] :
