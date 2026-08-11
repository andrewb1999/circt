// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{enable-pipeline-prearm=true})" %s | FileCheck %s

// Two pipelines launched concurrently from the SAME frame both refuse
// drain-edge pre-arming (gap-9), even though every init is a constant:
// pre-arm timing is reasoned per-launch against a frame with exactly one
// child, and a multi-launch frame couples the launch cycle to its
// sibling's state. Both issue gates stay plain `active` — no
// `or(active, child_start)` may exist for either machine.
//
// CHECK-LABEL: hw.module @two_pipes
// CHECK-NOT: loop0_done_prev
// CHECK-NOT: loop1_done_prev
// CHECK-LABEL: fsm.machine @two_pipes_fsm

loopschedule.func_sequential @two_pipes(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c0_i32 = arith.constant 0 : i32
  %h:2 = loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
    %hp0 = loopschedule.at 0 -> !loopschedule.handle {
      %l0 = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_i32) : (index, i32) -> i32 {
          %1:3 = loopschedule.at 0 -> (index, i32, i1) {
            %cond = arith.cmpi ult, %i, %c10 : index
            %next_i = arith.addi %i, %c1 : index
            %partial = arith.addi %acc, %arg0 : i32
            loopschedule.iter_arg_update %acc = %partial : i32
            loopschedule.iter_arg_update %i = %next_i : index
            loopschedule.yield %next_i, %partial, %cond : index, i32, i1
          }
          loopschedule.terminator condition(%1#2), results(%1#1) : i32
        }
        loopschedule.yield %pip : i32
      }
      loopschedule.yield %l0 : !loopschedule.handle
    }
    %hp1 = loopschedule.at 0 -> !loopschedule.handle {
      %l1 = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 1 iter_args(%j = %c0, %acc2 = %c0_i32) : (index, i32) -> i32 {
          %2:3 = loopschedule.at 0 -> (index, i32, i1) {
            %cond = arith.cmpi ult, %j, %c10 : index
            %next_j = arith.addi %j, %c1 : index
            %partial = arith.addi %acc2, %arg1 : i32
            loopschedule.iter_arg_update %acc2 = %partial : i32
            loopschedule.iter_arg_update %j = %next_j : index
            loopschedule.yield %next_j, %partial, %cond : index, i32, i1
          }
          loopschedule.terminator condition(%2#2), results(%2#1) : i32
        }
        loopschedule.yield %pip : i32
      }
      loopschedule.yield %l1 : !loopschedule.handle
    }
    loopschedule.yield %hp0, %hp1 : !loopschedule.handle, !loopschedule.handle
  }
  %result:2 = loopschedule.frame -> (i32, i32) {
    %v0 = loopschedule.await %h#0 -> i32
    %v1 = loopschedule.await %h#1 -> i32
    loopschedule.yield %v0, %v1 : i32, i32
  } do (%v0: i32, %v1: i32) {
    %r = loopschedule.at 0 -> i32 {
      %s = arith.addi %v0, %v1 : i32
      loopschedule.yield %s : i32
    }
    loopschedule.yield %r, %v1 : i32, i32
  }
  loopschedule.return %result#0 : i32
}
