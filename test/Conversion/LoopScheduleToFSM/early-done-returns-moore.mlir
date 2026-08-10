// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// NEGATIVE: a function that RETURNS a value keeps the Moore done path —
// module result registers latch on the exit edge, so a caller sampling
// result0 on a cut-through done would read the pre-latch value. No
// early_done_sent register may exist anywhere in the output.
module {
  loopschedule.func_sequential @seq_tail_returns() -> i32 attributes {top} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          %sum = loopschedule.sequential trip_count = 10 iter_args(%i = %c0) : (i32) -> i32 {
            %cond, %next = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c10 : i32
                %n = arith.addi %i, %c1 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield %c, %n : i1, i32
              }
              loopschedule.yield %r#0, %r#1 : i1, i32
            }
            loopschedule.terminator condition(%cond), results(%next) : i32
          }
          loopschedule.yield %sum : i32
        }
        loopschedule.yield %lh_launch : !loopschedule.handle
      }
      loopschedule.yield %lh : !loopschedule.handle
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
}

// CHECK: hw.module @seq_tail_returns
// CHECK-NOT: early_done_sent
