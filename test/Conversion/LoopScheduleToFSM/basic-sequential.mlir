// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// A simple sequential loop that counts from 0 to 10.
// The loop is wrapped in a top-level frame via `launch` (new shape).
module {
  loopschedule.func_sequential @count() attributes {top} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 10 iter_args(%i = %c0) : (i32) -> () {
            %cond, %next = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c10 : i32
                %n = arith.addi %i, %c1 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield %c, %n : i1, i32
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

// The loop inlines into a single per-function machine: no hw.module @loop0,
// just prefixed loop0_* states inside @count_fsm.
// CHECK: hw.module @count
// CHECK: fsm.hw_instance "count_fsm_inst" @count_fsm
// CHECK: fsm.machine @count_fsm
// CHECK: fsm.state @IDLE
// The loop condition is pure comb of the iter_args, so the COND bypass
// applies: the entry frame enters loop0_FRAME_0 directly and no loop0_COND
// state is emitted.
// CHECK: fsm.state @FRAME_0
// CHECK-NOT: fsm.state @loop0_COND
// CHECK: fsm.state @loop0_FRAME_0
// CHECK-NOT: fsm.state @loop0_COND
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0
