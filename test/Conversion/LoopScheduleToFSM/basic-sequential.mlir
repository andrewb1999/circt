// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A simple sequential loop that counts from 0 to 10.
module {
  func.func @count() attributes {top} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    loopschedule.frame {
      loopschedule.at 0 {
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
      loopschedule.yield
    }
    return
  }
}

// CHECK: hw.module @count
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: hw.module @loop0
// CHECK: fsm.hw_instance "loop0_fsm_inst" @loop0_fsm
// CHECK: fsm.machine @loop0_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @COND
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @DONE
