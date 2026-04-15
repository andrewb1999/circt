// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A simple sequential loop that counts from 0 to 10.
module {
  func.func @count() attributes {top} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    loopschedule.step {
      loopschedule.sequential trip_count = 10 iter_args(%i = %c0) : (i32) -> () {
        %0:2 = loopschedule.step {
          %cond = arith.cmpi slt, %i, %c10 : i32
          %next = arith.addi %i, %c1 : i32
          loopschedule.iter_arg_update %i = %next : i32
          loopschedule.register %next, %cond : i32, i1
        } : i32, i1
        loopschedule.terminator condition(%0#1), results()
      }
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
// CHECK: fsm.state @STEP_0
// CHECK: fsm.state @DONE
