// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A simple sequential loop that counts from 0 to 10.
module {
  func.func @count() attributes {top} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    loopschedule.step {
      loopschedule.sequential trip_count = 10 iter_args(%i = %c0) : (i32) -> () {
        %cond = arith.cmpi slt, %i, %c10 : i32
        loopschedule.register %cond : i1
      } do {
        %0 = loopschedule.step {
          %next = arith.addi %i, %c1 : i32
          loopschedule.register %next : i32
        } : i32
        loopschedule.terminator iter_args(%0), results() : (i32) -> ()
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
