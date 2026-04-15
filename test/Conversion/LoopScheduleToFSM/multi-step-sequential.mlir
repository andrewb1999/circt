// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// Two top-level steps, each containing a sequential loop.
// First loop initializes memory, second loop reads/modifies it.
module {
  func.func @two_loops(%arg0: memref<8xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c42_i32 = arith.constant 42 : i32
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c8_i4 = arith.constant -8 : i4
    loopschedule.step {
      loopschedule.sequential trip_count = 8 iter_args(%i = %c0_i4) : (i4) -> () {
        %0:2 = loopschedule.step {
          %cond = arith.cmpi ult, %i, %c8_i4 : i4
          loopschedule.store %c42_i32, %arg0[%i : i4] : memref<8xi32>
          %next = arith.addi %i, %c1_i4 : i4
          loopschedule.iter_arg_update %i = %next : i4
          loopschedule.register %next, %cond : i4, i1
        } : i4, i1
        loopschedule.terminator condition(%0#1), results()
      }
    }
    loopschedule.step {
      loopschedule.sequential trip_count = 8 iter_args(%j = %c0_i4) : (i4) -> () {
        %0:2 = loopschedule.step {
          %cond = arith.cmpi ult, %j, %c8_i4 : i4
          %val = loopschedule.load %arg0[%j : i4] : memref<8xi32>
          %inc = arith.addi %val, %c0_i32 : i32
          loopschedule.store %inc, %arg0[%j : i4] : memref<8xi32>
          %next = arith.addi %j, %c1_i4 : i4
          loopschedule.iter_arg_update %j = %next : i4
          loopschedule.register %next, %cond : i4, i1
        } : i4, i1
        loopschedule.terminator condition(%0#1), results()
      }
    }
    return
  }
}

// CHECK: hw.module @two_loops
// CHECK: fsm.hw_instance "two_loops_fsm_inst" @two_loops_fsm
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: hw.instance "loop1_inst" @loop1

// CHECK: fsm.machine @two_loops_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @STEP_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @STEP_1
// CHECK: fsm.state @WAIT_1
// CHECK: fsm.state @DONE

// CHECK: hw.module @loop0
// CHECK: fsm.machine @loop0_fsm

// CHECK: hw.module @loop1
// CHECK: fsm.machine @loop1_fsm
