// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A sequential loop whose body has THREE steps:
//   step 0 — computes the loop condition (regular step)
//   step 1 — contains a child sequential loop (wait step #0)
//   step 2 — contains a second child sequential loop (wait step #1)
//
// This exercises the unified FSM emission with multiple "wait" steps in a
// single parent loop: createSequentialFSM must produce two child_start
// outputs, two child_done inputs, two post_active outputs, and matching
// WAIT_1/POST_1 and WAIT_2/POST_2 states.
module {
  func.func @two_children(%arg0: memref<4xi32>, %arg1: memref<4xi32>)
      attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i4 = arith.constant 0 : i4
    %c1_i4 = arith.constant 1 : i4
    %c2_i4 = arith.constant 2 : i4
    %c42_i32 = arith.constant 42 : i32
    %c43_i32 = arith.constant 43 : i32
    loopschedule.step {
      loopschedule.sequential trip_count = 5 iter_args(%i = %c0_i32) : (i32) -> () {
        // Step 0: compute loop condition + iter_arg next.
        %0:2 = loopschedule.step {
          %cond = arith.cmpi slt, %i, %c5_i32 : i32
          %next = arith.addi %i, %c1_i32 : i32
          loopschedule.register %next, %cond : i32, i1
        } : i32, i1
        // Step 1: launch first child loop.
        loopschedule.step {
          loopschedule.sequential trip_count = 2 iter_args(%j = %c0_i4) : (i4) -> () {
            %1:2 = loopschedule.step {
              %cj = arith.cmpi ult, %j, %c2_i4 : i4
              loopschedule.store %c42_i32, %arg0[%j : i4] : memref<4xi32>
              %nj = arith.addi %j, %c1_i4 : i4
              loopschedule.register %nj, %cj : i4, i1
            } : i4, i1
            loopschedule.terminator condition(%1#1), iter_args(%1#0), results() : (i4) -> ()
          }
          loopschedule.register
        }
        // Step 2: launch second child loop.
        loopschedule.step {
          loopschedule.sequential trip_count = 2 iter_args(%k = %c0_i4) : (i4) -> () {
            %2:2 = loopschedule.step {
              %ck = arith.cmpi ult, %k, %c2_i4 : i4
              loopschedule.store %c43_i32, %arg1[%k : i4] : memref<4xi32>
              %nk = arith.addi %k, %c1_i4 : i4
              loopschedule.register %nk, %ck : i4, i1
            } : i4, i1
            loopschedule.terminator condition(%2#1), iter_args(%2#0), results() : (i4) -> ()
          }
          loopschedule.register
        }
        loopschedule.terminator condition(%0#1), iter_args(%0#0), results() : (i32) -> ()
      }
    }
    return
  }
}

// The parent loop becomes loop0; its body steps 1 and 2 each become a
// child module (loop0_loop1, loop0_loop2). The parent FSM (loop0_fsm)
// must contain BOTH WAIT_1/POST_1 and WAIT_2/POST_2.

// CHECK: hw.module @loop0
// CHECK-DAG: hw.instance "loop0_loop1_inst" @loop0_loop1
// CHECK-DAG: hw.instance "loop0_loop2_inst" @loop0_loop2

// CHECK: fsm.machine @loop0_fsm
// Two child_done inputs and two child_start / post_active outputs.
// CHECK-DAG: child_done_0
// CHECK-DAG: child_done_1
// CHECK-DAG: child_start_0
// CHECK-DAG: child_start_1
// CHECK-DAG: post_active_0
// CHECK-DAG: post_active_1
// CHECK-DAG: fsm.state @STEP_0
// CHECK-DAG: fsm.state @STEP_1
// CHECK-DAG: fsm.state @WAIT_1
// CHECK-DAG: fsm.state @POST_1
// CHECK-DAG: fsm.state @STEP_2
// CHECK-DAG: fsm.state @WAIT_2
// CHECK-DAG: fsm.state @POST_2

// CHECK: hw.module @loop0_loop1
// CHECK: hw.module @loop0_loop2
