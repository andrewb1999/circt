// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A sequential loop whose body has THREE frames:
//   frame 0 — computes the loop condition (regular frame)
//   frame 1 — launches a nested sequential loop (wait frame #0)
//   frame 2 — launches a second nested sequential loop (wait frame #1)
//
// Exercises the unified FSM emission with multiple "wait" frames in a
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
    %hwrap = loopschedule.frame -> (!loopschedule.handle) {
      %louter = loopschedule.launch at 0 : !loopschedule.handle {
        loopschedule.sequential trip_count = 5 iter_args(%i = %c0_i32) : (i32) -> () {
          // Frame 0: compute loop condition + iter_arg next.
          %cond, %next = loopschedule.frame -> (i1, i32) {
            %r:2 = loopschedule.at 0 -> (i1, i32) {
              %c = arith.cmpi slt, %i, %c5_i32 : i32
              %n = arith.addi %i, %c1_i32 : i32
              loopschedule.iter_arg_update %i = %n : i32
              loopschedule.yield %c, %n : i1, i32
            }
            loopschedule.yield %r#0, %r#1 : i1, i32
          }
          // Frame 1: first nested loop via launch.
          %h1 = loopschedule.frame -> (!loopschedule.handle) {
            %l1 = loopschedule.launch at 0 : !loopschedule.handle {
              loopschedule.sequential trip_count = 2 iter_args(%j = %c0_i4) : (i4) -> () {
                %jcond, %jnext = loopschedule.frame -> (i1, i4) {
                  %r:2 = loopschedule.at 0 -> (i1, i4) {
                    %cj = arith.cmpi ult, %j, %c2_i4 : i4
                    loopschedule.store %c42_i32, %arg0[%j : i4] : memref<4xi32>
                    %nj = arith.addi %j, %c1_i4 : i4
                    loopschedule.iter_arg_update %j = %nj : i4
                    loopschedule.yield %cj, %nj : i1, i4
                  }
                  loopschedule.yield %r#0, %r#1 : i1, i4
                }
                loopschedule.terminator condition(%jcond), results()
              }
              loopschedule.yield
            }
            loopschedule.yield %l1 : !loopschedule.handle
          }
          // Frame 2: second nested loop via launch.
          %h2 = loopschedule.frame -> (!loopschedule.handle) {
            loopschedule.await %h1
            loopschedule.yield
          } do {
            %l2 = loopschedule.launch at 0 : !loopschedule.handle {
              loopschedule.sequential trip_count = 2 iter_args(%k = %c0_i4) : (i4) -> () {
                %kcond, %knext = loopschedule.frame -> (i1, i4) {
                  %r:2 = loopschedule.at 0 -> (i1, i4) {
                    %ck = arith.cmpi ult, %k, %c2_i4 : i4
                    loopschedule.store %c43_i32, %arg1[%k : i4] : memref<4xi32>
                    %nk = arith.addi %k, %c1_i4 : i4
                    loopschedule.iter_arg_update %k = %nk : i4
                    loopschedule.yield %ck, %nk : i1, i4
                  }
                  loopschedule.yield %r#0, %r#1 : i1, i4
                }
                loopschedule.terminator condition(%kcond), results()
              }
              loopschedule.yield
            }
            loopschedule.yield %l2 : !loopschedule.handle
          }
          loopschedule.terminator condition(%cond), await(%h2), results()
        }
        loopschedule.yield
      }
      loopschedule.yield %louter : !loopschedule.handle
    }
    loopschedule.frame {
      loopschedule.await %hwrap
      loopschedule.yield
    } do {
      loopschedule.yield
    }
    return
  }
}

// The parent loop becomes loop0; its frames 1 and 2 each become a
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
// CHECK-DAG: fsm.state @FRAME_0
// CHECK-DAG: fsm.state @FRAME_1
// CHECK-DAG: fsm.state @WAIT_1
// CHECK-DAG: fsm.state @POST_1
// CHECK-DAG: fsm.state @FRAME_2
// CHECK-DAG: fsm.state @WAIT_2
// CHECK-DAG: fsm.state @POST_2

// CHECK: hw.module @loop0_loop1
// CHECK: hw.module @loop0_loop2
