// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// Two top-level frames, each containing a launched sequential loop.
// First loop initializes memory, second loop reads/modifies it.
module {
  func.func @two_loops(%arg0: memref<8xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c42_i32 = arith.constant 42 : i32
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c8_i4 = arith.constant -8 : i4
    %h1 = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8 iter_args(%i = %c0_i4) : (i4) -> () {
            %cond, %next = loopschedule.frame -> (i1, i4) {
              %r:2 = loopschedule.at 0 -> (i1, i4) {
                %c = arith.cmpi ult, %i, %c8_i4 : i4
                loopschedule.store %c42_i32, %arg0[%i : i4] : memref<8xi32>
                %n = arith.addi %i, %c1_i4 : i4
                loopschedule.iter_arg_update %i = %n : i4
                loopschedule.yield %c, %n : i1, i4
              }
              loopschedule.yield %r#0, %r#1 : i1, i4
            }
            loopschedule.terminator condition(%cond), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %lh_launch : !loopschedule.handle
      }
      loopschedule.yield %lh : !loopschedule.handle
    }
    %h2 = loopschedule.frame -> (!loopschedule.handle) {
      loopschedule.await %h1
      loopschedule.yield
    } do {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8 iter_args(%j = %c0_i4) : (i4) -> () {
            %cond, %next = loopschedule.frame -> (i1, i4) {
              %r:2 = loopschedule.at 0 -> (i1, i4) {
                %c = arith.cmpi ult, %j, %c8_i4 : i4
                %val = loopschedule.load %arg0[%j : i4] : memref<8xi32>
                %inc = arith.addi %val, %c0_i32 : i32
                loopschedule.store %inc, %arg0[%j : i4] : memref<8xi32>
                %n = arith.addi %j, %c1_i4 : i4
                loopschedule.iter_arg_update %j = %n : i4
                loopschedule.yield %c, %n : i1, i4
              }
              loopschedule.yield %r#0, %r#1 : i1, i4
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
      loopschedule.await %h2
      loopschedule.yield
    } do {
      loopschedule.yield
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
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @WAIT_0
// CHECK: fsm.state @FRAME_1
// CHECK: fsm.state @WAIT_1
// CHECK: fsm.state @DONE

// CHECK: hw.module @loop0
// CHECK: fsm.machine @loop0_fsm

// CHECK: hw.module @loop1
// CHECK: fsm.machine @loop1_fsm
