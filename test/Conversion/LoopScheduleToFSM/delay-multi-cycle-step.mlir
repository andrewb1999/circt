// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// Hand-crafted sequential loop whose first frame has latency 3 (an `at 2`
// store inside the frame body). The FSM should expand FRAME_0 into 3
// sub-states, gate the inner store with the offset-2 cycle output, and
// keep frame_active_0 high in all 3 cycles.

module {
  func.func @delay_step(%arg0: memref<16xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 10 iter_args(%i = %c0_i32) : (i32) -> () {
            %cond, %next = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c10_i32 : i32
                %n = arith.addi %i, %c1_i32 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield %c, %n : i1, i32
              }
              loopschedule.at 2 {
                loopschedule.store %r#1, %arg0[%i : i32] : memref<16xi32>
                loopschedule.yield
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
    return
  }
}

// CHECK: fsm.machine @loop0_fsm
// Multi-cycle frames expand into FRAME_<i>_<c> sub-states.
// CHECK-DAG: fsm.state @FRAME_0_0
// CHECK-DAG: fsm.state @FRAME_0_1
// CHECK-DAG: fsm.state @FRAME_0_2
// Result names should expose the per-cycle gates.
// CHECK-DAG: frame_cycle_0_0
// CHECK-DAG: frame_cycle_0_1
// CHECK-DAG: frame_cycle_0_2
