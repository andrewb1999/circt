// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s
// XFAIL: *
// XFAIL reason: LoopScheduleToFSM pass needs updating to consume the new
// launch-inside-at dialect shape (AMC Category A refactor). Tracking issue:
// port FSM pass to the new shape analogous to LoopScheduleToCalyx.


// A sequential loop that writes to a locally allocated memory.
module {
  func.func @fill_local() attributes {top} {
    %c42 = arith.constant 42 : i32
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c8_i4 = arith.constant -8 : i4
    %alloc = memref.alloc() : memref<8xi32>
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8 iter_args(%i = %c0_i4) : (i4) -> () {
            %cond, %next = loopschedule.frame -> (i1, i4) {
              %r:2 = loopschedule.at 0 -> (i1, i4) {
                %c = arith.cmpi ult, %i, %c8_i4 : i4
                loopschedule.store %c42, %alloc[%i : i4] : memref<8xi32>
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
    loopschedule.frame {
      loopschedule.await %h
      loopschedule.yield
    } do {
      loopschedule.yield
    }
    return
  }
}

// CHECK: hw.module @fill_local
// CHECK: seq.hlmem @local_mem0
// CHECK: seq.read
// CHECK: seq.write
// CHECK: hw.instance "loop0_inst" @loop0

// The loop module should have memory ports for the local mem
// CHECK: hw.module @loop0
// CHECK-SAME: in %mem0_rd_data : i32
// CHECK-SAME: out mem0_addr : i3
// CHECK-SAME: out mem0_wr_data : i32
// CHECK-SAME: out mem0_wr_en : i1
