// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation,lower-loopschedule-to-fsm)" %s | FileCheck %s

// A sequential loop that writes to a memref.
module {
  func.func @fill(%arg0: memref<32xi32>) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %c32_i6 = arith.constant 32 : i6
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.launch at 0 : !loopschedule.handle {
        loopschedule.sequential trip_count = 32 iter_args(%i = %c0_i6) : (i6) -> () {
          %cond, %next = loopschedule.frame -> (i1, i6) {
            %r:2 = loopschedule.at 0 -> (i1, i6) {
              %c = arith.cmpi ult, %i, %c32_i6 : i6
              loopschedule.store %c42, %arg0[%i : i6] : memref<32xi32>
              %n = arith.addi %i, %c1_i6 : i6
              loopschedule.iter_arg_update %i = %n : i6
              loopschedule.yield %c, %n : i1, i6
            }
            loopschedule.yield %r#0, %r#1 : i1, i6
          }
          loopschedule.terminator condition(%cond), results()
        }
        loopschedule.yield
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

// CHECK: hw.module @fill
// CHECK-SAME: in %mem0_rd_data : i32
// CHECK-SAME: out mem0_addr : i5
// CHECK-SAME: out mem0_wr_data : i32
// CHECK-SAME: out mem0_wr_en : i1
// CHECK: hw.instance "loop0_inst" @loop0
// CHECK: hw.module @loop0
// CHECK-SAME: in %mem0_rd_data : i32
// CHECK-SAME: out mem0_addr : i5
// CHECK-SAME: out mem0_wr_data : i32
// CHECK-SAME: out mem0_wr_en : i1
// CHECK: fsm.machine @loop0_fsm
