// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A sequential loop that writes to a memref.
module {
  func.func @fill(%arg0: memref<32xi32>) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %c32_i6 = arith.constant 32 : i6
    loopschedule.step {
      loopschedule.sequential trip_count = 32 iter_args(%i = %c0_i6) : (i6) -> () {
        %cond = arith.cmpi ult, %i, %c32_i6 : i6
        loopschedule.register %cond : i1
      } do {
        %0 = loopschedule.step {
          loopschedule.store %c42, %arg0[%i : i6] : memref<32xi32>
          %next = arith.addi %i, %c1_i6 : i6
          loopschedule.register %next : i6
        } : i6
        loopschedule.terminator iter_args(%0), results() : (i6) -> ()
      }
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
