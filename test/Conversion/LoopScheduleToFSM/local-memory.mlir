// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A sequential loop that writes to a locally allocated memory.
module {
  func.func @fill_local() attributes {top} {
    %c42 = arith.constant 42 : i32
    %c1_i4 = arith.constant 1 : i4
    %c0_i4 = arith.constant 0 : i4
    %c8_i4 = arith.constant -8 : i4
    %alloc = memref.alloc() : memref<8xi32>
    loopschedule.step {
      loopschedule.sequential trip_count = 8 iter_args(%i = %c0_i4) : (i4) -> () {
        %cond = arith.cmpi ult, %i, %c8_i4 : i4
        loopschedule.register %cond : i1
      } do {
        %0 = loopschedule.step {
          loopschedule.store %c42, %alloc[%i : i4] : memref<8xi32>
          %next = arith.addi %i, %c1_i4 : i4
          loopschedule.register %next : i4
        } : i4
        loopschedule.terminator iter_args(%0), results() : (i4) -> ()
      }
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
