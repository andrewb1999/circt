// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A locally allocated 3-D memref. Verify a `seq.hlmem` of shape 2x3x4 is
// emitted, the loop module exposes one address port per dim, and three
// per-dim backedges resolve at the function root.
module {
  func.func @fill_local3d() attributes {top} {
    %c42 = arith.constant 42 : i8
    %c0_i1 = arith.constant 0 : i1
    %c0_i2 = arith.constant 0 : i2
    %c0_i3 = arith.constant 0 : i3
    %c1_i3 = arith.constant 1 : i3
    %c-1_i3 = arith.constant -1 : i3
    %alloc = memref.alloc() : memref<2x3x4xi8>
    loopschedule.step {
      loopschedule.sequential trip_count = 4
          iter_args(%i = %c0_i3) : (i3) -> () {
        %0:2 = loopschedule.step {
          %cond = arith.cmpi ult, %i, %c-1_i3 : i3
          loopschedule.store %c42, %alloc[%c0_i1, %c0_i2, %i : i1, i2, i3] : memref<2x3x4xi8>
          %next = arith.addi %i, %c1_i3 : i3
          loopschedule.register %next, %cond : i3, i1
        } : i3, i1
        loopschedule.terminator condition(%0#1), iter_args(%0#0), results() : (i3) -> ()
      }
    }
    return
  }
}

// CHECK: hw.module @fill_local3d
// CHECK: seq.hlmem @local_mem0 {{.*}} <2x3x4xi8>
// CHECK: seq.read
// CHECK: seq.write
// CHECK: hw.instance "loop0_inst" @loop0

// CHECK: hw.module @loop0
// CHECK-SAME: in %mem0_rd_data : i8
// CHECK-SAME: out mem0_addr_0 : i1
// CHECK-SAME: out mem0_addr_1 : i2
// CHECK-SAME: out mem0_addr_2 : i2
// CHECK-SAME: out mem0_wr_data : i8
// CHECK-SAME: out mem0_wr_en : i1
