// RUN: circt-opt --lower-loopschedule-to-fsm=disable-flatten-memrefs=true %s | FileCheck %s

// Two nested sequential loops over an 8x8 memref. Verify per-dim address
// ports flow through both the outer and inner loop hw.modules and that the
// child instance per-dim address muxes wire up correctly.
module {
  func.func @nested2d(%arg0: memref<8x8xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i4 = arith.constant 0 : i4
    %c1_i4 = arith.constant 1 : i4
    %c-8_i4 = arith.constant -8 : i4
    loopschedule.step {
      loopschedule.sequential trip_count = 8
          iter_args(%i = %c0_i4) : (i4) -> () {
        %outer:2 = loopschedule.step {
          %icond = arith.cmpi ult, %i, %c-8_i4 : i4
          loopschedule.sequential trip_count = 8
              iter_args(%j = %c0_i4) : (i4) -> () {
            %inner:2 = loopschedule.step {
              %jcond = arith.cmpi ult, %j, %c-8_i4 : i4
              %ie = arith.trunci %i : i4 to i3
              %je = arith.trunci %j : i4 to i3
              loopschedule.store %c0_i32, %arg0[%ie, %je : i3, i3] : memref<8x8xi32>
              %jnext = arith.addi %j, %c1_i4 : i4
              loopschedule.iter_arg_update %j = %jnext : i4
              loopschedule.register %jnext, %jcond : i4, i1
            } : i4, i1
            loopschedule.terminator condition(%inner#1), results()
          }
          %inext = arith.addi %i, %c1_i4 : i4
          loopschedule.iter_arg_update %i = %inext : i4
          loopschedule.register %inext, %icond : i4, i1
        } : i4, i1
        loopschedule.terminator condition(%outer#1), results()
      }
    }
    return
  }
}

// CHECK: hw.module @nested2d
// CHECK-SAME: out mem0_addr_0 : i3
// CHECK-SAME: out mem0_addr_1 : i3

// CHECK: hw.module @loop0
// CHECK-SAME: out mem0_addr_0 : i3
// CHECK-SAME: out mem0_addr_1 : i3
// CHECK: hw.instance "loop0_loop1_inst" @loop0_loop1

// CHECK: hw.module @loop0_loop1
// CHECK-SAME: out mem0_addr_0 : i3
// CHECK-SAME: out mem0_addr_1 : i3
