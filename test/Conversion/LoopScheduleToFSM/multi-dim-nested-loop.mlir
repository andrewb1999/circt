// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{disable-flatten-memrefs=true})" %s | FileCheck %s


// Two nested sequential loops over an 8x8 memref. Verify per-dim address
// ports flow through both the outer and inner loop hw.modules and that the
// child instance per-dim address muxes wire up correctly.
module {
  loopschedule.func_sequential @nested2d(%arg0: memref<8x8xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i4 = arith.constant 0 : i4
    %c1_i4 = arith.constant 1 : i4
    %c-8_i4 = arith.constant -8 : i4
    %houter = loopschedule.frame -> (!loopschedule.handle) {
      %lo = loopschedule.at 0 -> !loopschedule.handle {
        %lo_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8
              iter_args(%i = %c0_i4) : (i4) -> () {
            // Frame 0: compute cond + iter update + a narrow copy of `%i`.
            %icond, %ie = loopschedule.frame -> (i1, i3) {
              %r:2 = loopschedule.at 0 -> (i1, i3) {
                %c = arith.cmpi ult, %i, %c-8_i4 : i4
                %ie_local = arith.trunci %i : i4 to i3
                %n = arith.addi %i, %c1_i4 : i4
                loopschedule.iter_arg_update %i = %n : i4
                loopschedule.yield %c, %ie_local : i1, i3
              }
              loopschedule.yield %r#0, %r#1 : i1, i3
            }
            // Frame 1: launches the inner (nested) sequential loop.
            %hinner = loopschedule.frame -> (!loopschedule.handle) {
              %li = loopschedule.at 0 -> !loopschedule.handle {
                %li_launch = loopschedule.launch : !loopschedule.handle {
                  loopschedule.sequential trip_count = 8
                      iter_args(%j = %c0_i4) : (i4) -> () {
                    %jcond, %jnext = loopschedule.frame -> (i1, i4) {
                      %r:2 = loopschedule.at 0 -> (i1, i4) {
                        %c = arith.cmpi ult, %j, %c-8_i4 : i4
                        %je = arith.trunci %j : i4 to i3
                        loopschedule.store %c0_i32, %arg0[%ie, %je : i3, i3] : memref<8x8xi32>
                        %n = arith.addi %j, %c1_i4 : i4
                        loopschedule.iter_arg_update %j = %n : i4
                        loopschedule.yield %c, %n : i1, i4
                      }
                      loopschedule.yield %r#0, %r#1 : i1, i4
                    }
                    loopschedule.terminator condition(%jcond), results()
                  }
                  loopschedule.yield
                }
                loopschedule.yield %li_launch : !loopschedule.handle
              }
              loopschedule.yield %li : !loopschedule.handle
            }
            loopschedule.terminator condition(%icond), await(%hinner), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %lo_launch : !loopschedule.handle
      }
      loopschedule.yield %lo : !loopschedule.handle
    }
    loopschedule.frame {
      loopschedule.await %houter
      loopschedule.yield
    } do {
      loopschedule.yield
    }
    loopschedule.return
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
