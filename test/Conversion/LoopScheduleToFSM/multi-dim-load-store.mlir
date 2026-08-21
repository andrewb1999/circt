// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{disable-flatten-memrefs=true})" %s | FileCheck %s


// Multi-dim memref passed in as a function argument: load + store with two
// indices. Verify the top-level module exposes one address port per dim and
// the inlined loop drives per-dim addresses under the frame-active gate.
module {
  loopschedule.func_sequential @copy2d(%arg0: memref<4x8xi32>) attributes {top} {
    %c0_i2 = arith.constant 0 : i2
    %c1_i2 = arith.constant 1 : i2
    %c0_i3 = arith.constant 0 : i3
    %c1_i3 = arith.constant 1 : i3
    %c3_i2 = arith.constant -1 : i2
    %c0_i32 = arith.constant 0 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 4
              iter_args(%i = %c0_i2) : (i2) -> () {
            %cond, %next = loopschedule.frame -> (i1, i2) {
              %r:2 = loopschedule.at 0 -> (i1, i2) {
                %c = arith.cmpi ult, %i, %c3_i2 : i2
                %v = loopschedule.load %arg0[%i, %c0_i3 : i2, i3] : memref<4x8xi32>
                loopschedule.store %v, %arg0[%i, %c1_i3 : i2, i3] : memref<4x8xi32>
                %n = arith.addi %i, %c1_i2 : i2
                loopschedule.iter_arg_update %i = %n : i2
                loopschedule.yield %c, %n : i1, i2
              }
              loopschedule.yield %r#0, %r#1 : i1, i2
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
    loopschedule.return
  }
}

// CHECK: hw.module @copy2d
// CHECK-SAME: in %mem0_rd_data : i32
// CHECK-SAME: out mem0_addr_0 : i2
// CHECK-SAME: out mem0_addr_1 : i3
// CHECK-SAME: out mem0_wr_data : i32
// CHECK-SAME: out mem0_wr_en : i1

// The loop inlines into a single per-function machine (no hw.module @loop0):
// per-dim addresses are computed in the function body and muxed onto the
// addr ports under the loop frame-active result. The store's dim-1 index
// (constant 1) shows up in that mux chain.
// CHECK: fsm.hw_instance "copy2d_fsm_inst" @copy2d_fsm
// CHECK: comb.mux {{%.+}}, %c1_i3, {{%.+}} : i3
// CHECK-NOT: hw.module @loop0
// CHECK: fsm.machine @copy2d_fsm
// CHECK-SAME: argNames = ["start", "loop0_cond_entry", "loop0_cond_next", "loop0_stall"]
// CHECK-SAME: "loop0_frame_active_0"
// CHECK: fsm.state @loop0_FRAME_0
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0
