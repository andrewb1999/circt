// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// A sequential loop that writes to a memref.
module {
  loopschedule.func_sequential @fill(%arg0: memref<32xi32>) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %c32_i6 = arith.constant 32 : i6
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
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

// CHECK: hw.module @fill
// CHECK-SAME: in %mem0_rd_data : i32
// CHECK-SAME: out mem0_addr : i5
// CHECK-SAME: out mem0_wr_data : i32
// CHECK-SAME: out mem0_wr_en : i1

// The loop inlines into a single per-function machine (no hw.module @loop0);
// the store data (constant 42) is muxed onto the write port under the loop
// frame-active result in the function body.
// CHECK: fsm.hw_instance "fill_fsm_inst" @fill_fsm
// CHECK: %loop0_iter_arg_0 = seq.compreg.ce sym @loop0_iter_arg_0
// CHECK: comb.mux {{%.+}}, %c42_i32, {{%.+}} : i32
// CHECK-NOT: hw.module @loop0
// CHECK: fsm.machine @fill_fsm
// CHECK-SAME: argNames = ["start", "loop0_cond_entry", "loop0_cond_next", "loop0_stall"]
// CHECK: fsm.state @loop0_FRAME_0
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0
