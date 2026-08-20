// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s


// A sequential loop that writes to a locally allocated memory.
module {
  loopschedule.func_sequential @fill_local() attributes {top} {
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
    loopschedule.return
  }
}

// CHECK: hw.module @fill_local
// CHECK: seq.hlmem @local_mem0
// CHECK: seq.read
// CHECK: seq.write

// The loop inlines into a single per-function machine (no hw.module @loop0);
// the i3 store address and the store data (constant 42) are computed in the
// function body and muxed toward the local memory ports under the loop
// frame-active result.
// CHECK: fsm.hw_instance "fill_local_fsm_inst" @fill_local_fsm
// CHECK: %loop0_iter_arg_0 = seq.compreg.ce sym @loop0_iter_arg_0
// CHECK: comb.extract {{%.+}} from 0 : (i4) -> i3
// CHECK: comb.mux {{%.+}}, %c42_i32, {{%.+}} : i32
// CHECK-NOT: hw.module @loop0
// CHECK: fsm.machine @fill_local_fsm
// CHECK: fsm.state @loop0_FRAME_0
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0
