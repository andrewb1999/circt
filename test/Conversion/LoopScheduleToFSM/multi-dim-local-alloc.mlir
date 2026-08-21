// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{disable-flatten-memrefs=true})" %s | FileCheck %s


// A locally allocated 3-D memref. Verify a `seq.hlmem` of shape 2x3x4 is
// emitted and the inlined loop drives one address per dim, with all three
// per-dim addresses resolving into the read/write ports at the function root.
module {
  loopschedule.func_sequential @fill_local3d() attributes {top} {
    %c42 = arith.constant 42 : i8
    %c0_i1 = arith.constant 0 : i1
    %c0_i2 = arith.constant 0 : i2
    %c0_i3 = arith.constant 0 : i3
    %c1_i3 = arith.constant 1 : i3
    %c-1_i3 = arith.constant -1 : i3
    %alloc = memref.alloc() : memref<2x3x4xi8>
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 4
              iter_args(%i = %c0_i3) : (i3) -> () {
            %cond, %next = loopschedule.frame -> (i1, i3) {
              %r:2 = loopschedule.at 0 -> (i1, i3) {
                %c = arith.cmpi ult, %i, %c-1_i3 : i3
                loopschedule.store %c42, %alloc[%c0_i1, %c0_i2, %i : i1, i2, i3] : memref<2x3x4xi8>
                %n = arith.addi %i, %c1_i3 : i3
                loopschedule.iter_arg_update %i = %n : i3
                loopschedule.yield %c, %n : i1, i3
              }
              loopschedule.yield %r#0, %r#1 : i1, i3
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

// CHECK: hw.module @fill_local3d
// CHECK: seq.hlmem @local_mem0 {{.*}} <2x3x4xi8>
// CHECK: seq.read %local_mem0[{{%.+}}, {{%.+}}, {{%.+}}] rden {{.*}} : !seq.hlmem<2x3x4xi8>
// CHECK: seq.write %local_mem0[{{%.+}}, {{%.+}}, {{%.+}}] {{%.+}} wren {{.*}} : !seq.hlmem<2x3x4xi8>

// The loop inlines into a single per-function machine (no hw.module @loop0);
// the store data (constant 42 : i8) is muxed onto the write port under the
// loop frame-active result in the function body.
// CHECK: fsm.hw_instance "fill_local3d_fsm_inst" @fill_local3d_fsm
// CHECK: %loop0_iter_arg_0 = seq.compreg.ce sym @loop0_iter_arg_0
// CHECK: comb.mux {{%.+}}, %c42_i8, {{%.+}} : i8
// CHECK-NOT: hw.module @loop0
// CHECK: fsm.machine @fill_local3d_fsm
// CHECK-SAME: argNames = ["start", "loop0_cond_entry", "loop0_cond_next", "loop0_stall"]
// CHECK: fsm.state @loop0_FRAME_0
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0
