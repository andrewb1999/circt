// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A caller that calls a minimal callee. The caller's FSM should assert the
// callee's start and stall until done; the caller's memref port outputs are
// driven by the callee instance during the call.

// CHECK: hw.module @callee
loopschedule.func_sequential @callee(%arg0: memref<4xi32>) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i2
  %c1_i32 = arith.constant 1 : i32
  loopschedule.frame {
    loopschedule.at 0 {
      loopschedule.store %c1_i32, %arg0[%c0 : i2] : memref<4xi32>
    }
  }
  loopschedule.return
}

// The caller module instantiates @callee and feeds its `done` into the FSM's
// wait input. Caller's memref port outputs are driven by the callee instance
// while the call is active.
// CHECK: hw.module @kernel_top
// CHECK: fsm.hw_instance {{.+}} @kernel_top_fsm(%start, %callee_inst.done)
// CHECK: hw.instance "callee_inst" @callee
// CHECK-SAME: clk: %clk
// CHECK-SAME: rst: %rst
// CHECK-SAME: -> (mem0_addr: i2, mem0_rd_en: i1, mem0_wr_data: i32, mem0_wr_en: i1, ready: i1, done: i1)
// CHECK: hw.output
loopschedule.func_sequential @kernel_top(%arg0: memref<4xi32>) attributes {top, oplib.library = @lib} {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        loopschedule.call @callee(%arg0) : (memref<4xi32>) -> ()
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return
}

oplib.library @lib {}
