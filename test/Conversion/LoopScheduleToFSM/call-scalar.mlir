// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A callee that takes an i32 and returns an i32 + 1. Caller passes a scalar
// operand and captures the scalar result via the handle-forward chain.

// CHECK-LABEL: hw.module @add_one
loopschedule.func_sequential @add_one(%x: i32) -> i32 attributes {oplib.library = @lib} {
  %c1 = arith.constant 1 : i32
  %s = arith.addi %x, %c1 : i32
  loopschedule.frame {
    loopschedule.at 0 {
    }
  }
  loopschedule.return %s : i32
}

// CHECK-LABEL: hw.module @scalar_top
// CHECK: hw.instance "add_one_inst" @add_one
// CHECK-SAME: arg0: %arg0
// CHECK-SAME: -> (ready: i1, done: i1, result0: i32)
loopschedule.func_sequential @scalar_top(%x: i32) -> i32 attributes {top, oplib.library = @lib} {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.call @add_one(%x) : (i32) -> i32
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  %v = loopschedule.frame -> (i32) await {
    %r = loopschedule.await %h -> i32
    loopschedule.yield %r : i32
  }
  loopschedule.return %v : i32
}

oplib.library @lib {}
