// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// Caller allocates a local BRAM, passes it to a callee that stores a value
// into it. The callee's memref output ports should drive the caller-local
// hlmem's write port.

// CHECK-LABEL: hw.module @writer
loopschedule.func_sequential @writer(%buf: memref<4xi32>) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i2
  %c42 = arith.constant 42 : i32
  loopschedule.frame {
    loopschedule.at 0 {
      loopschedule.store %c42, %buf[%c0 : i2] : memref<4xi32>
    }
  }
  loopschedule.return
}

// CHECK-LABEL: hw.module @local_top
// CHECK: seq.hlmem @{{.+}} <4xi32>
// CHECK: hw.instance "writer_inst" @writer
loopschedule.func_sequential @local_top() attributes {top, oplib.library = @lib} {
  %local = memref.alloc() : memref<4xi32>
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        loopschedule.call @writer(%local) : (memref<4xi32>) -> ()
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
