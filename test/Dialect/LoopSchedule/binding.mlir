// RUN: circt-opt --loopschedule-binding %s | FileCheck %s

// The binding pass is a no-op stub today: it walks loopschedule func-level
// containers and leaves the IR unchanged. This test anchors registration
// and the pass's `ModuleOp`-rooted iteration over both func variants.

// CHECK-LABEL: loopschedule.func_sequential @seq
loopschedule.func_sequential @seq(%a: memref<4xi32>) {
  loopschedule.return
}

// CHECK-LABEL: loopschedule.func_pipeline ii = 1 @pipe
loopschedule.func_pipeline ii = 1 @pipe(%x: i32) -> i32 {
  loopschedule.return %x : i32
}
