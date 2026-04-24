// RUN: circt-opt %s -verify-diagnostics -split-input-file | circt-opt -verify-diagnostics -split-input-file | FileCheck %s

// CHECK-LABEL: loopschedule.func_sequential @vadd
loopschedule.func_sequential @vadd(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) {
  loopschedule.return
}

// CHECK-LABEL: func.func @call_no_results
// CHECK: loopschedule.frame
// CHECK: loopschedule.at 0
// CHECK: loopschedule.launch
// CHECK: loopschedule.call @vadd(%arg0, %arg1, %arg2) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
// CHECK: loopschedule.await
func.func @call_no_results(%a: memref<16xi32>, %b: memref<16xi32>, %c: memref<16xi32>) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %stage = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        loopschedule.call @vadd(%a, %b, %c) : (memref<16xi32>, memref<16xi32>, memref<16xi32>) -> ()
        loopschedule.yield
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %stage : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  return
}

// -----

// Call from inside another loopschedule.func_sequential.
loopschedule.func_sequential @leaf(%a: memref<4xi32>) {
  loopschedule.return
}

// CHECK-LABEL: loopschedule.func_sequential @top_calls_leaf
// CHECK: loopschedule.launch
// CHECK: loopschedule.call @leaf
loopschedule.func_sequential @top_calls_leaf(%a: memref<4xi32>) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %stage = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        loopschedule.call @leaf(%a) : (memref<4xi32>) -> ()
        loopschedule.yield
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %stage : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}
