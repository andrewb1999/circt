// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

loopschedule.func_sequential @callee(%a: memref<4xi32>) {
  loopschedule.return
}

// Round-trip: the `await` form should parse, re-print as `await`, and re-parse.
// CHECK-LABEL: func.func @void_await
// CHECK: loopschedule.frame await {
// CHECK: loopschedule.await
func.func @void_await(%a: memref<4xi32>) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        loopschedule.call @callee(%a) : (memref<4xi32>) -> ()
        loopschedule.yield
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
    loopschedule.yield
  }
  return
}

// -----

// With a forwarded i32: the `await` form carries the frame result via a
// synthesized passthrough body.
// CHECK-LABEL: func.func @result_await
// CHECK: %[[R:.+]] = loopschedule.frame -> (i32) await {
// CHECK:   %[[V:.+]] = loopschedule.await %{{.+}} -> i32
// CHECK:   loopschedule.yield %[[V]] : i32
// CHECK: return %[[R]]
func.func @result_await(%arg0: i32) -> i32 {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hr = loopschedule.at 0 -> !loopschedule.handle {
      %hr_launch = loopschedule.launch : !loopschedule.handle {
        %dummy = arith.constant 0 : i32
        loopschedule.yield
      }
      loopschedule.yield %hr_launch : !loopschedule.handle
    }
    loopschedule.yield %hr : !loopschedule.handle
  }
  %y = loopschedule.frame -> (i32) await {
    %v = loopschedule.await %h -> i32
    loopschedule.yield %v : i32
  }
  return %y : i32
}
