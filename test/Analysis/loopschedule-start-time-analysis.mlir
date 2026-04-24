// RUN: circt-opt --test-loopschedule-start-time-analysis %s | FileCheck %s

// Bindable ops = ops with `loopschedule.operator` attr, loopschedule.load,
// loopschedule.store, or LoadInterface/StoreInterface impls.
//
// The analysis produces AP occupancy + concurrency group:
//   startTime = wall-clock cycle in the group's timebase
//   period    = II of the enclosing pipeline (0 for static ops, elided)
//   count     = trip count of the enclosing pipeline (1 for static ops, elided)
//   group     = innermost inline frame / func_pipeline

// Pipeline stages inside a frame: period=II, count=trip_count, and
// startTime = launch_at_offset + stage_offset (folded into the frame's
// timebase because pipelines lower inline). All ops share the frame's
// group id — not the pipeline's — since they share the wall-clock window
// with the frame's static ops.
loopschedule.func_sequential @pipeline_in_frame(
    %a: memref<32xi32>, %b: memref<32xi32>, %c: memref<32xi32>)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i7
  %c32 = arith.constant 32 : i7
  %c1 = arith.constant 1 : i7
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %p = loopschedule.pipeline II = 1 trip_count = 32 iter_args(%i = %c0) : (i7) -> i7 {
          %r:5 = loopschedule.at 0 -> (i7, i6, i32, i32, i1) {
            // Stage 0: cmp, load. startTime = 0 (launch at-0 + stage 0).
            // CHECK: arith.cmpi
            // CHECK-SAME: loopschedule.count = 32
            // CHECK-SAME: loopschedule.start_time = 0
            %cond = arith.cmpi slt, %i, %c32 {loopschedule.operator = @cmp_l0} : i7
            // CHECK: loopschedule.load
            // CHECK-SAME: loopschedule.count = 32
            // CHECK-SAME: loopschedule.start_time = 0
            %idx = arith.trunci %i : i7 to i6
            %av = loopschedule.load %a[%idx : i6] : memref<32xi32>
            %next = arith.addi %i, %c1 : i7
            loopschedule.iter_arg_update %i = %next : i7
            loopschedule.yield %next, %idx, %av, %av, %cond : i7, i6, i32, i32, i1
          }
          %s1:3 = loopschedule.at 1 -> (i7, i6, i32) {
            // Stage 1: add. startTime = 1.
            // CHECK: arith.addi %{{.+}}#2, %{{.+}}#3
            // CHECK-SAME: loopschedule.count = 32
            // CHECK-SAME: loopschedule.start_time = 1
            %sum = arith.addi %r#2, %r#3 {loopschedule.operator = @add_i32_l0} : i32
            loopschedule.yield %r#0, %r#1, %sum : i7, i6, i32
          }
          %s2 = loopschedule.at 2 -> i7 {
            // Stage 2: store. startTime = 2.
            // CHECK: loopschedule.store
            // CHECK-SAME: loopschedule.count = 32
            // CHECK-SAME: loopschedule.start_time = 2
            loopschedule.store %s1#2, %c[%s1#1 : i6] : memref<32xi32>
            loopschedule.yield %s1#0 : i7
          }
          loopschedule.terminator condition(%r#4), results(%r#0) : i7
        }
        loopschedule.yield %p : i7
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

// Static frame: no period/count attrs (both defaults), start_time is the
// at offset, and a distinct concurrency-group id from any other frame.
loopschedule.func_sequential @static_frame(%buf: memref<4xi32>) attributes {top} {
  %c0 = arith.constant 0 : i2
  %c7 = arith.constant 7 : i32
  loopschedule.frame {
    loopschedule.at 3 {
      // CHECK: loopschedule.store
      // CHECK-SAME: loopschedule.start_time = 3
      // CHECK-NOT: loopschedule.period
      // CHECK-NOT: loopschedule.count
      loopschedule.store %c7, %buf[%c0 : i2] : memref<4xi32>
    }
  }
  loopschedule.return
}

// Non-bindable ops are never stamped.
// CHECK-NOT: arith.constant {{.*}}loopschedule.start_time
// CHECK-NOT: arith.trunci {{.*}}loopschedule.start_time

oplib.library @lib {}
