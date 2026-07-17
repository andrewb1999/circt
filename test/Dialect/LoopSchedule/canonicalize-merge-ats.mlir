// RUN: circt-opt -canonicalize --split-input-file %s | FileCheck %s

// Two same-offset `at`s in one frame merge into a single at whose body
// and yield are the concatenation of the originals. The frame's yield
// (and any later-phase user) sees the merged results.

// CHECK-LABEL: func.func @merge_same_offset
func.func @merge_same_offset(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK: loopschedule.frame -> (i32, i32)
  // CHECK: %[[AT:.*]]:2 = loopschedule.at 0 -> (i32, i32) {
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: arith.muli
  // CHECK-NEXT: loopschedule.yield %{{.*}}, %{{.*}} : i32, i32
  // CHECK-NEXT: }
  // CHECK-NOT: loopschedule.at
  // CHECK: loopschedule.yield %[[AT]]#0, %[[AT]]#1 : i32, i32
  %r:2 = loopschedule.frame -> (i32, i32) {
    %a = loopschedule.at 0 -> i32 {
      %s = arith.addi %arg0, %arg1 : i32
      loopschedule.yield %s : i32
    }
    %b = loopschedule.at 0 -> i32 {
      %m = arith.muli %arg0, %arg1 : i32
      loopschedule.yield %m : i32
    }
    loopschedule.yield %a, %b : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}

// -----

// Same-offset ats whose bodies each launch a variable-latency child: the
// launches become siblings of ONE phase (the shape the FSM lowering runs
// concurrently). Different-offset ats stay separate.

// CHECK-LABEL: func.func @merge_launches
func.func @merge_launches(%arg0: i32) {
  // CHECK: loopschedule.at 0 -> (!loopschedule.handle, !loopschedule.handle) {
  // CHECK-COUNT-2: loopschedule.launch
  // CHECK: loopschedule.yield
  // CHECK: loopschedule.at 1 -> i32
  %f:2 = loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
    %h0 = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %c = arith.addi %arg0, %arg0 : i32
        loopschedule.yield %c : i32
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    %h1 = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %c = arith.muli %arg0, %arg0 : i32
        loopschedule.yield %c : i32
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    %x = loopschedule.at 1 -> i32 {
      %c = arith.subi %arg0, %arg0 : i32
      loopschedule.yield %c : i32
    }
    loopschedule.yield %h0, %h1 : !loopschedule.handle, !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %f#0
    loopschedule.await %f#1
  }
  return
}

// -----

// A same-offset at that READS the earlier at's result must not merge:
// the dependence means the pair is not order-free within the cycle.

// CHECK-LABEL: func.func @no_merge_dependent
func.func @no_merge_dependent(%arg0: i32) -> i32 {
  // CHECK: loopschedule.at 0 -> i32
  // CHECK: loopschedule.at 0 -> i32
  %r = loopschedule.frame -> (i32) {
    %a = loopschedule.at 0 -> i32 {
      %s = arith.addi %arg0, %arg0 : i32
      loopschedule.yield %s : i32
    }
    %b = loopschedule.at 0 -> i32 {
      %m = arith.muli %a, %arg0 : i32
      loopschedule.yield %m : i32
    }
    loopschedule.yield %b : i32
  }
  return %r : i32
}
