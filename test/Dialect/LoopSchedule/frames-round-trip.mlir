// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// Case 1: frame with no await region (short form); body has two `at` regions;
// results flow through the body's terminating yield.
// CHECK-LABEL: func.func @frame_no_children
func.func @frame_no_children(%arg0: i32, %arg1: i32) -> (i1, i32) {
  // CHECK: loopschedule.frame -> (i1, i32) {
  // CHECK-NOT: do
  %c, %a = loopschedule.frame -> (i1, i32) {
    // CHECK: loopschedule.at 0 -> i1
    %cmp = loopschedule.at 0 -> i1 {
      %t = arith.cmpi slt, %arg0, %arg1 : i32
      // CHECK: loopschedule.yield %{{.*}} : i1
      loopschedule.yield %t : i1
    }
    // CHECK: loopschedule.at 2 -> i32
    %sum = loopschedule.at 2 -> i32 {
      %s = arith.addi %arg0, %arg1 : i32
      loopschedule.yield %s : i32
    }
    // CHECK: loopschedule.yield %{{.*}}, %{{.*}} : i1, i32
    loopschedule.yield %cmp, %sum : i1, i32
  }
  return %c, %a : i1, i32
}

// Case 1b: single-region short form with no results.
// CHECK-LABEL: func.func @frame_no_results_short
func.func @frame_no_results_short(%arg0: i32) {
  // CHECK: loopschedule.frame {
  // CHECK-NOT: do
  loopschedule.frame {
    loopschedule.at 0 {
      %t = arith.addi %arg0, %arg0 : i32
      loopschedule.yield
    }
    loopschedule.yield
  }
  return
}

// Case 2: two-frame flow. First frame launches a child and yields the handle;
// second frame's await region consumes it.
// CHECK-LABEL: func.func @frame_launch_await
func.func @frame_launch_await(%arg0: i32) -> i32 {
  // CHECK: loopschedule.frame -> (!loopschedule.handle) {
  // CHECK-NOT: do
  %h = loopschedule.frame -> (!loopschedule.handle) {
    // CHECK: loopschedule.at 0 -> !loopschedule.handle
    // CHECK:   loopschedule.launch : !loopschedule.handle
    %hr = loopschedule.at 0 -> !loopschedule.handle {
      %hr_launch = loopschedule.launch : !loopschedule.handle {
        %dummy = arith.constant 0 : i32
        loopschedule.yield
      }
      loopschedule.yield %hr_launch : !loopschedule.handle
    }
    loopschedule.yield %hr : !loopschedule.handle
  }
  // CHECK: loopschedule.frame -> (i32)
  %y = loopschedule.frame -> (i32) {
    // CHECK: loopschedule.await %{{.*}} -> i32
    %v = loopschedule.await %h -> i32
    // CHECK: loopschedule.yield %{{.*}} : i32
    loopschedule.yield %v : i32
  // CHECK: do (%{{.*}}: i32) {
  } do (%v: i32) {
    // CHECK: loopschedule.at 0 -> i32
    %r = loopschedule.at 0 -> i32 {
      %x = arith.addi %v, %arg0 : i32
      loopschedule.yield %x : i32
    }
    loopschedule.yield %r : i32
  }
  return %y : i32
}

// Case 3: two concurrent launches in one frame, awaited together.
// CHECK-LABEL: func.func @frame_two_launches
func.func @frame_two_launches() -> (i32, i32) {
  %hs:2 = loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
    // CHECK: loopschedule.at 0 -> !loopschedule.handle
    // CHECK:   loopschedule.launch : !loopschedule.handle
    %hA = loopschedule.at 0 -> !loopschedule.handle {
      %hA_launch = loopschedule.launch : !loopschedule.handle {
        %dummy = arith.constant 0 : i32
        loopschedule.yield
      }
      loopschedule.yield %hA_launch : !loopschedule.handle
    }
    // CHECK: loopschedule.at 0 -> !loopschedule.handle
    // CHECK:   loopschedule.launch : !loopschedule.handle
    %hB = loopschedule.at 0 -> !loopschedule.handle {
      %hB_launch = loopschedule.launch : !loopschedule.handle {
        %dummy = arith.constant 0 : i32
        loopschedule.yield
      }
      loopschedule.yield %hB_launch : !loopschedule.handle
    }
    loopschedule.yield %hA, %hB : !loopschedule.handle, !loopschedule.handle
  }
  %res:2 = loopschedule.frame -> (i32, i32) {
    // CHECK: loopschedule.await %{{.*}}, %{{.*}} -> (i32, i32)
    %a, %b = loopschedule.await %hs#0, %hs#1 -> (i32, i32)
    loopschedule.yield %a, %b : i32, i32
  // CHECK: do (%{{.*}}: i32, %{{.*}}: i32) {
  } do (%a: i32, %b: i32) {
    %r:2 = loopschedule.at 0 -> (i32, i32) {
      loopschedule.yield %a, %b : i32, i32
    }
    loopschedule.yield %r#0, %r#1 : i32, i32
  }
  return %res#0, %res#1 : i32, i32
}

// Case 4: launch overlaps with static work in the same frame.
// CHECK-LABEL: func.func @frame_overlap
func.func @frame_overlap(%arg0: i32) -> (i32, i32) {
  %pair:2 = loopschedule.frame -> (!loopschedule.handle, i32) {
    %hA = loopschedule.at 0 -> !loopschedule.handle {
      %hA_launch = loopschedule.launch : !loopschedule.handle {
        %dummy = arith.constant 0 : i32
        loopschedule.yield
      }
      loopschedule.yield %hA_launch : !loopschedule.handle
    }
    %w2 = loopschedule.at 2 -> i32 {
      %x = arith.addi %arg0, %arg0 : i32
      loopschedule.yield %x : i32
    }
    loopschedule.yield %hA, %w2 : !loopschedule.handle, i32
  }
  %r = loopschedule.frame -> (i32) {
    %a = loopschedule.await %pair#0 -> i32
    loopschedule.yield %a : i32
  } do (%a: i32) {
    %x = loopschedule.at 0 -> i32 {
      %s = arith.addi %a, %pair#1 : i32
      loopschedule.yield %s : i32
    }
    loopschedule.yield %x : i32
  }
  return %pair#1, %r : i32, i32
}

// Case 6: sequential outer loop that launches a nested pipeline via
// frame/launch, then awaits its completion handle in a later frame.
// CHECK-LABEL: func.func @sequential_launches_pipeline
func.func @sequential_launches_pipeline(%c0: index, %c1: index, %c10: index) {
  // CHECK: loopschedule.sequential
  loopschedule.sequential iter_args(%iv = %c0) : (index) -> () {
    // CHECK: loopschedule.frame -> (i1, index)
    %cond, %iv_next = loopschedule.frame -> (i1, index) {
      %r:2 = loopschedule.at 0 -> (i1, index) {
        %c = arith.cmpi ult, %iv, %c10 : index
        %n = arith.addi %iv, %c1 : index
        loopschedule.iter_arg_update %iv = %n : index
        loopschedule.yield %c, %n : i1, index
      }
      loopschedule.yield %r#0, %r#1 : i1, index
    }

    // CHECK: loopschedule.frame -> (!loopschedule.handle)
    %h = loopschedule.frame -> (!loopschedule.handle) {
      // CHECK: loopschedule.at 0 -> !loopschedule.handle
      // CHECK:   loopschedule.launch : !loopschedule.handle
      %hp = loopschedule.at 0 -> !loopschedule.handle {
        %hp_launch = loopschedule.launch : !loopschedule.handle {
          // CHECK: loopschedule.pipeline
          loopschedule.pipeline II = 1 iter_args(%jv = %c0) : (index) -> () {
            %1:2 = loopschedule.at 0 -> (index, i1) {
              %jcond = arith.cmpi ult, %jv, %c10 : index
              %jv_n = arith.addi %jv, %c1 : index
              loopschedule.iter_arg_update %jv = %jv_n : index
              loopschedule.yield %jv_n, %jcond : index, i1
            }
            loopschedule.terminator condition(%1#1), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %hp_launch : !loopschedule.handle
      }
      loopschedule.yield %hp : !loopschedule.handle
    }

    // The launched pipeline's handle is awaited at the iteration boundary
    // via the terminator's `await(...)` list rather than an empty trailing
    // frame.
    // CHECK: loopschedule.terminator condition(%{{.*}}), await(%{{.*}}), results()
    loopschedule.terminator condition(%cond), await(%h), results()
  }
  return
}

// Case 7: two concurrent launches both awaited at the iteration boundary via
// the terminator's variadic `await(...)` list.
// CHECK-LABEL: func.func @sequential_terminator_awaits_two_handles
func.func @sequential_terminator_awaits_two_handles(%c0: index, %c1: index, %c10: index) {
  loopschedule.sequential iter_args(%iv = %c0) : (index) -> () {
    %cond, %iv_next = loopschedule.frame -> (i1, index) {
      %r:2 = loopschedule.at 0 -> (i1, index) {
        %c = arith.cmpi ult, %iv, %c10 : index
        %n = arith.addi %iv, %c1 : index
        loopschedule.iter_arg_update %iv = %n : index
        loopschedule.yield %c, %n : i1, index
      }
      loopschedule.yield %r#0, %r#1 : i1, index
    }

    %hs:2 = loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
      %hA = loopschedule.at 0 -> !loopschedule.handle {
        %hA_launch = loopschedule.launch : !loopschedule.handle {
          %dummy = arith.constant 0 : i32
          loopschedule.yield
        }
        loopschedule.yield %hA_launch : !loopschedule.handle
      }
      %hB = loopschedule.at 0 -> !loopschedule.handle {
        %hB_launch = loopschedule.launch : !loopschedule.handle {
          %dummy = arith.constant 0 : i32
          loopschedule.yield
        }
        loopschedule.yield %hB_launch : !loopschedule.handle
      }
      loopschedule.yield %hA, %hB : !loopschedule.handle, !loopschedule.handle
    }

    // CHECK: loopschedule.terminator condition(%{{.*}}), await(%{{.*}}, %{{.*}}), results()
    loopschedule.terminator condition(%cond), await(%hs#0, %hs#1), results()
  }
  return
}

// Case 5b: iter_arg_update directly inside a pipeline stage.
// CHECK-LABEL: func.func @iter_arg_update_in_pipeline_stage
func.func @iter_arg_update_in_pipeline_stage(%c0: index, %c1: index, %c10: index) {
  loopschedule.pipeline II = 1 iter_args(%iv = %c0) : (index) -> () {
    %0:2 = loopschedule.at 0 -> (index, i1) {
      %cond = arith.cmpi ult, %iv, %c10 : index
      %iv_n = arith.addi %iv, %c1 : index
      // CHECK: loopschedule.iter_arg_update %{{.*}} = %{{.*}} : index
      loopschedule.iter_arg_update %iv = %iv_n : index
      loopschedule.yield %iv_n, %cond : index, i1
    }
    loopschedule.terminator condition(%0#1), results()
  }
  return
}
