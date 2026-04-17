// RUN: circt-opt %s -split-input-file -verify-diagnostics

// `loopschedule.at` cannot appear outside a frame or pipeline.
func.func @at_outside_frame() {
  // expected-error @+1 {{op expects parent op to be one of 'loopschedule.frame, loopschedule.pipeline'}}
  loopschedule.at 0 {
    loopschedule.yield
  }
  return
}

// -----

// `loopschedule.await` cannot appear outside a frame.
func.func @await_outside_frame(%h: !loopschedule.handle) {
  // expected-error @+1 {{op expects parent op 'loopschedule.frame'}}
  loopschedule.await %h
  return
}

// -----

// `loopschedule.launch` cannot appear outside a frame.
func.func @launch_outside_frame() {
  // expected-error @+1 {{op expects parent op 'loopschedule.frame'}}
  %h = loopschedule.launch at 0 : !loopschedule.handle {
    loopschedule.yield
  }
  return
}

// -----

// `loopschedule.at` is illegal inside a frame's await region.
func.func @at_in_await_region() {
  // expected-error @+1 {{await region may contain only loopschedule.await ops}}
  loopschedule.frame {
    loopschedule.at 0 {
      loopschedule.yield
    }
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  return
}

// -----

// `loopschedule.await` is illegal inside a frame's body region.
func.func @await_in_body_region(%h: !loopschedule.handle) {
  // expected-error @+1 {{body region may contain only loopschedule.at and loopschedule.launch ops}}
  loopschedule.frame {
    loopschedule.yield
  } do {
    loopschedule.await %h
    loopschedule.yield
  }
  return
}

// -----

// Body-region yield operand types must match frame result types.
func.func @body_yield_type_mismatch() -> i32 {
  %r = loopschedule.frame -> (i32) {
    loopschedule.yield
  } do {
    %x = loopschedule.at 0 -> i1 {
      %t = arith.constant true
      loopschedule.yield %t : i1
    }
    // expected-error @+1 {{body-region yield types must match frame op result types}}
    loopschedule.yield %x : i1
  }
  return %r : i32
}

// -----

// Await-region yield operand types must match body entry-block arg types.
func.func @await_yield_block_arg_mismatch(%h: !loopschedule.handle) {
  loopschedule.frame {
    %v = loopschedule.await %h -> i32
    // expected-error @+1 {{await-region yield types must match frame body block arg types}}
    loopschedule.yield %v : i32
  } do (%v: i64) {
    loopschedule.yield
  }
  return
}

// -----

// iter_arg_update must be nested inside a sequential or pipeline loop.
func.func @iter_arg_update_outside_loop(%iv: index, %c1: index) {
  loopschedule.frame {
    loopschedule.at 0 {
      %iv_n = arith.addi %iv, %c1 : index
      // expected-error @+1 {{must be nested inside a loopschedule.sequential or loopschedule.pipeline op}}
      loopschedule.iter_arg_update %iv = %iv_n : index
      loopschedule.yield
    }
    loopschedule.yield
  }
  return
}

// -----

// A launched handle must be awaited exactly once; a handle yielded out of a
// frame but never consumed is an error.
func.func @handle_never_awaited() {
  loopschedule.frame -> (!loopschedule.handle) {
    // expected-error @+1 {{handle is never awaited}}
    %h = loopschedule.launch at 0 : !loopschedule.handle {
      loopschedule.yield
    }
    loopschedule.yield %h : !loopschedule.handle
  }
  return
}

// -----

// A handle awaited twice (once via await op, once via terminator await list)
// is an error.
func.func @handle_awaited_twice(%c0: index, %c1: index, %c10: index) {
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
    %h = loopschedule.frame -> (!loopschedule.handle) {
      // expected-error @+1 {{handle is awaited more than once}}
      %hp = loopschedule.launch at 0 : !loopschedule.handle {
        loopschedule.yield
      }
      loopschedule.yield %hp : !loopschedule.handle
    }
    loopschedule.frame {
      loopschedule.await %h
      loopschedule.yield
    } do {
      loopschedule.yield
    }
    loopschedule.terminator condition(%cond), await(%h), results()
  }
  return
}

// -----

// Terminator await operand must trace back to a launch. A !loopschedule.handle
// function argument is not a launch result.
func.func @terminator_await_not_from_launch(%h: !loopschedule.handle,
                                             %c0: index, %c1: index, %c10: index) {
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
    // expected-error @+1 {{'await' operand must be produced by a 'loopschedule.launch'}}
    loopschedule.terminator condition(%cond), await(%h), results()
  }
  return
}

// -----

// Duplicate iter_arg_update for the same iter-arg is an error.
func.func @duplicate_iter_arg_update(%c0: index, %c1: index, %c10: index) {
  loopschedule.sequential iter_args(%iv = %c0) : (index) -> () {
    %cond, %n = loopschedule.frame -> (i1, index) {
      %r:2 = loopschedule.at 0 -> (i1, index) {
        %c = arith.cmpi ult, %iv, %c10 : index
        %m = arith.addi %iv, %c1 : index
        loopschedule.iter_arg_update %iv = %m : index
        // expected-error @+1 {{duplicate iter_arg_update for iter-arg}}
        loopschedule.iter_arg_update %iv = %iv : index
        loopschedule.yield %c, %m : i1, index
      }
      loopschedule.yield %r#0, %r#1 : i1, index
    }
    loopschedule.terminator condition(%cond), results()
  }
  return
}

// -----

// iter_arg_update LHS must be an iter-arg block argument of the enclosing loop.
func.func @iter_arg_update_wrong_lhs(%c0: index, %c1: index, %c10: index) {
  loopschedule.sequential iter_args(%iv = %c0) : (index) -> () {
    %cond, %n = loopschedule.frame -> (i1, index) {
      %r:2 = loopschedule.at 0 -> (i1, index) {
        %c = arith.cmpi ult, %iv, %c10 : index
        %m = arith.addi %iv, %c1 : index
        loopschedule.iter_arg_update %iv = %m : index
        // expected-error @+1 {{'iterArg' must be an iter-arg block argument of the enclosing loopschedule.sequential or loopschedule.pipeline op}}
        loopschedule.iter_arg_update %c0 = %m : index
        loopschedule.yield %c, %m : i1, index
      }
      loopschedule.yield %r#0, %r#1 : i1, index
    }
    loopschedule.terminator condition(%cond), results()
  }
  return
}

// -----

// Frame body children (at + launch) must appear in non-decreasing offset
// order; an `at 0` placed after an `at 2` is a verifier error.
func.func @frame_body_non_monotonic_ats(%arg0: i32) -> i32 {
  %r = loopschedule.frame -> (i32) {
    %a = loopschedule.at 2 -> i32 {
      loopschedule.yield %arg0 : i32
    }
    // expected-error @+1 {{op offset must be >= previous child's offset (2)}}
    %b = loopschedule.at 0 -> i32 {
      loopschedule.yield %arg0 : i32
    }
    loopschedule.yield %a : i32
  }
  return %r : i32
}

// -----

// Monotonic invariant applies to launches too.
func.func @frame_body_non_monotonic_launches() {
  loopschedule.frame {
    %h1 = loopschedule.launch at 3 : !loopschedule.handle {
      loopschedule.yield
    }
    // expected-error @+1 {{op offset must be >= previous child's offset (3)}}
    %h2 = loopschedule.launch at 1 : !loopschedule.handle {
      loopschedule.yield
    }
    loopschedule.yield
  }
  return
}
