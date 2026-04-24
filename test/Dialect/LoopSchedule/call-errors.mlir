// RUN: circt-opt %s -verify-diagnostics -split-input-file

loopschedule.func_sequential @callee(%a: memref<4xi32>) {
  loopschedule.return
}

func.func @call_not_in_launch(%a: memref<4xi32>) {
  // expected-error @+1 {{must be directly nested in a loopschedule.launch}}
  loopschedule.call @callee(%a) : (memref<4xi32>) -> ()
  return
}

// -----

loopschedule.func_sequential @callee(%a: i32) -> i32 {
  loopschedule.return %a : i32
}

func.func @bad_operand_count(%a: i32, %b: i32) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %stage = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        // expected-error @+1 {{incorrect number of operands for callee: expected 1 got 2}}
        %r = loopschedule.call @callee(%a, %b) : (i32, i32) -> i32
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

loopschedule.func_sequential @callee(%a: i32) -> i32 {
  loopschedule.return %a : i32
}

func.func @bad_operand_type(%a: i64) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %stage = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        // expected-error @+1 {{operand #0 type mismatch: expected 'i32' got 'i64'}}
        %r = loopschedule.call @callee(%a) : (i64) -> i32
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

func.func @not_a_schedule_callee(%a: i32) -> i32 {
  return %a : i32
}

func.func @wrong_callee_kind(%a: i32) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %stage = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        // expected-error @+1 {{'not_a_schedule_callee' must reference a loopschedule.func_sequential or loopschedule.func_pipeline}}
        %r = loopschedule.call @not_a_schedule_callee(%a) : (i32) -> i32
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

func.func @missing_symbol(%a: i32) {
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %stage = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        // expected-error @+1 {{'ghost' does not reference a valid symbol}}
        %r = loopschedule.call @ghost(%a) : (i32) -> i32
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
