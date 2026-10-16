// RUN: circt-opt %s -split-input-file -verify-diagnostics

// The verifiers check a schedule against its ops' latencies: an operator's
// from the function's operator library, here a 2-cycle multiply.

oplib.library @lib {
  oplib.operator @mul_l2 latency<2> {}
}

func.func @not_ready_next_phase(%x: i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  loopschedule.pipeline II = 1 iter_args(%i = %c0) : (i32) -> () {
    %0:3 = loopschedule.at 0 -> (i32, i32, i1) {
      %next = arith.addi %i, %c1 : i32
      %cond = arith.cmpi slt, %next, %c8 : i32
      %p = arith.muli %x, %x {loopschedule.operator = @mul_l2} : i32
      loopschedule.iter_arg_update %i = %next : i32
      loopschedule.yield %next, %p, %cond : i32, i32, i1
    }
    loopschedule.at 1 {
      // expected-error @+1 {{'arith.addi' op operand #0 is ready in cycle 2, after the op issues in cycle 1}}
      %s = arith.addi %0#1, %x : i32
      loopschedule.yield
    }
    loopschedule.terminator condition(%0#2), results()
  }
  return
}

// -----

oplib.library @lib {
  oplib.operator @mul_l2 latency<2> {}
}

func.func @not_ready_same_phase(%x: i32) attributes {oplib.library = @lib} {
  loopschedule.frame {
    loopschedule.at 0 {
      %p = arith.muli %x, %x {loopschedule.operator = @mul_l2} : i32
      // expected-error @+1 {{'arith.addi' op operand #0 is ready in cycle 2, after the op issues in cycle 0}}
      %s = arith.addi %p, %x : i32
      loopschedule.yield
    }
  }
  return
}

// -----

// A bitcast only relabels bits: it passes its operand's cycle on.
oplib.library @lib {
  oplib.operator @mul_l2 latency<2> {}
}

func.func @not_ready_through_bitcast(%x: i32) attributes {oplib.library = @lib} {
  loopschedule.frame {
    %0 = loopschedule.at 0 -> f32 {
      %p = arith.muli %x, %x {loopschedule.operator = @mul_l2} : i32
      %f = arith.bitcast %p : i32 to f32
      loopschedule.yield %f : f32
    }
    loopschedule.at 1 {
      // expected-error @+1 {{'arith.addf' op operand #0 is ready in cycle 2, after the op issues in cycle 1}}
      %s = arith.addf %0, %0 : f32
      loopschedule.yield
    }
  }
  return
}

// -----

// An iter-arg updated in cycle 1 is visible from cycle 2; at II 1 the next
// iteration reads it in cycle 1.
func.func @update_too_late() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  loopschedule.pipeline II = 1 iter_args(%i = %c0) : (i32) -> () {
    %0:2 = loopschedule.at 0 -> (i32, i1) {
      // expected-error @+1 {{'arith.addi' op reads an iter-arg in cycle 0; the next iteration reads it in cycle 1, before its update becomes visible in cycle 2}}
      %next = arith.addi %i, %c1 : i32
      %cond = arith.cmpi slt, %next, %c8 : i32
      loopschedule.yield %next, %cond : i32, i1
    }
    loopschedule.at 1 {
      // expected-note @+1 {{the update}}
      loopschedule.iter_arg_update %i = %0#0 : i32
      loopschedule.yield
    }
    loopschedule.terminator condition(%0#1), results()
  }
  return
}

// -----

// An operator's result goes to the next iteration from its output
// register: the update is visible when the result is ready, and a read in
// cycle 0 at II 1 sees it only if that is cycle 1.
oplib.library @lib {
  oplib.operator @mul_l2 latency<2> {}
}

func.func @update_ready_too_late(%x: i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c1) : (i32, i32) -> () {
    %0 = loopschedule.at 0 -> i1 {
      %next = arith.addi %i, %c1 : i32
      %cond = arith.cmpi slt, %next, %c8 : i32
      loopschedule.iter_arg_update %i = %next : i32
      // expected-error @+1 {{'arith.muli' op reads an iter-arg in cycle 0; the next iteration reads it in cycle 1, before its update becomes visible in cycle 2}}
      %p = arith.muli %acc, %x {loopschedule.operator = @mul_l2} : i32
      // expected-note @+1 {{the update}}
      loopschedule.iter_arg_update %acc = %p : i32
      loopschedule.yield %cond : i1
    }
    loopschedule.terminator condition(%0), results()
  }
  return
}

// -----

// Two ops on one operator instance issue in the same cycle: at II 1, stages
// 0 and 1 are one slot.
oplib.library @lib {
  oplib.operator @mul_l0 latency<0> {}
}

func.func @shared_unit_one_slot(%x: i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  loopschedule.pipeline II = 1 iter_args(%i = %c0) : (i32) -> () {
    %0:2 = loopschedule.at 0 -> (i32, i1) {
      %next = arith.addi %i, %c1 : i32
      %cond = arith.cmpi slt, %next, %c8 : i32
      // expected-note @+1 {{the other op}}
      %p = arith.muli %x, %x {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l0} : i32
      loopschedule.iter_arg_update %i = %next : i32
      loopschedule.yield %p, %cond : i32, i1
    }
    loopschedule.at 1 {
      // expected-error @+1 {{'arith.muli' op shares operator instance @mul_l0 #0 with another op in slot 0 of II 1}}
      %q = arith.muli %0#0, %x {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l0} : i32
      loopschedule.yield
    }
    loopschedule.terminator condition(%0#1), results()
  }
  return
}

// -----

oplib.library @lib {
  oplib.operator @mul_l0 latency<0> {}
}

func.func @shared_unit_one_cycle(%x: i32) attributes {oplib.library = @lib} {
  loopschedule.frame {
    loopschedule.at 3 {
      // expected-note @+1 {{the other op}}
      %p = arith.muli %x, %x {loopschedule.binding = 1 : i64, loopschedule.operator = @mul_l0} : i32
      // expected-error @+1 {{'arith.muli' op shares operator instance @mul_l0 #1 with another op in cycle 3}}
      %q = arith.muli %x, %p {loopschedule.binding = 1 : i64, loopschedule.operator = @mul_l0} : i32
      loopschedule.yield
    }
  }
  return
}

// -----

// The pipeline launches its next iteration in cycle II: a condition
// decided later would launch a second one speculatively.
func.func @condition_too_late() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  loopschedule.pipeline II = 1 iter_args(%i = %c0) : (i32) -> () {
    %0 = loopschedule.at 0 -> i32 {
      %next = arith.addi %i, %c1 : i32
      loopschedule.iter_arg_update %i = %next : i32
      loopschedule.yield %next : i32
    }
    %1 = loopschedule.at 1 -> i32 {
      loopschedule.yield %0 : i32
    }
    %2 = loopschedule.at 2 -> i1 {
      %cond = arith.cmpi slt, %1, %c8 : i32
      loopschedule.yield %cond : i1
    }
    // expected-error @+1 {{'loopschedule.terminator' op condition is ready in cycle 2, past the first initiation interval (1)}}
    loopschedule.terminator condition(%2), results()
  }
  return
}
