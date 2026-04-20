// RUN: circt-opt %s -split-input-file -verify-diagnostics

oplib.library @lib0 {
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
}

// -----

// expected-error @+1 {{can only contain OperatorOps}}
oplib.library @lib0 {
  func.func @main() {
    func.return
  }
}

// -----

oplib.library @lib0 {
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    // expected-error @+1 {{operator body may only contain target ops, match ops, and hoisted constants}}
    func.return
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{must contain at least one match op}}
  oplib.operator @addi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
  }
}

// -----

oplib.library @lib0 {
  oplib.operator @addi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    // expected-error @+1 {{reference to undefined target 'missing_target'}}
    oplib.hw_match(@missing_target : (i32, i32) -> i32) produce {
      %l, %r, %o = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%l, %r : i32, i32), outs(%o : i32)
    }
  }
}

// -----

oplib.library @lib0 {
  oplib.operator @addi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    // expected-error @+1 {{yielded different number of inputs than expected by target type}}
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce {
      %l, %r, %o = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%l : i32), outs(%o : i32)
    }
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{must have either both incDelay and outDelay or neither}}
  oplib.operator @addi latency<0>, incDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{incDelay and outDelay of combinational operators must be the same}}
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.5> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "addi" in "arith"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.calyx_match(@target0 : (i32, i32) -> i32) produce {
      %left, %right, %out = calyx.std_add @add : i32, i32, i32
      oplib.yield ins(%left, %right : i32, i32), outs(%out : i32)
    }
  }
}
