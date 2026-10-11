// RUN: circt-opt %s -split-input-file -verify-diagnostics

oplib.library @lib0 {
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce (in %l : i32, in %r : i32) {
      %o = comb.add %l, %r : i32
      oplib.hw_return %o : i32
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
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
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
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
  }
}

// -----

oplib.library @lib0 {
  oplib.operator @addi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    // expected-error @+1 {{reference to undefined target 'missing_target'}}
    oplib.hw_match(@missing_target : (i32, i32) -> i32) produce (in %l : i32, in %r : i32) {
      %o = comb.add %l, %r : i32
      oplib.hw_return %o : i32
    }
  }
}

// -----

oplib.library @lib0 {
  oplib.operator @addi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    // expected-error @+1 {{body has 1 `in`-role block arg(s), target expects 2}}
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce (in %l : i32) {
      oplib.hw_return %l : i32
    }
  }
}

// -----

oplib.library @lib0 {
  oplib.operator @addi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    // expected-error @+1 {{hw_return yields 2 value(s), target expects 1}}
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce (in %l : i32, in %r : i32) {
      oplib.hw_return %l, %r : i32, i32
    }
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{must have either both incDelay and outDelay or neither}}
  oplib.operator @addi latency<0>, incDelay<0.2> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce (in %l : i32, in %r : i32) {
      %o = comb.add %l, %r : i32
      oplib.hw_return %o : i32
    }
  }
}

// -----

oplib.library @lib0 {
  // expected-error @+1 {{incDelay and outDelay of combinational operators must be the same}}
  oplib.operator @addi latency<0>, incDelay<0.2>, outDelay<0.5> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.addi"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce (in %l : i32, in %r : i32) {
      %o = comb.add %l, %r : i32
      oplib.hw_return %o : i32
    }
  }
}
