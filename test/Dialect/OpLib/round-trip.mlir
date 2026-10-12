// RUN: circt-opt %s | circt-opt | FileCheck %s

hw.module.extern @ext_muli(in %clk : i1, in %reset : i1, in %ce : i1, in %din0 : i32, in %din1 : i32, out dout : i32) attributes {filename = "muli.sv", verilogName = "muli"}

// An operator's module may also be an hw.module, e.g. once a pass has built
// the operator from a vendor core.
hw.module @cmpf_impl(in %clk : i1, in %reset : i1, in %ce : i1, in %din0 : i32, in %din1 : i32, in %opcode : i4, out dout : i1) {
  %false = hw.constant false
  hw.output %false : i1
}

// CHECK-LABEL: @lib0

// CHECK-LABEL: oplib.operator @addi latency<0>, incDelay<2.000000e-01>, outDelay<2.000000e-01>
// CHECK-NEXT: oplib.target @target0(%[[L:.*]]: i32, %[[R:.*]]: i32) -> i32
// CHECK-NEXT: %[[O:.*]] = oplib.operation "arith.addi"(%[[L]], %[[R]] : i32, i32) : i32
// CHECK-NEXT: oplib.output %[[O]] : i32
// CHECK:      oplib.hw_match(@target0 : (i32, i32) -> i32) produce (in %[[ML:.*]] : i32, in %[[MR:.*]] : i32)
// CHECK-NEXT: %[[SUM:.*]] = comb.add %[[ML]], %[[MR]] : i32
// CHECK-NEXT: oplib.hw_return %[[SUM]] : i32

// CHECK-LABEL: oplib.operator @cmpi latency<0>
// CHECK:      oplib.operation "arith.cmpi" with {predicate = 0 : i64}
// CHECK:      comb.icmp eq

// CHECK-LABEL: oplib.operator @muli latency<4>, incDelay<5.000000e-01>, outDelay<5.000000e-01>
// CHECK:      oplib.hw_match(@target0 : (i32, i32) -> i32) produce (clk %[[CLK:.*]] : i1, reset %[[RST:.*]] : i1, ce %[[CE:.*]] : i1, in %[[A:.*]] : i32, in %[[B:.*]] : i32)
// CHECK-NEXT: %[[P:.*]] = oplib.hw_instance "mul" @ext_muli(clk: %[[CLK]]: i1, reset: %[[RST]]: i1, ce: %[[CE]]: i1, din0: %[[A]]: i32, din1: %[[B]]: i32) -> (dout: i32)
// CHECK-NEXT: oplib.hw_return %[[P]] : i32

// An `attr` argument is a constant the lowering takes from one of the op's
// attributes (here the predicate); it is not an operand of the target.
// CHECK-LABEL: oplib.operator @cmpf latency<2>
// CHECK:      oplib.hw_match(@target0 : (f32, f32) -> i1) produce (clk %{{.*}} : i1, reset %{{.*}} : i1, ce %{{.*}} : i1, in %{{.*}} : i32, in %{{.*}} : i32, attr %[[OPC:.*]] : i4)
// CHECK-NEXT: oplib.hw_instance "cmp" @cmpf_impl({{.*}}opcode: %[[OPC]]: i4) -> (dout: i1)

// CHECK-LABEL: oplib.operator @mem latency<1>, limit<2>

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
  oplib.operator @cmpi latency<0> {
    oplib.target @target0(%l: i32, %r: i32) -> i1 {
      %o = oplib.operation "arith.cmpi" with {predicate = 0 : i64}(%l, %r : i32, i32) : i1
      oplib.output %o : i1
    }
    oplib.hw_match(@target0 : (i32, i32) -> i1) produce (in %l : i32, in %r : i32) {
      %o = comb.icmp eq %l, %r : i32
      oplib.hw_return %o : i1
    }
  }
  oplib.operator @muli latency<4>, incDelay<0.5>, outDelay<0.5> {
    oplib.target @target0(%l: i32, %r: i32) -> i32 {
      %o = oplib.operation "arith.muli"(%l, %r : i32, i32) : i32
      oplib.output %o : i32
    }
    oplib.hw_match(@target0 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %p = oplib.hw_instance "mul" @ext_muli(clk: %clk: i1, reset: %rst: i1, ce: %ce: i1, din0: %a: i32, din1: %b: i32) -> (dout: i32)
      oplib.hw_return %p : i32
    }
  }
  oplib.operator @cmpf latency<2> {
    oplib.target @target0(%l: f32, %r: f32) -> i1 {
      %o = oplib.operation "arith.cmpf"(%l, %r : f32, f32) : i1
      oplib.output %o : i1
    }
    oplib.hw_match(@target0 : (f32, f32) -> i1) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32, attr %opc : i4) {
      %p = oplib.hw_instance "cmp" @cmpf_impl(clk: %clk: i1, reset: %rst: i1, ce: %ce: i1, din0: %a: i32, din1: %b: i32, opcode: %opc: i4) -> (dout: i1)
      oplib.hw_return %p : i1
    }
  }
  // An operator without a target carries properties only (here the memory
  // operators operator-allocation emits), so it needs no match op.
  oplib.operator @mem latency<1>, limit<2> {
  }
}
