// RUN: circt-opt --loopschedule-binding %s | FileCheck %s

// An operator whose hw_match takes the cycles its op issues (a `valid`
// argument) takes one step per execution of its op: a core that takes an
// operand only every so many cycles, or one with state. It is its op's
// alone, never bound, even where the schedule would let two ops share an
// instance (frames of a func_sequential serialize); the FSM lowering builds
// it per op and drives `valid` from that op's gate. The fully pipelined
// operator next to it shares as usual.

oplib.library @lib {
  oplib.operator @div_rate_l14 latency<14> {
    oplib.target @arith_divui_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.divui"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_divui_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32, valid %valid : i1) {
      %0 = oplib.hw_instance "div_inst" @div_rate_l14(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, a: %arg0 : i32, b: %arg1 : i32, valid: %valid : i1) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
  oplib.operator @mul_i32_l4 latency<4> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_inst" @int_mul_pipe_i32_l4(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, a: %arg0 : i32, b: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @div_rate_l14(in %clk: i1, in %reset: i1, in %ce: i1, in %a: i32, in %b: i32, in %valid: i1, out out: i32)
hw.module.extern @int_mul_pipe_i32_l4(in %clk: i1, in %reset: i1, in %ce: i1, in %a: i32, in %b: i32, out out: i32)

// CHECK-LABEL: loopschedule.func_sequential @two_frames
loopschedule.func_sequential @two_frames(%x: i32, %y: i32) -> i32
    attributes {oplib.library = @lib, top} {
  loopschedule.frame {
    loopschedule.at 0 {
      // CHECK: arith.divui
      // CHECK-NOT: loopschedule.binding
      // CHECK-SAME: : i32
      %a = arith.divui %x, %y {loopschedule.operator = @div_rate_l14, loopschedule.cycle_latency = 14 : i64} : i32
      // CHECK: arith.muli {{.*}}loopschedule.binding = 0
      %m = arith.muli %x, %y {loopschedule.operator = @mul_i32_l4} : i32
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      // CHECK: arith.divui
      // CHECK-NOT: loopschedule.binding
      // CHECK-SAME: : i32
      %b = arith.divui %y, %x {loopschedule.operator = @div_rate_l14, loopschedule.cycle_latency = 14 : i64} : i32
      // CHECK: arith.muli {{.*}}loopschedule.binding = 0
      %n = arith.muli %y, %y {loopschedule.operator = @mul_i32_l4} : i32
    }
  }
  loopschedule.return %x : i32
}
