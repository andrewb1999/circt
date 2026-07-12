// RUN: circt-opt --loopschedule-binding %s | FileCheck %s
// RUN: circt-opt --loopschedule-binding="share-operators=false" %s | FileCheck %s --check-prefix=OFF

// Share-mode binding: an extern-instance operator with NO declared limit
// binds onto the minimal number of physical instances consistent with
// the schedule. The two concurrent cycle-0 muls need their own units;
// the mul in the second frame is guaranteed non-concurrent (frames of a
// func_sequential serialize) and coalesces onto instance 0.
//
// With share-operators=false no compute op is tagged.

oplib.library @lib {
  oplib.operator @mul_i32_l4 latency<4> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_pipe_i32_hw_inst" @int_mul_pipe_i32_l4(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, a: %arg0 : i32, b: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @int_mul_pipe_i32_l4(in %clk: i1, in %reset: i1, in %ce: i1, in %a: i32, in %b: i32, out out: i32)

// CHECK-LABEL: loopschedule.func_sequential @share_muls
// OFF-LABEL: loopschedule.func_sequential @share_muls
loopschedule.func_sequential @share_muls(%x: i32, %y: i32) -> i32
    attributes {oplib.library = @lib, top} {
  loopschedule.frame {
    loopschedule.at 0 {
      // Concurrent muls: distinct instances.
      // CHECK: arith.muli
      // CHECK-SAME: loopschedule.binding = 0
      // OFF: arith.muli
      // OFF-NOT: loopschedule.binding
      %a = arith.muli %x, %y {loopschedule.operator = @mul_i32_l4} : i32
      // CHECK: arith.muli
      // CHECK-SAME: loopschedule.binding = 1
      %b = arith.muli %y, %y {loopschedule.operator = @mul_i32_l4} : i32
    }
  }
  loopschedule.frame {
    loopschedule.at 0 {
      // Later frame: serialized against the first — shares instance 0.
      // CHECK: arith.muli
      // CHECK-SAME: loopschedule.binding = 0
      %c = arith.muli %x, %x {loopschedule.operator = @mul_i32_l4} : i32
    }
  }
  loopschedule.return %x : i32
}
