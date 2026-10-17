// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A multi-cycle operator's result is its output, valid in the one cycle its
// latency after issue: this latency-2 multiply, issued in stage 0, in stage
// 2. Stage 2 reads the output itself; later stages read copies of it, the
// first captured in stage 2 (it was captured a cycle late, and stage 3 read
// the output itself, by then the next iteration's product).

// CHECK-LABEL: hw.module @late_reads
// CHECK:       %[[P:.+]] = hw.instance "mul_0" @mul_pipe
// Stage 2: the output itself.
// CHECK:       %[[A:.+]] = arith.addi %[[P]], %c1_i32 : i32
// CHECK:       %loop0_s2_r0 = seq.compreg.ce sym @loop0_s2_r0 %[[A]],
// Stage 3: the copy captured in stage 2.
// CHECK:       %loop0_s2_dly_r0_from_s0 = seq.compreg.ce sym @loop0_s2_dly_r0_from_s0 %[[P]], %clk
// CHECK:       %[[B:.+]] = arith.addi %loop0_s2_r0, %loop0_s2_dly_r0_from_s0 : i32
// Stage 4: the next copy.
// CHECK:       %loop0_s3_dly_r0_from_s0 = seq.compreg.ce sym @loop0_s3_dly_r0_from_s0 %loop0_s2_dly_r0_from_s0, %clk
// CHECK:       hw.output %{{.+}}, %{{.+}}, %[[B]], %{{.+}}, %{{.+}}, %{{.+}}, %loop0_s3_dly_r0_from_s0,

oplib.library @lib {
  oplib.operator @mul_i32_l2 latency<2>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul" @mul_pipe(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @mul_pipe(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})

loopschedule.func_sequential @late_reads(%x: i32, %A: memref<16xi32>, %B: memref<16xi32>) attributes {oplib.library = @lib, top} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %lh = loopschedule.at 0 -> !loopschedule.handle {
      %lh_launch = loopschedule.launch : !loopschedule.handle {
        loopschedule.pipeline II = 1 latency = 5 iter_args(%i = %c0_i32) : (i32) -> () {
          %s0:3 = loopschedule.at 0 -> (i32, i32, i1) {
            %p = arith.muli %i, %x {loopschedule.operator = @mul_i32_l2} : i32
            %ni = arith.addi %i, %c1_i32 : i32
            %cond = arith.cmpi slt, %ni, %c16_i32 : i32
            loopschedule.iter_arg_update %i = %ni : i32
            loopschedule.yield %p, %i, %cond : i32, i32, i1
          }
          %s1 = loopschedule.at 1 -> i32 {
            loopschedule.yield %s0#1 : i32
          }
          %s2:2 = loopschedule.at 2 -> (i32, i32) {
            %a = arith.addi %s0#0, %c1_i32 : i32
            loopschedule.yield %a, %s1 : i32, i32
          }
          %s3 = loopschedule.at 3 -> i32 {
            %b = arith.addi %s2#0, %s0#0 : i32
            loopschedule.store %b, %A[%s2#1 : i32] : memref<16xi32>
            loopschedule.yield %s2#1 : i32
          }
          loopschedule.at 4 {
            loopschedule.store %s0#0, %B[%s3 : i32] : memref<16xi32>
            loopschedule.yield
          }
          loopschedule.terminator condition(%s0#2), results()
        }
        loopschedule.yield
      }
      loopschedule.yield %lh_launch : !loopschedule.handle
    }
    loopschedule.yield %lh : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}
