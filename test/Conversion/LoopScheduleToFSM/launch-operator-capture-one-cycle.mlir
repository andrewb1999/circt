// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A one-cycle frame: its only cycle computes a value on a shared operator
// instance (a combinational multiply binding gave a number, which another
// user may drive once this one has issued) and launches the pipeline that
// reads it. The pipeline runs after that cycle,
// while the instance shows whatever its inputs do next, so it reads the
// value from a capture latched in the frame's cycle. Only a multi-cycle
// frame used to capture for its launched child: a one-cycle frame left
// the child reading the instance's output live (gemm over AXI, whose `k`
// frame pops A[i][k] and multiplies it by alpha on the multiplier the
// `C *= beta` pipeline shares, read beta * C instead).

// CHECK-LABEL: hw.module @scale
// CHECK:       %[[MUL:.+]] = hw.instance "mul_0" @mul_pipe
// CHECK:       %[[HELD:.+]] = seq.compreg.ce sym @loop0_f0_at0_r1_latched %[[MUL]]
// CHECK:       arith.addi %{{.+}}, %[[HELD]] : i32

oplib.library @lib {
  oplib.operator @mul_i32_l0 latency<0>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
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

loopschedule.func_sequential @scale(%x: i32)
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %lh = loopschedule.at 0 -> !loopschedule.handle {
      %lh_launch = loopschedule.launch : !loopschedule.handle {
        loopschedule.sequential trip_count = 8 iter_args(%i = %c0) : (i32) -> () {
          %f:2 = loopschedule.frame -> (i1, !loopschedule.handle) {
            %r:2 = loopschedule.at 0 -> (i1, i32) {
              %c = arith.cmpi slt, %i, %c8 : i32
              %p = arith.muli %i, %x {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_i32_l0} : i32
              loopschedule.yield %c, %p : i1, i32
            }
            %hc = loopschedule.at 0 -> !loopschedule.handle {
              %hl = loopschedule.launch : !loopschedule.handle {
                %pip:2 = loopschedule.pipeline II = 1 trip_count = 8 iter_args(%j = %c0, %acc = %c0) : (i32, i32) -> (i32, i32) {
                  %s:3 = loopschedule.at 0 -> (i32, i32, i1) {
                    %cj = arith.cmpi slt, %j, %c8 : i32
                    %jn = arith.addi %j, %c1 : i32
                    %v = arith.addi %acc, %r#1 : i32
                    loopschedule.iter_arg_update %j = %jn : i32
                    loopschedule.iter_arg_update %acc = %v : i32
                    loopschedule.yield %jn, %v, %cj : i32, i32, i1
                  }
                  loopschedule.terminator condition(%s#2), results(%s#0, %s#1) : i32, i32
                }
                loopschedule.yield %pip#0, %pip#1 : i32, i32
              }
              loopschedule.yield %hl : !loopschedule.handle
            }
            loopschedule.yield %r#0, %hc : i1, !loopschedule.handle
          }
          loopschedule.frame {
            loopschedule.await %f#1
            loopschedule.yield
          } do {
            loopschedule.at 0 {
              %n = arith.addi %i, %c1 : i32
              loopschedule.iter_arg_update %i = %n : i32
              loopschedule.yield
            }
            loopschedule.yield
          }
          loopschedule.terminator condition(%f#0), results()
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
