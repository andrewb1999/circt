// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// The early child_start peephole pulls a pipeline's launch ahead of its
// scheduled cycle as far as the values its stages read allow, its first
// stages then running alongside the frame's remaining cycles. The
// scheduler and binding put the child after its scheduled cycle, so the
// frame and the pipeline may share an operator instance: here the latency-2
// multiply issued in frame cycle 1 and the pipeline's are both instance 0.
// The pipeline reads nothing from the frame, and the peephole started it
// in cycle 0, its multiply issuing with the frame's (runtime-size adi's
// pipelined sweep, under the CP-SAT scheduler). The frame's operation must
// be out of the instance first, issued and ready (a stall of the pipeline
// would freeze it under the frame's): child_start (the 9th output) pulses
// in cycle 3, the multiply's ready cycle, not before.

// CHECK-LABEL: hw.module @scale
// CHECK:       %[[MUL:.+]] = hw.instance "mul_0" @mul_pipe
// CHECK:       seq.compreg.ce sym @loop0_f0_at1_r0_latched %[[MUL]]
// CHECK:       label "shared_operator_one_user"
// CHECK-LABEL: fsm.machine @scale_fsm
// CHECK:       fsm.state @loop0_FRAME_0_0 output {
// CHECK-NEXT:    fsm.output %false, %false, %true, %false, %{{.+}}, %false, %true, %false, %false,
// CHECK:       fsm.state @loop0_FRAME_0_2 output {
// CHECK-NEXT:    fsm.output %false, %false, %true, %false, %{{.+}}, %false, %true, %false, %false,
// CHECK:       fsm.state @loop0_FRAME_0_3 output {
// CHECK-NEXT:    fsm.output %false, %false, %true, %false, %{{.+}}, %false, %true, %false, %true,

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

loopschedule.func_sequential @scale(%x: i32, %y: i32)
      attributes {oplib.library = @lib, top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %lh = loopschedule.at 0 -> !loopschedule.handle {
      %lh_launch = loopschedule.launch : !loopschedule.handle {
        loopschedule.sequential trip_count = 8 iter_args(%i = %c0, %s = %c0) : (i32, i32) -> () {
          %f:3 = loopschedule.frame -> (i1, i32, !loopschedule.handle) {
            %c = loopschedule.at 0 -> i1 {
              %c = arith.cmpi slt, %i, %c8 : i32
              loopschedule.yield %c : i1
            }
            %p = loopschedule.at 1 -> i32 {
              %p = arith.muli %i, %x {loopschedule.operator = @mul_i32_l2, loopschedule.binding = 0 : i64} : i32
              loopschedule.yield %p : i32
            }
            %hc = loopschedule.at 4 -> !loopschedule.handle {
              %hl = loopschedule.launch : !loopschedule.handle {
                %pip:2 = loopschedule.pipeline II = 1 trip_count = 8 iter_args(%j = %c0, %acc = %c0) : (i32, i32) -> (i32, i32) {
                  %st:3 = loopschedule.at 0 -> (i32, i32, i1) {
                    %cj = arith.cmpi slt, %j, %c8 : i32
                    %jn = arith.addi %j, %c1 : i32
                    %m = arith.muli %j, %y {loopschedule.operator = @mul_i32_l2, loopschedule.binding = 0 : i64} : i32
                    loopschedule.iter_arg_update %j = %jn : i32
                    loopschedule.yield %jn, %m, %cj : i32, i32, i1
                  }
                  %t = loopschedule.at 2 -> i32 {
                    %v = arith.addi %acc, %st#1 : i32
                    loopschedule.iter_arg_update %acc = %v : i32
                    loopschedule.yield %v : i32
                  }
                  loopschedule.terminator condition(%st#2), results(%st#0, %t) : i32, i32
                }
                loopschedule.yield %pip#0, %pip#1 : i32, i32
              }
              loopschedule.yield %hl : !loopschedule.handle
            }
            loopschedule.yield %c, %p, %hc : i1, i32, !loopschedule.handle
          }
          loopschedule.frame {
            loopschedule.await %f#2
            loopschedule.yield
          } do {
            loopschedule.at 0 {
              %n = arith.addi %i, %c1 : i32
              %sn = arith.addi %s, %f#1 : i32
              loopschedule.iter_arg_update %i = %n : i32
              loopschedule.iter_arg_update %s = %sn : i32
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
