// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A frame captures an at result a later at reads past the last cycle it is
// live. An operator instance's result is live in the cycle it is ready: a
// pipelined instance moves on, a shared one takes the next user's operands.
// A value computed from it in that cycle is live only in that cycle too, so
// a reader in a later one takes a capture made then.
//
// The multiply (latency 1, on a shared instance) issues in cycle 0 and is
// ready in cycle 1, where `at 1` adds to it; `at 2` stores the sum. Both
// sit in `loopschedule.if`s, as nussinov's max over its f64 add does: the
// walks that find the instance and the ready read follow an `if`'s result
// into what its body yields. Without the capture, `at 2` read the sum off
// the multiplier's output a cycle late, right only while the multiply was
// its instance's first user, whose operands stand between issues. Both
// frame lowerings: a frame without a launch (frame_hold), and one that
// launches a child (its `_latched` registers; nussinov's).

oplib.library @lib {
  oplib.operator @mul_l1 latency<1>, incDelay<5.000000e-01>, outDelay<5.000000e-01> {
    oplib.target @arith_muli_i32(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_hw_inst" @mul_l1(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %arg0 : i32, rhs: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @mul_l1(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32}) attributes {verilogName = "mul_l1"}

// The sum is captured in cycle 1 (the machine's loop0_frame_cycle_0_1), as
// it is computed, and the store writes the capture.
// CHECK-LABEL: hw.module @ready_read(
// CHECK:         %mul_hw_inst_0.out = hw.instance "mul_hw_inst_0" @mul_l1
// CHECK:         %[[SUM:.+]] = arith.addi %mul_hw_inst_0.out, %{{.+}} : i32
// CHECK:         %[[HOLD:frame_hold_[0-9]+]] = seq.compreg.ce %[[SUM]], %clk, %0#8
// CHECK:         hw.output %{{[^,]+}}, %{{[^,]+}}, %[[HOLD]], 
loopschedule.func_sequential @ready_read(%m: memref<8xi32>, %x: i32)
      attributes {oplib.library = @lib, top} {
    %c0 = arith.constant 0 : i32
    %c8 = arith.constant 8 : i32
    %c1 = arith.constant 1 : i32
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %lh_launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 8 iter_args(%i = %c0) : (i32) -> () {
            %f:2 = loopschedule.frame -> (i1, i32) {
              %r:2 = loopschedule.at 0 -> (i1, i32) {
                %c = arith.cmpi slt, %i, %c8 : i32
                %p = loopschedule.if %c -> (i32) {
                  %v = arith.muli %i, %x {loopschedule.operator = @mul_l1, loopschedule.binding = 0 : i64} : i32
                  loopschedule.yield %v : i32
                }
                loopschedule.yield %c, %p : i1, i32
              }
              %s = loopschedule.at 1 -> i32 {
                %q = loopschedule.if %r#0 -> (i32) {
                  %v = arith.addi %r#1, %c1 : i32
                  loopschedule.yield %v : i32
                }
                loopschedule.yield %q : i32
              }
              loopschedule.at 2 {
                %idx = arith.trunci %i : i32 to i3
                loopschedule.store %s, %m[%idx : i3] : memref<8xi32>
                %n = arith.addi %i, %c1 : i32
                loopschedule.iter_arg_update %i = %n : i32
                loopschedule.yield
              }
              loopschedule.yield %r#0, %s : i1, i32
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

// The same body in a frame that launches an inner pipeline: the sum's
// capture is the frame's `_latched` register for `at 1`'s result, enabled
// in cycle 1.
// CHECK-LABEL: hw.module @ready_read_launch(
// CHECK:         %[[SUM:.+]] = arith.addi %mul_hw_inst_0.out, %{{.+}} : i32
// CHECK:         seq.compreg.ce sym @loop0_f0_at1_r0_latched %[[SUM]], %clk, %{{.+}}
loopschedule.func_sequential @ready_read_launch(%m: memref<8xi32>, %x: i32)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c8 = arith.constant 8 : i32
  %c1 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0) : (i32) -> i32 {
          %f:4 = loopschedule.frame -> (i1, i32, i32, !loopschedule.handle) {
            %r:3 = loopschedule.at 0 -> (i1, i32, i32) {
              %c = arith.cmpi slt, %i, %c8 : i32
              %n = arith.addi %i, %c1 : i32
              %p = loopschedule.if %c -> (i32) {
                %v = arith.muli %i, %x {loopschedule.operator = @mul_l1, loopschedule.binding = 0 : i64} : i32
                loopschedule.yield %v : i32
              }
              loopschedule.iter_arg_update %i = %n : i32
              loopschedule.yield %c, %n, %p : i1, i32, i32
            }
            %s = loopschedule.at 1 -> i32 {
              %q = loopschedule.if %r#0 -> (i32) {
                %v = arith.addi %r#2, %c1 : i32
                loopschedule.yield %v : i32
              }
              loopschedule.yield %q : i32
            }
            loopschedule.at 2 {
              %idx = arith.trunci %i : i32 to i3
              loopschedule.store %s, %m[%idx : i3] : memref<8xi32>
              loopschedule.yield
            }
            %l = loopschedule.at 3 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %pip = loopschedule.pipeline II = 1 iter_args(%k = %c0) : (i32) -> i32 {
                  %p0:2 = loopschedule.at 0 -> (i32, i1) {
                    %kc = arith.cmpi slt, %k, %c8 : i32
                    %knext = arith.addi %k, %c1 : i32
                    loopschedule.iter_arg_update %k = %knext : i32
                    loopschedule.yield %knext, %kc : i32, i1
                  }
                  loopschedule.terminator condition(%p0#1), results(%p0#0) : i32
                }
                loopschedule.yield %pip : i32
              }
              loopschedule.yield %inner : !loopschedule.handle
            }
            loopschedule.yield %r#0, %r#1, %s, %l : i1, i32, i32, !loopschedule.handle
          }
          loopschedule.frame await {
            loopschedule.await %f#3
          }
          loopschedule.terminator condition(%f#0), results(%f#1) : i32
        }
        loopschedule.yield %seq : i32
      }
      loopschedule.yield %launch : !loopschedule.handle
    }
    loopschedule.yield %outer : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return
}
