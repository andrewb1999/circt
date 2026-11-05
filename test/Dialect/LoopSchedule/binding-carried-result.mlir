// RUN: circt-opt --loopschedule-binding %s | FileCheck %s

// A multi-cycle operator's result a pipeline carries to its next iteration
// (an iter_arg_update) is read there from the instance's output, which
// holds it only while the op's operands do, a stage register's: an
// instance with other users takes their operands in between. Read in the
// cycle it is ready, the output is the op's whatever the other users do;
// read after it, the op keeps an instance of its own.

// At II 2 the next iteration reads `acc` (stage 1, cycle 3) a cycle after
// the multiply's result is ready (cycle 2). The two multiplies issue in
// slots 1 and 0 and would share one instance; the one updating `acc` does
// not.
// CHECK-LABEL: loopschedule.func_sequential @carried
// CHECK:         arith.muli %{{.*}} {loopschedule.binding = 1 : i64, loopschedule.operator = @mul_l1} : i32
// CHECK:         arith.muli %{{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l1} : i32

oplib.library @lib {
  oplib.operator @mul_l1 latency<1>, incDelay<1.0>, outDelay<0.5> {
    oplib.target @arith_muli(%arg0: i32, %arg1: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%arg0, %arg1 : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %arg0 : i32, in %arg1 : i32) {
      %0 = oplib.hw_instance "mul_hw_inst" @mul_l1(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, a: %arg0 : i32, b: %arg1 : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}
hw.module.extern @mul_l1(in %clk: i1, in %reset: i1, in %ce: i1, in %a: i32, in %b: i32, out out: i32)

loopschedule.func_sequential @carried(%x: i32, %n: i7) -> i32
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i7
  %c1 = arith.constant 1 : i7
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %p:2 = loopschedule.pipeline II = 2 iter_args(%i = %c0, %acc = %x) : (i7, i32) -> (i7, i32) {
          %r:2 = loopschedule.at 0 -> (i7, i1) {
            %next = arith.addi %i, %c1 : i7
            loopschedule.iter_arg_update %i = %next : i7
            %cond = arith.cmpi slt, %next, %n : i7
            loopschedule.yield %next, %cond : i7, i1
          }
          %m = loopschedule.at 1 -> i32 {
            %a = arith.muli %acc, %x {loopschedule.operator = @mul_l1} : i32
            loopschedule.yield %a : i32
          }
          loopschedule.at 2 {
            loopschedule.iter_arg_update %acc = %m : i32
            %b = arith.muli %x, %x {loopschedule.operator = @mul_l1} : i32
            loopschedule.yield
          }
          loopschedule.terminator condition(%r#1), results(%r#0, %acc) : i7, i32
        }
        loopschedule.yield %p#0, %p#1 : i7, i32
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return %x : i32
}

// At II 1 the next iteration's multiply reads `acc` as it issues (stage 1,
// cycle 2), the cycle the result is ready: it shares with the multiply of
// the frame before, and so does the pipeline's result, latched then.

// CHECK-LABEL: loopschedule.func_sequential @at_ready
// CHECK:         arith.muli %{{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l1} : i32
// CHECK:         arith.muli %{{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l1} : i32
loopschedule.func_sequential @at_ready(%x: i32, %n: i7) -> i32
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i7
  %c1 = arith.constant 1 : i7
  loopschedule.frame {
    loopschedule.at 0 {
      %b = arith.muli %x, %x {loopschedule.operator = @mul_l1} : i32
    }
  }
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %p:2 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %x) : (i7, i32) -> (i7, i32) {
          %r:2 = loopschedule.at 0 -> (i7, i1) {
            %next = arith.addi %i, %c1 : i7
            loopschedule.iter_arg_update %i = %next : i7
            %cond = arith.cmpi slt, %next, %n : i7
            loopschedule.yield %next, %cond : i7, i1
          }
          %m = loopschedule.at 1 -> i32 {
            %a = arith.muli %acc, %x {loopschedule.operator = @mul_l1} : i32
            loopschedule.iter_arg_update %acc = %a : i32
            loopschedule.yield %a : i32
          }
          loopschedule.terminator condition(%r#1), results(%r#0, %m) : i7, i32
        }
        loopschedule.yield %p#0, %p#1 : i7, i32
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return %x : i32
}

// The same pipeline returning the iter-arg itself: after the loop it is
// read from the instance's output, long past the ready cycle.

// CHECK-LABEL: loopschedule.func_sequential @iter_arg_result
// CHECK:         arith.muli %{{.*}} {loopschedule.binding = 0 : i64, loopschedule.operator = @mul_l1} : i32
// CHECK:         arith.muli %{{.*}} {loopschedule.binding = 1 : i64, loopschedule.operator = @mul_l1} : i32
loopschedule.func_sequential @iter_arg_result(%x: i32, %n: i7) -> i32
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i7
  %c1 = arith.constant 1 : i7
  loopschedule.frame {
    loopschedule.at 0 {
      %b = arith.muli %x, %x {loopschedule.operator = @mul_l1} : i32
    }
  }
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %p:2 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %x) : (i7, i32) -> (i7, i32) {
          %r:2 = loopschedule.at 0 -> (i7, i1) {
            %next = arith.addi %i, %c1 : i7
            loopschedule.iter_arg_update %i = %next : i7
            %cond = arith.cmpi slt, %next, %n : i7
            loopschedule.yield %next, %cond : i7, i1
          }
          %m = loopschedule.at 1 -> i32 {
            %a = arith.muli %acc, %x {loopschedule.operator = @mul_l1} : i32
            loopschedule.iter_arg_update %acc = %a : i32
            loopschedule.yield %a : i32
          }
          loopschedule.terminator condition(%r#1), results(%r#0, %acc) : i7, i32
        }
        loopschedule.yield %p#0, %p#1 : i7, i32
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return %x : i32
}
