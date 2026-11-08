// RUN: circt-opt --loopschedule-binding %s | FileCheck %s

// A shared instance's clock enable is the AND of its users' not-stalls: a
// user that stalls freezes the other users' operations in the instance,
// and its issues fall out of step with theirs. A frame's ops and those of
// a pipeline it launched share a concurrency group (the pipeline lowers
// inline): the frame's ops issue by the launch, and the frame counts plain
// cycles to their results. Against a pipeline that can stall (a dynamic
// access: a launched load), a frame op holds its instance from its issue
// to its ready cycle; two pipelines launched together share none. Without
// a stall, only the issue cycles count.

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

// The frame's multiply, issued in cycle 0, is in the instance through cycle
// 3; the pipeline launched in cycle 1 issues its multiply from then on, and
// reads A behind a launch: an instance of its own.
// CHECK-LABEL: loopschedule.func_sequential @in_flight_stalls
// CHECK:         arith.muli %arg1, %arg1 {loopschedule.binding = 0 : i64
// CHECK:         arith.muli %{{.+}}, %arg1 {loopschedule.binding = 1 : i64
loopschedule.func_sequential @in_flight_stalls(%A: memref<16xi32>, %x: i32)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c15 = arith.constant 15 : i4
  %h = loopschedule.frame -> (!loopschedule.handle) {
    loopschedule.at 0 {
      %p = arith.muli %x, %x {loopschedule.operator = @mul_i32_l4} : i32
    }
    %s = loopschedule.at 1 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.pipeline II = 1 trip_count = 16 iter_args(%iv = %c0) : (i4) -> i4 {
          %hl, %done = loopschedule.at 0 -> (!loopschedule.handle, i1) {
            %idx = arith.index_cast %iv : i4 to index
            %ld = loopschedule.launch : !loopschedule.handle {
              %v = memref.load %A[%idx] : memref<16xi32>
              loopschedule.yield %v : i32
            }
            %cond = arith.cmpi slt, %iv, %c15 : i4
            %n = arith.addi %iv, %c1 : i4
            loopschedule.iter_arg_update %iv = %n : i4
            loopschedule.yield %ld, %cond : !loopschedule.handle, i1
          }
          loopschedule.at 1 {
            %v = loopschedule.expect %hl : i32
            %m = arith.muli %v, %x {loopschedule.operator = @mul_i32_l4} : i32
          }
          loopschedule.terminator condition(%done), results(%iv) : i4
        }
        loopschedule.yield %r : i4
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}

// The same pipeline with no dynamic access never stalls: the multiplies'
// issue cycles never meet, and they share.
// CHECK-LABEL: loopschedule.func_sequential @in_flight_no_stall
// CHECK:         arith.muli %arg1, %arg1 {loopschedule.binding = 0 : i64
// CHECK:         arith.muli %{{.+}}, %arg1 {loopschedule.binding = 0 : i64
loopschedule.func_sequential @in_flight_no_stall(%A: memref<16xi32>, %x: i32)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c15 = arith.constant 15 : i4
  %h = loopschedule.frame -> (!loopschedule.handle) {
    loopschedule.at 0 {
      %p = arith.muli %x, %x {loopschedule.operator = @mul_i32_l4} : i32
    }
    %s = loopschedule.at 1 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.pipeline II = 1 trip_count = 16 iter_args(%iv = %c0) : (i4) -> i4 {
          %w, %done = loopschedule.at 0 -> (i32, i1) {
            %w = arith.extui %iv : i4 to i32
            %cond = arith.cmpi slt, %iv, %c15 : i4
            %n = arith.addi %iv, %c1 : i4
            loopschedule.iter_arg_update %iv = %n : i4
            loopschedule.yield %w, %cond : i32, i1
          }
          loopschedule.at 1 {
            %m = arith.muli %w, %x {loopschedule.operator = @mul_i32_l4} : i32
          }
          loopschedule.terminator condition(%done), results(%iv) : i4
        }
        loopschedule.yield %r : i4
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}

// Two pipelines launched together at II 2, one issuing its multiply in
// even cycles, the other in odd ones. The first reads A behind a launch:
// once it stalls, their issues fall out of step, and they share no
// instance.
// CHECK-LABEL: loopschedule.func_sequential @siblings_stall
// CHECK:         arith.muli %{{.+}}, %arg1 {loopschedule.binding = 1 : i64
// CHECK:         arith.muli %{{.+}}, %arg1 {loopschedule.binding = 0 : i64
loopschedule.func_sequential @siblings_stall(%A: memref<16xi32>, %x: i32)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c15 = arith.constant 15 : i4
  %h:2 = loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
    %s0 = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.pipeline II = 2 trip_count = 16 iter_args(%iv = %c0) : (i4) -> i4 {
          %hl, %done = loopschedule.at 0 -> (!loopschedule.handle, i1) {
            %idx = arith.index_cast %iv : i4 to index
            %ld = loopschedule.launch : !loopschedule.handle {
              %v = memref.load %A[%idx] : memref<16xi32>
              loopschedule.yield %v : i32
            }
            %cond = arith.cmpi slt, %iv, %c15 : i4
            %n = arith.addi %iv, %c1 : i4
            loopschedule.iter_arg_update %iv = %n : i4
            loopschedule.yield %ld, %cond : !loopschedule.handle, i1
          }
          loopschedule.at 2 {
            %v = loopschedule.expect %hl : i32
            %m = arith.muli %v, %x {loopschedule.operator = @mul_i32_l4} : i32
          }
          loopschedule.terminator condition(%done), results(%iv) : i4
        }
        loopschedule.yield %r : i4
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    %s1 = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.pipeline II = 2 trip_count = 16 iter_args(%iv = %c0) : (i4) -> i4 {
          %w, %done = loopschedule.at 0 -> (i32, i1) {
            %w = arith.extui %iv : i4 to i32
            %cond = arith.cmpi slt, %iv, %c15 : i4
            %n = arith.addi %iv, %c1 : i4
            loopschedule.iter_arg_update %iv = %n : i4
            loopschedule.yield %w, %cond : i32, i1
          }
          loopschedule.at 1 {
            %m = arith.muli %w, %x {loopschedule.operator = @mul_i32_l4} : i32
          }
          loopschedule.terminator condition(%done), results(%iv) : i4
        }
        loopschedule.yield %r : i4
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s0, %s1 : !loopschedule.handle, !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h#0
    loopschedule.await %h#1
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}

// The same with no dynamic access: neither stalls, their issues stay in
// even and odd cycles, and they share.
// CHECK-LABEL: loopschedule.func_sequential @siblings_no_stall
// CHECK:         arith.muli %{{.+}}, %arg1 {loopschedule.binding = 0 : i64
// CHECK:         arith.muli %{{.+}}, %arg1 {loopschedule.binding = 0 : i64
loopschedule.func_sequential @siblings_no_stall(%A: memref<16xi32>, %x: i32)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c15 = arith.constant 15 : i4
  %h:2 = loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
    %s0 = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.pipeline II = 2 trip_count = 16 iter_args(%iv = %c0) : (i4) -> i4 {
          %w, %done = loopschedule.at 0 -> (i32, i1) {
            %w = arith.extui %iv : i4 to i32
            %cond = arith.cmpi slt, %iv, %c15 : i4
            %n = arith.addi %iv, %c1 : i4
            loopschedule.iter_arg_update %iv = %n : i4
            loopschedule.yield %w, %cond : i32, i1
          }
          loopschedule.at 2 {
            %m = arith.muli %w, %x {loopschedule.operator = @mul_i32_l4} : i32
          }
          loopschedule.terminator condition(%done), results(%iv) : i4
        }
        loopschedule.yield %r : i4
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    %s1 = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %r = loopschedule.pipeline II = 2 trip_count = 16 iter_args(%iv = %c0) : (i4) -> i4 {
          %w, %done = loopschedule.at 0 -> (i32, i1) {
            %w = arith.extui %iv : i4 to i32
            %cond = arith.cmpi slt, %iv, %c15 : i4
            %n = arith.addi %iv, %c1 : i4
            loopschedule.iter_arg_update %iv = %n : i4
            loopschedule.yield %w, %cond : i32, i1
          }
          loopschedule.at 1 {
            %m = arith.muli %w, %x {loopschedule.operator = @mul_i32_l4} : i32
          }
          loopschedule.terminator condition(%done), results(%iv) : i4
        }
        loopschedule.yield %r : i4
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s0, %s1 : !loopschedule.handle, !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h#0
    loopschedule.await %h#1
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}
