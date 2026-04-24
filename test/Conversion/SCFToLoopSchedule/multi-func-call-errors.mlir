// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s -verify-diagnostics -split-input-file

func.func @callee(%x: i32) -> i32 {
  return %x : i32
}

func.func @call_in_pipelined_loop(%a: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cN = arith.constant 16 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    // expected-error @+1 {{'func.call' op func.call is not allowed inside a pipelined loop (dynamic-latency op)}}
    %v = func.call @callee(%i) : (i32) -> i32
    %idx = arith.index_cast %i : i32 to index
    memref.store %v, %a[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  } attributes {hls.pipeline}
  return
}

// -----

func.func @callee(%x: i32) -> i32 {
  return %x : i32
}

func.func @call_in_pipelined_func(%a: i32) -> i32 attributes {top, hls.pipeline = 1 : i64} {
  // expected-error @+1 {{'func.call' op func.call is not allowed inside a pipelined func (dynamic-latency op)}}
  %v = func.call @callee(%a) : (i32) -> i32
  return %v : i32
}
