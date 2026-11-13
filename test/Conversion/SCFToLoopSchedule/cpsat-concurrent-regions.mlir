// REQUIRES: or-tools
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule-cpsat{cycle-time-ns=3.0 cpsat-workers=1 cpsat-cycle-slack=0 share-mux-delays=0.3,0.3,0.6})" %s | FileCheck %s --check-prefixes=CHECK,TOGETHER
// RUN: circt-opt --pass-pipeline="builtin.module(convert-scf-to-loopschedule-cpsat{cycle-time-ns=3.0 cpsat-workers=1 cpsat-cycle-slack=1 share-mux-delays=0.3,0.3,0.6})" %s | FileCheck %s --check-prefixes=CHECK,APART

// The CP-SAT scheduler shares an instance across a function's regions,
// which mostly run one at a time. A child loop runs from its launch until
// its parent's next frame, which awaits it: alongside a sibling launched in
// the same cycle, and alongside the parent's ops still in an instance then
// (issued by the launch, ready after it). The model keeps an instance's
// users apart there: such siblings launch in different cycles, and such a
// parent op is out of the instance by the launch or comes after it.

oplib.library @lib {
  oplib.operator @mul_l2 latency<2>, incDelay<0.5>, outDelay<0.5> attributes {amc.area_dsp = 1 : i64} {
    oplib.target @arith_muli_i32(%a: i32, %b: i32) -> i32 {
      %0 = oplib.operation "arith.muli"(%a, %b : i32, i32) : i32
      oplib.output %0 : i32
    }
    oplib.hw_match(@arith_muli_i32 : (i32, i32) -> i32) produce (clk %clk : i1, reset %rst : i1, ce %ce : i1, in %a : i32, in %b : i32) {
      %0 = oplib.hw_instance "mul" @mul_l2(clk: %clk : i1, reset: %rst : i1, ce: %ce : i1, lhs: %a : i32, rhs: %b : i32) -> (out: i32)
      oplib.hw_return %0 : i32
    }
  }
}

hw.module.extern @mul_l2(in %clk : i1 {oplib.clock}, in %reset : i1 {oplib.reset}, in %ce : i1 {oplib.enable}, in %lhs : i32 {oplib.operand = 0 : i32}, in %rhs : i32 {oplib.operand = 1 : i32}, out out : i32 {oplib.result = 0 : i32})


// Two independent pipelines that would share one multiplier. Launched
// together they would both issue on it; so they share it launched one
// after the other, or launch together on two. Launched together, they
// overlap: the function takes the longer's cycles, not their sum, which
// the model credits. The fastest design (slack 0) launches them together
// on two; allowed twice the cycles (a slack of 1), one multiplier is less
// area.
// CHECK-LABEL:     loopschedule.func_sequential @siblings
// TOGETHER:          loopschedule.frame -> (!loopschedule.handle, !loopschedule.handle) {
// TOGETHER:            loopschedule.launch
// TOGETHER:              arith.muli {{.*}}loopschedule.binding = [[B:[01]]] : i64
// TOGETHER:            loopschedule.launch
// TOGETHER-NOT:          loopschedule.binding = [[B]] : i64
// TOGETHER:              arith.muli {{.*}}loopschedule.binding = {{[01]}} : i64
// TOGETHER:          loopschedule.frame
// TOGETHER-NEXT:       loopschedule.await
// TOGETHER-NEXT:       loopschedule.await
// APART:             %[[F0:.+]] = loopschedule.frame -> (!loopschedule.handle) {
// APART:               loopschedule.launch
// APART:                 arith.muli {{.*}}loopschedule.binding = 0 : i64
// APART-NOT:           loopschedule.launch
// APART:             loopschedule.frame
// APART-NEXT:          loopschedule.await %[[F0]]
// APART:               loopschedule.launch
// APART:                 arith.muli {{.*}}loopschedule.binding = 0 : i64
func.func @siblings(%a: i32, %b: i32) -> (i32, i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %r0:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %m = arith.muli %i, %a {loopschedule.operator = @mul_l2} : i32
    %ns = arith.addi %s, %m : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline}
  %r1:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %m = arith.muli %i, %b {loopschedule.operator = @mul_l2} : i32
    %ns = arith.addi %s, %m : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline}
  return %r0#1, %r1#1 : i32, i32
}

// The parent's multiply (2 cycles) shares the pipeline's only out of the
// launch's frame: issued in it, it would be in the instance while the child
// runs. The fastest design (slack 0) issues it there, on an instance of
// its own; allowed the cycles, it shares, in a frame of its own after the
// child.
// CHECK-LABEL: loopschedule.func_sequential @overlap
// CHECK:         loopschedule.frame -> {{.*}}!loopschedule.handle{{.*}} {
// TOGETHER:        arith.muli %arg0, %arg1 {loopschedule.binding = [[P:[01]]] : i64
// TOGETHER:        loopschedule.launch
// TOGETHER-NOT:        loopschedule.binding = [[P]] : i64
// TOGETHER:            arith.muli %{{.+}}, %arg1 {loopschedule.binding = {{[01]}} : i64
// APART-NOT:       arith.muli %arg0, %arg1
// APART:           loopschedule.launch
// APART:               arith.muli %{{.+}}, %arg1 {loopschedule.binding = 0 : i64
// APART:           arith.muli %arg0, %arg1 {loopschedule.binding = 0 : i64
func.func @overlap(%a: i32, %b: i32) -> (i32, i32) attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c16 = arith.constant 16 : i32
  %p = arith.muli %a, %b {loopschedule.operator = @mul_l2} : i32
  %res:2 = scf.while (%i = %c0, %s = %c0) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %i, %c16 : i32
    scf.condition(%cond) %i, %s : i32, i32
  } do {
  ^bb0(%i: i32, %s: i32):
    %m = arith.muli %i, %b {loopschedule.operator = @mul_l2} : i32
    %ns = arith.addi %s, %m : i32
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni, %ns : i32, i32
  } attributes {hls.pipeline}
  return %p, %res#1 : i32, i32
}
