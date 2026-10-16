// RUN: circt-opt %s | circt-opt | FileCheck %s
// RUN: circt-opt --cse --canonicalize %s | FileCheck %s --check-prefix=CANON

// A multiply-add is a pure operation; a multiply-accumulate sums across its
// executions, the running value being state that each one reads and writes.

// CHECK-LABEL: func.func @fmadd_fmacc(
// CHECK:         %[[R:.+]] = loopschedule.fmadd %arg0, %arg1, %arg2 : f32
// CHECK:         %[[S:.+]] = loopschedule.fmacc %arg0, %arg1, %arg3 : f32
// CHECK:         return %[[R]], %[[S]] : f32, f32
func.func @fmadd_fmacc(%a: f32, %b: f32, %c: f32, %last: i1) -> (f32, f32) {
  %r = loopschedule.fmadd %a, %b, %c : f32
  %s = loopschedule.fmacc %a, %b, %last : f32
  return %r, %s : f32, f32
}

// Two identical multiply-adds are one, and an unused one goes; two
// multiply-accumulates are two beats, and one is a beat whether or not its
// result is read.
// CANON-LABEL: func.func @cse(
// CANON:         %[[R:.+]] = loopschedule.fmadd %arg0, %arg1, %arg2 : f32
// CANON-NOT:     loopschedule.fmadd
// CANON:         loopschedule.fmacc %arg0, %arg1, %arg3 : f32
// CANON:         %[[S:.+]] = loopschedule.fmacc %arg0, %arg1, %arg3 : f32
// CANON:         %[[T:.+]] = loopschedule.fmacc %arg0, %arg1, %arg3 : f32
// CANON:         return %[[R]], %[[R]], %[[S]], %[[T]]
func.func @cse(%a: f32, %b: f32, %c: f32, %last: i1) -> (f32, f32, f32, f32) {
  %unused = loopschedule.fmadd %a, %b, %a : f32
  %r1 = loopschedule.fmadd %a, %b, %c : f32
  %r2 = loopschedule.fmadd %a, %b, %c : f32
  %dropped = loopschedule.fmacc %a, %b, %last : f32
  %s1 = loopschedule.fmacc %a, %b, %last : f32
  %s2 = loopschedule.fmacc %a, %b, %last : f32
  return %r1, %r2, %s1, %s2 : f32, f32, f32, f32
}
