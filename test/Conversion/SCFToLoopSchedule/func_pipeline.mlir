// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule)" %s | FileCheck %s

// Function-level pipelining: a straight-line function carrying `hls.pipeline`
// gets wrapped in a `loopschedule.func_pipeline` instead of
// `loopschedule.func_sequential`. Stages live as direct children of the
// pipeline op; the body terminates with `loopschedule.return`.

// CHECK-LABEL: loopschedule.func_pipeline ii = {{[0-9]+}} @addmul
// CHECK-SAME:    (%[[A:.+]]: i32, %[[B:.+]]: i32, %[[OUT:.+]]: memref<2xi32>)
// CHECK:         loopschedule.at 0
// CHECK:           arith.addi %[[A]], %[[B]]
// CHECK:           arith.muli %[[A]], %[[B]]
// CHECK:         loopschedule.at
// CHECK:         loopschedule.return
func.func @addmul(%a: i32, %b: i32, %out: memref<2xi32>) attributes {hls.pipeline} {
  %sum = arith.addi %a, %b : i32
  %prod = arith.muli %a, %b : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  memref.store %sum, %out[%c0] : memref<2xi32>
  memref.store %prod, %out[%c1] : memref<2xi32>
  return
}
