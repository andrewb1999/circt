// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule))" %s | FileCheck %s

// Scheduler wraps an op carrying `loopschedule.dynamic` in a launch+expect
// pair. Uses memref.load's latency (= 1) so the expect lives one stage after
// the launch.
// CHECK-LABEL: func.func @dynamic_load
// CHECK: loopschedule.pipeline
// CHECK: loopschedule.at 0 -> (i64, !loopschedule.handle, i32, i1)
// CHECK:   %[[H:.+]] = loopschedule.launch : !loopschedule.handle
// CHECK:     loopschedule.load %{{.*}}[%{{.*}} : i64] {loopschedule.dynamic{{.*}}} : memref<16xi32>
// CHECK:     loopschedule.yield %{{.*}} : i32
// CHECK: loopschedule.at 1
// CHECK:   loopschedule.expect %{{.+}} : i32

func.func @dynamic_load(%a: memref<16xi32>, %out: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %av = memref.load %a[%idx] {loopschedule.dynamic} : memref<16xi32>
    %inc = arith.addi %av, %c1 : i32
    memref.store %inc, %out[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  } attributes {hls.pipeline}
  return
}
