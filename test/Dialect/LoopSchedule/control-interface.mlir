// RUN: circt-opt %s -verify-diagnostics -split-input-file | circt-opt -verify-diagnostics -split-input-file | FileCheck %s

// The block-control protocol is an inherent attribute with a dedicated
// `control = <kind>` clause (it must NOT round-trip through the discardable
// attribute dictionary).

// CHECK-LABEL: loopschedule.func_sequential control = ap_ctrl_hs @seq_hs
// CHECK-NOT: attributes
loopschedule.func_sequential control = ap_ctrl_hs @seq_hs(%a: memref<16xi32>) {
  loopschedule.return
}

// CHECK-LABEL: loopschedule.func_sequential control = none @seq_none
loopschedule.func_sequential control = none @seq_none(%a: memref<16xi32>) {
  loopschedule.return
}

// Absent clause round-trips as absent.
// CHECK-LABEL: loopschedule.func_sequential @seq_plain
// CHECK-NOT: control
loopschedule.func_sequential @seq_plain(%a: memref<16xi32>) {
  loopschedule.return
}

// CHECK-LABEL: loopschedule.func_pipeline ii = 2 control = ap_ctrl_hs @pipe_hs
loopschedule.func_pipeline ii = 2 control = ap_ctrl_hs @pipe_hs(%a: memref<16xi32>) {
  loopschedule.return
}
