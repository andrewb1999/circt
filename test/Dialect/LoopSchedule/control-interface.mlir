// RUN: circt-opt %s -verify-diagnostics -split-input-file | circt-opt -verify-diagnostics -split-input-file | FileCheck %s

// The block-control protocol is an inherent attribute with a dedicated
// `control = <kind>` clause (it must NOT round-trip through the discardable
// attribute dictionary).

// CHECK-LABEL: loopschedule.func_sequential control = axil_handshake @seq_axil
// CHECK-NOT: attributes
loopschedule.func_sequential control = axil_handshake @seq_axil(%a: memref<16xi32>) {
  loopschedule.return
}

// CHECK-LABEL: loopschedule.func_sequential control = handshake @seq_handshake
loopschedule.func_sequential control = handshake @seq_handshake(%a: memref<16xi32>) {
  loopschedule.return
}

// Absent clause round-trips as absent.
// CHECK-LABEL: loopschedule.func_sequential @seq_plain
// CHECK-NOT: control
loopschedule.func_sequential @seq_plain(%a: memref<16xi32>) {
  loopschedule.return
}

// CHECK-LABEL: loopschedule.func_pipeline ii = 2 control = axil_handshake @pipe_axil
loopschedule.func_pipeline ii = 2 control = axil_handshake @pipe_axil(%a: memref<16xi32>) {
  loopschedule.return
}
