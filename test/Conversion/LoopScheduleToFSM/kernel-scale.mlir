// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal),convert-scf-to-loopschedule,lower-loopschedule-to-fsm)" %s | FileCheck %s


// End-to-end scalar-memref multiply, multi-cycle mul. The loop inlines
// into the single per-function machine as loop0_* states; the pipelined
// mul stretches the body to six cycle states (FRAME_0_0..FRAME_0_5).
// CHECK: hw.module @scale
// CHECK: fsm.hw_instance "scale_fsm_inst" @scale_fsm
// CHECK: hw.instance "mul_pipe_i32_hw_inst_0" @int_mul_pipe_i32_l4
// CHECK: fsm.machine @scale_fsm
// CHECK: fsm.state @IDLE
// CHECK: fsm.state @FRAME_0
// CHECK: fsm.state @loop0_FRAME_0_0
// CHECK: fsm.state @loop0_FRAME_0_5
// CHECK: fsm.state @DONE
// CHECK-NOT: hw.module @loop0

func.func @scale(%a: i32, %A: memref<16xi32>, %B: memref<16xi32>) attributes {top} {
  %c0 = arith.constant 0 : i32
  %cN = arith.constant 16 : i32
  %c1 = arith.constant 1 : i32
  scf.while (%i = %c0) : (i32) -> i32 {
    %cond = arith.cmpi slt, %i, %cN : i32
    scf.condition(%cond) %i : i32
  } do {
  ^bb0(%i: i32):
    %idx = arith.index_cast %i : i32 to index
    %av = memref.load %A[%idx] : memref<16xi32>
    %mul = arith.muli %a, %av : i32
    memref.store %mul, %B[%idx] : memref<16xi32>
    %ni = arith.addi %i, %c1 : i32
    scf.yield %ni : i32
  }
  return
}
