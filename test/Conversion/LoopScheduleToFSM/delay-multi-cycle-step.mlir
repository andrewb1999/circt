// RUN: circt-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// Hand-crafted sequential loop whose first step has latency 3 (a delay-2
// load wrapped in loopschedule.delay). The FSM should expand STEP_0 into 3
// sub-states, gate the inner store with the delay-2 cycle output, and keep
// step_active_0 high in all 3 cycles.

module {
  func.func @delay_step(%arg0: memref<16xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    loopschedule.step {
    loopschedule.sequential trip_count = 10 iter_args(%i = %c0_i32) : (i32) -> () {
      %0:2 = loopschedule.step {
        %cond = arith.cmpi slt, %i, %c10_i32 : i32
        %next = arith.addi %i, %c1_i32 : i32
        loopschedule.delay 2 {
          loopschedule.store %next, %arg0[%i : i32] : memref<16xi32>
          loopschedule.register
        }
        loopschedule.register %next, %cond : i32, i1
      } : i32, i1
      loopschedule.terminator condition(%0#1), iter_args(%0#0), results() : (i32) -> ()
    }
    }
    return
  }
}

// CHECK: fsm.machine @loop0_fsm
// First sub-state keeps the legacy STEP_0 name as the entry.
// CHECK-DAG: fsm.state @STEP_0
// CHECK-DAG: fsm.state @STEP_0_c1
// CHECK-DAG: fsm.state @STEP_0_c2
// Result names should expose the per-cycle gates.
// CHECK-DAG: step_cycle_0_0
// CHECK-DAG: step_cycle_0_1
// CHECK-DAG: step_cycle_0_2
