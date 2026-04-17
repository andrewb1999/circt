// REQUIRES: verilator
// RUN: rm -rf %t.dir %t.hex %t.obj && mkdir %t.dir %t.hex
// RUN: python3 %S/json-to-hex.py %S/sim-delay-step-input.json %t.hex/
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation{data-dir=%t.hex/},export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/" %t.dir/top.sv %t.dir/delay_fill.sv %t.dir/loop0.sv %t.dir/loop0_fsm.sv --cycles 500 2>&1 | FileCheck %s

// End-to-end test for `loopschedule.delay`: a sequential loop whose body
// consists of a single multi-cycle step containing a delay-2 wrapped store.
// The FSM lowering should expand STEP_0 into 3 sub-states and gate the store
// with the delay-2 cycle output. The loop fills mem[0..9] with i+1 and
// leaves mem[10..15] zero.

// CHECK: @MEM mem0 16
// CHECK-NEXT: 00000001
// CHECK-NEXT: 00000002
// CHECK-NEXT: 00000003
// CHECK-NEXT: 00000004
// CHECK-NEXT: 00000005
// CHECK-NEXT: 00000006
// CHECK-NEXT: 00000007
// CHECK-NEXT: 00000008
// CHECK-NEXT: 00000009
// CHECK-NEXT: 0000000a
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000000
// CHECK-NEXT: 00000000
// CHECK: DONE

module {
  func.func @delay_fill(%arg0: memref<16xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    loopschedule.frame {
      loopschedule.at 0 {
        loopschedule.sequential trip_count = 10 iter_args(%i = %c0_i32) : (i32) -> () {
          %cond, %next = loopschedule.frame -> (i1, i32) {
            %r:2 = loopschedule.at 0 -> (i1, i32) {
              %c = arith.cmpi slt, %i, %c10_i32 : i32
              %n = arith.addi %i, %c1_i32 : i32
              loopschedule.iter_arg_update %i = %n : i32
              loopschedule.yield %c, %n : i1, i32
            }
            loopschedule.at 2 {
              loopschedule.store %r#1, %arg0[%i : i32] : memref<16xi32>
              loopschedule.yield
            }
            loopschedule.yield %r#0, %r#1 : i1, i32
          }
          loopschedule.terminator condition(%cond), results()
        }
        loopschedule.yield
      }
      loopschedule.yield
    }
    return
  }
}
