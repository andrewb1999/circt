// REQUIRES: verilator
// RUN: rm -rf %t.dir %t.hex %t.obj && mkdir %t.dir %t.hex
// RUN: python3 %S/json-to-hex.py %S/sim-memory-input.json %t.hex/
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation{data-dir=%t.hex/},export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/" %t.dir/top.sv %t.dir/fill.sv %t.dir/loop0.sv %t.dir/loop0_fsm.sv --cycles 500 2>&1 | FileCheck %s
// CHECK: @MEM mem0 32
// CHECK-COUNT-32: 0000002a
// CHECK: DONE

// Fill a 32-element memory with the constant 42.
module {
  func.func @fill(%arg0: memref<32xi32>) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c42 = arith.constant 42 : i32
    %c1_i6 = arith.constant 1 : i6
    %c0_i6 = arith.constant 0 : i6
    %c32_i6 = arith.constant 32 : i6
    loopschedule.frame {
      loopschedule.at 0 {
        loopschedule.sequential trip_count = 32 iter_args(%i = %c0_i6) : (i6) -> () {
          %cond, %next = loopschedule.frame -> (i1, i6) {
            %r:2 = loopschedule.at 0 -> (i1, i6) {
              %c = arith.cmpi ult, %i, %c32_i6 : i6
              loopschedule.store %c42, %arg0[%i : i6] : memref<32xi32>
              %n = arith.addi %i, %c1_i6 : i6
              loopschedule.iter_arg_update %i = %n : i6
              loopschedule.yield %c, %n : i1, i6
            }
            loopschedule.yield %r#0, %r#1 : i1, i6
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
