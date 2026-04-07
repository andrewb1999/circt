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
    loopschedule.step {
      loopschedule.sequential trip_count = 32 iter_args(%i = %c0_i6) : (i6) -> () {
        %cond = arith.cmpi ult, %i, %c32_i6 : i6
        loopschedule.register %cond : i1
      } do {
        %0 = loopschedule.step {
          loopschedule.store %c42, %arg0[%i : i6] : memref<32xi32>
          %next = arith.addi %i, %c1_i6 : i6
          loopschedule.register %next : i6
        } : i6
        loopschedule.terminator iter_args(%0), results() : (i6) -> ()
      }
    }
    return
  }
}
