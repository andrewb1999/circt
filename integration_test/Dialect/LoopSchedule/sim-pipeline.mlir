// REQUIRES: verilator
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation,export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/" %t.dir/top.sv %t.dir/pipeline_add.sv %t.dir/pipeline_add_fsm.sv --cycles 200 2>&1 | FileCheck %s
// CHECK: PASS

// Pipeline accumulator: adds arg0 each of 10 iterations. With arg0=5, expects result=50.
module {
  func.func @pipeline_add(%arg0: i32) -> i32 attributes {top, "testbench.inputs" = [5 : i32], "testbench.expected_outputs" = [50 : i32]} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %c0_acc = arith.constant 0 : i32
    %0 = loopschedule.pipeline II = 1 iter_args(%i = %c0, %acc = %c0_acc) : (i32, i32) -> i32 {
      %1:3 = loopschedule.at 0 -> (i32, i32, i1) {
        %cond = arith.cmpi ult, %i, %c10 : i32
        %next_i = arith.addi %i, %c1 : i32
        %sum = arith.addi %acc, %arg0 : i32
        loopschedule.iter_arg_update %acc = %sum : i32
        loopschedule.iter_arg_update %i = %next_i : i32
        loopschedule.yield %next_i, %sum, %cond : i32, i32, i1
      }
      loopschedule.terminator condition(%1#2), results(%1#1) : i32
    }
    return %0 : i32
  }
}
