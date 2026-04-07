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
      %cond = arith.cmpi ult, %i, %c10 : i32
      loopschedule.register %cond : i1
    } do {
      %1:2 = loopschedule.pipeline.stage start = 0 end = 1 {
        %next_i = arith.addi %i, %c1 : i32
        %sum = arith.addi %acc, %arg0 : i32
        loopschedule.register %next_i, %sum : i32, i32
      } : i32, i32
      loopschedule.terminator iter_args(%1#0, %1#1), results(%1#1) : (i32, i32) -> (i32)
    }
    return %0 : i32
  }
}
