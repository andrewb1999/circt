// REQUIRES: verilator
// RUN: rm -rf %t.dir && mkdir %t.dir
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation,export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/" %t.dir/top.sv %t.dir/count.sv %t.dir/loop0.sv %t.dir/loop0_fsm.sv --cycles 200 2>&1 | FileCheck %s
// CHECK: PASS

// A sequential loop counting from 0 to 10. No scalar I/O — just verifies done fires.
module {
  func.func @count() attributes {top, "testbench.inputs" = [], "testbench.expected_outputs" = []} {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %c1 = arith.constant 1 : i32
    loopschedule.frame {
      loopschedule.at 0 {
        loopschedule.sequential trip_count = 10 iter_args(%i = %c0) : (i32) -> () {
          %cond, %next = loopschedule.frame -> (i1, i32) {
            %r:2 = loopschedule.at 0 -> (i1, i32) {
              %c = arith.cmpi slt, %i, %c10 : i32
              %n = arith.addi %i, %c1 : i32
              loopschedule.iter_arg_update %i = %n : i32
              loopschedule.yield %c, %n : i1, i32
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
