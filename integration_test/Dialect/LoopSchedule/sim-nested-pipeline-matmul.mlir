// REQUIRES: verilator
// RUN: rm -rf %t.dir %t.hex %t.obj && mkdir %t.dir %t.hex
// RUN: python3 %S/json-to-hex.py %S/sim-nested-pipeline-matmul-input.json %t.hex/
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation{data-dir=%t.hex/},export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/ -Wno-LATCH" %t.dir/top.sv %t.dir/matmul.sv %t.dir/loop0.sv %t.dir/loop0_loop1.sv %t.dir/loop0_fsm.sv %t.dir/loop0_loop1_fsm.sv --cycles 2000 2>&1 | FileCheck %s

// Triply nested 2x2 matrix multiply with pipeline inner loop.
// Outer(i) and middle(j) are sequential, inner(k) is a pipeline (II=1).
//
// A = [[1, 2], [3, 4]]  flat: [1, 2, 3, 4]
// B = [[5, 6], [7, 8]]  flat: [5, 6, 7, 8]
// C = [[19, 22], [43, 50]]  flat: [19, 22, 43, 50]
//                                  [0x13, 0x16, 0x2b, 0x32]

// CHECK: @MEM mem2 4
// CHECK-NEXT: 00000013
// CHECK-NEXT: 00000016
// CHECK-NEXT: 0000002b
// CHECK-NEXT: 00000032
// CHECK: DONE

module {
  func.func @matmul(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i2 = arith.constant 0 : i2
    %c2_i2 = arith.constant -2 : i2
    %c1_i2 = arith.constant 1 : i2
    %c2_i4 = arith.constant 2 : i4
    loopschedule.step {
      // Outer loop: i = 0..1
      loopschedule.sequential trip_count = 2 iter_args(%i = %c0_i2) : (i2) -> () {
        %0:2 = loopschedule.step {
          %cond_i = arith.cmpi ult, %i, %c2_i2 : i2
          // Middle loop: j = 0..1
          loopschedule.sequential trip_count = 2 iter_args(%j = %c0_i2) : (i2) -> () {
            %1:2 = loopschedule.step {
              %cond_j = arith.cmpi ult, %j, %c2_i2 : i2
              // Inner loop: pipeline with k=0..1, accumulating into acc
              %acc = loopschedule.pipeline II = 1 iter_args(%k = %c0_i2, %pacc = %c0_i32) : (i2, i32) -> i32 {
                %2:3 = loopschedule.at 0 -> (i2, i32, i1) {
                  %cond_k = arith.cmpi ult, %k, %c2_i2 : i2
                  // a_addr = i*2 + k
                  %i_ext = arith.extui %i : i2 to i4
                  %k_ext = arith.extui %k : i2 to i4
                  %j_ext = arith.extui %j : i2 to i4
                  %i2val = arith.muli %i_ext, %c2_i4 : i4
                  %a_addr = arith.addi %i2val, %k_ext : i4
                  // b_addr = k*2 + j
                  %k2val = arith.muli %k_ext, %c2_i4 : i4
                  %b_addr = arith.addi %k2val, %j_ext : i4
                  // Load and multiply
                  %a_val = loopschedule.load %arg0[%a_addr : i4] : memref<4xi32>
                  %b_val = loopschedule.load %arg1[%b_addr : i4] : memref<4xi32>
                  %prod = arith.muli %a_val, %b_val : i32
                  %new_acc = arith.addi %pacc, %prod : i32
                  // Advance k
                  %next_k = arith.addi %k, %c1_i2 : i2
                  loopschedule.iter_arg_update %pacc = %new_acc : i32
                  loopschedule.iter_arg_update %k = %next_k : i2
                  loopschedule.yield %next_k, %new_acc, %cond_k : i2, i32, i1
                }
                loopschedule.terminator condition(%2#2), results(%2#1) : i32
              }
              // After inner pipeline: store C[i*2+j] = acc
              %i_ext2 = arith.extui %i : i2 to i4
              %j_ext2 = arith.extui %j : i2 to i4
              %i2val2 = arith.muli %i_ext2, %c2_i4 : i4
              %c_addr = arith.addi %i2val2, %j_ext2 : i4
              loopschedule.store %acc, %arg2[%c_addr : i4] : memref<4xi32>
              // Advance j
              %next_j = arith.addi %j, %c1_i2 : i2
              loopschedule.iter_arg_update %j = %next_j : i2
              loopschedule.register %next_j, %cond_j : i2, i1
            } : i2, i1
            loopschedule.terminator condition(%1#1), results()
          }
          // Advance i
          %next_i = arith.addi %i, %c1_i2 : i2
          loopschedule.iter_arg_update %i = %next_i : i2
          loopschedule.register %next_i, %cond_i : i2, i1
        } : i2, i1
        loopschedule.terminator condition(%0#1), results()
      }
    }
    return
  }
}
