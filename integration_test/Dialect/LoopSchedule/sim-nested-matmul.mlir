// REQUIRES: verilator
// RUN: rm -rf %t.dir %t.hex %t.obj && mkdir %t.dir %t.hex
// RUN: python3 %S/json-to-hex.py %S/sim-nested-matmul-input.json %t.hex/
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation{data-dir=%t.hex/},export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/ -Wno-LATCH" %t.dir/top.sv %t.dir/matmul.sv %t.dir/loop0.sv %t.dir/loop0_loop1.sv %t.dir/loop0_loop1_loop2.sv %t.dir/loop0_fsm.sv %t.dir/loop0_loop1_fsm.sv %t.dir/loop0_loop1_loop2_fsm.sv --cycles 2000 2>&1 | FileCheck %s

// Triply nested 2x2 matrix multiply: C = A * B using 1D memrefs (row-major).
//
// A = [[1, 2], [3, 4]]  flat: [1, 2, 3, 4]
// B = [[5, 6], [7, 8]]  flat: [5, 6, 7, 8]
// C = [[19, 22], [43, 50]]  flat: [19, 22, 43, 50]
//                                  [0x13, 0x16, 0x2b, 0x32]
//
// Three nested loops: outer(i=0..1), middle(j=0..1), inner(k=0..1).
// Inner body: acc += A[i*2+k] * B[k*2+j]
// After inner loop: C[i*2+j] = acc

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
        %cond_i = arith.cmpi ult, %i, %c2_i2 : i2
        loopschedule.register %cond_i : i1
      } do {
        %0 = loopschedule.step {
          // Middle loop: j = 0..1
          loopschedule.sequential trip_count = 2 iter_args(%j = %c0_i2) : (i2) -> () {
            %cond_j = arith.cmpi ult, %j, %c2_i2 : i2
            loopschedule.register %cond_j : i1
          } do {
            %1 = loopschedule.step {
              // Inner loop: k = 0..1, accumulating into acc
              %inner:2 = loopschedule.sequential trip_count = 2 iter_args(%k = %c0_i2, %acc = %c0_i32) : (i2, i32) -> (i2, i32) {
                %cond_k = arith.cmpi ult, %k, %c2_i2 : i2
                loopschedule.register %cond_k : i1
              } do {
                %2:2 = loopschedule.step {
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
                  %new_acc = arith.addi %acc, %prod : i32
                  // Advance k
                  %next_k = arith.addi %k, %c1_i2 : i2
                  loopschedule.register %next_k, %new_acc : i2, i32
                } : i2, i32
                loopschedule.terminator iter_args(%2#0, %2#1), results(%2#0, %2#1) : (i2, i32) -> (i2, i32)
              }
              // After inner loop: store C[i*2+j] = acc
              %i_ext2 = arith.extui %i : i2 to i4
              %j_ext2 = arith.extui %j : i2 to i4
              %i2val2 = arith.muli %i_ext2, %c2_i4 : i4
              %c_addr = arith.addi %i2val2, %j_ext2 : i4
              loopschedule.store %inner#1, %arg2[%c_addr : i4] : memref<4xi32>
              // Advance j
              %next_j = arith.addi %j, %c1_i2 : i2
              loopschedule.register %next_j : i2
            } : i2
            loopschedule.terminator iter_args(%1), results() : (i2) -> ()
          }
          // Advance i
          %next_i = arith.addi %i, %c1_i2 : i2
          loopschedule.register %next_i : i2
        } : i2
        loopschedule.terminator iter_args(%0), results() : (i2) -> ()
      }
    }
    return
  }
}
