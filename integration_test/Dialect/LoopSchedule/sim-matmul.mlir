// REQUIRES: verilator
// RUN: rm -rf %t.dir %t.hex %t.obj && mkdir %t.dir %t.hex
// RUN: python3 %S/json-to-hex.py %S/sim-matmul-input.json %t.hex/
// RUN: circt-opt --pass-pipeline="builtin.module(lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation{data-dir=%t.hex/},export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/" %t.dir/top.sv %t.dir/matmul.sv %t.dir/loop0.sv %t.dir/loop0_fsm.sv --cycles 2000 2>&1 | FileCheck %s

// Flat 2x2 matrix multiply: C = A * B using 1D memrefs (row-major).
//
// A = [[1, 2], [3, 4]]  flat: [1, 2, 3, 4]
// B = [[5, 6], [7, 8]]  flat: [5, 6, 7, 8]
// C = [[19, 22], [43, 50]]  flat: [19, 22, 43, 50]
//                                  [0x13, 0x16, 0x2b, 0x32]
//
// Single sequential loop with 8 iterations (flat_idx = 0..7).
// Encoding: out_idx = flat_idx >> 1, k = flat_idx & 1
//   row = out_idx >> 1, col = out_idx & 1
// Even flat_idx (k=0): acc = A[row*2+0] * B[0*2+col]
// Odd flat_idx  (k=1): acc = acc + A[row*2+1] * B[1*2+col], store to C[out_idx]

// CHECK: @MEM mem2 4
// CHECK-NEXT: 00000013
// CHECK-NEXT: 00000016
// CHECK-NEXT: 0000002b
// CHECK-NEXT: 00000032
// CHECK: DONE

module {
  func.func @matmul(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>) attributes {top} {
    %c0_i32 = arith.constant 0 : i32
    %c0_i4 = arith.constant 0 : i4
    %c8_i4 = arith.constant 8 : i4
    %c1_i4 = arith.constant 1 : i4
    %c2_i4 = arith.constant 2 : i4
    loopschedule.step {
      loopschedule.sequential trip_count = 8 iter_args(%flat = %c0_i4, %acc = %c0_i32) : (i4, i32) -> () {
        %0:3 = loopschedule.step {
          %cond = arith.cmpi ult, %flat, %c8_i4 : i4
          // Decode flat index.
          %out_idx = arith.shrui %flat, %c1_i4 : i4
          %k = arith.andi %flat, %c1_i4 : i4
          %row = arith.shrui %out_idx, %c1_i4 : i4
          %col = arith.andi %out_idx, %c1_i4 : i4
          // A_addr = row*2 + k, B_addr = k*2 + col.
          %row2 = arith.muli %row, %c2_i4 : i4
          %a_addr = arith.addi %row2, %k : i4
          %k2 = arith.muli %k, %c2_i4 : i4
          %b_addr = arith.addi %k2, %col : i4
          // Load A[a_addr] and B[b_addr], compute product.
          %a_val = loopschedule.load %arg0[%a_addr : i4] : memref<4xi32>
          %b_val = loopschedule.load %arg1[%b_addr : i4] : memref<4xi32>
          %prod = arith.muli %a_val, %b_val : i32
          // Accumulate: k=0 starts fresh, k=1 adds to previous.
          %is_k0 = arith.cmpi eq, %k, %c0_i4 : i4
          %sum = arith.addi %acc, %prod : i32
          %new_acc = arith.select %is_k0, %prod, %sum : i32
          // Store result to C[out_idx] (intermediate on k=0, final on k=1).
          loopschedule.store %new_acc, %arg2[%out_idx : i4] : memref<4xi32>
          // Advance.
          %next_flat = arith.addi %flat, %c1_i4 : i4
          loopschedule.iter_arg_update %acc = %new_acc : i32
          loopschedule.iter_arg_update %flat = %next_flat : i4
          loopschedule.register %next_flat, %new_acc, %cond : i4, i32, i1
        } : i4, i32, i1
        loopschedule.terminator condition(%0#2), results()
      }
    }
    return
  }
}
