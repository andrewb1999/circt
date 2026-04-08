// REQUIRES: verilator
// RUN: rm -rf %t.dir %t.hex %t.obj && mkdir %t.dir %t.hex
// RUN: python3 %S/json-to-hex.py %S/sim-conv2d-input.json %t.hex/
// RUN: circt-opt --pass-pipeline="builtin.module(func.func(mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule),lower-loopschedule-to-fsm,convert-fsm-to-sv,map-arith-to-comb,canonicalize,lower-seq-to-sv,loopschedule-testbench-generation{data-dir=%t.hex/},export-split-verilog{dir-name=%t.dir})" %s -o /dev/null
// RUN: circt-rtl-sim.py --objdir %t.obj --compileargs="-I%t.dir/ -Wno-LATCH" %t.dir/top.sv %t.dir/conv2d.sv %t.dir/loop0.sv %t.dir/loop0_loop1.sv %t.dir/loop0_loop1_loop2.sv %t.dir/loop0_loop1_loop2_loop3.sv %t.dir/loop0_loop1_loop2_loop3_loop4.sv %t.dir/loop0_loop1_loop2_loop3_loop4_loop5.sv %t.dir/loop0_fsm.sv %t.dir/loop0_loop1_fsm.sv %t.dir/loop0_loop1_loop2_fsm.sv %t.dir/loop0_loop1_loop2_loop3_fsm.sv %t.dir/loop0_loop1_loop2_loop3_loop4_fsm.sv %t.dir/loop0_loop1_loop2_loop3_loop4_loop5_fsm.sv --cycles 5000 2>&1 | FileCheck %s

// 6-deep nested 2D conv: oc x ic x oh x ow x kh x kw, with 1D memrefs
// (MemrefToLoopSchedule does not yet support multi-dim memrefs).
//   in: 16xi32  (logical 4x4x1, row-major)
//   w:   8xi32  (logical 2x2x1x2 (kh,kw,ic,oc))
//   out: 18xi32 (logical 3x3x2)
//
// input ramps 1..16, weights ramp 1..8.
// expected output (row-major, [oh][ow][oc]):
//   [74, 88, 90, 108, 106, 128, 138, 168, 154, 188, 170, 208, 202, 248, 218, 268, 234, 288]
// hex: 4a 58 5a 6c 6a 80 8a a8 9a bc aa d0 ca f8 da 10c ea 120

// CHECK: @MEM mem2 32
// CHECK-NEXT: 0000004a
// CHECK-NEXT: 00000058
// CHECK-NEXT: 0000005a
// CHECK-NEXT: 0000006c
// CHECK-NEXT: 0000006a
// CHECK-NEXT: 00000080
// CHECK-NEXT: 0000008a
// CHECK-NEXT: 000000a8
// CHECK-NEXT: 0000009a
// CHECK-NEXT: 000000bc
// CHECK-NEXT: 000000aa
// CHECK-NEXT: 000000d0
// CHECK-NEXT: 000000ca
// CHECK-NEXT: 000000f8
// CHECK-NEXT: 000000da
// CHECK-NEXT: 0000010c
// CHECK-NEXT: 000000ea
// CHECK-NEXT: 00000120
// CHECK: DONE

module {
  func.func @conv2d(%in: memref<16xi32>,
                    %w:  memref<8xi32>,
                    %out: memref<18xi32>) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    // oc: 0..2
    %0 = scf.while (%oc = %c0) : (i32) -> i32 {
      %c = arith.cmpi slt, %oc, %c2 : i32
      scf.condition(%c) %oc : i32
    } do {
    ^bb0(%oc: i32):
      // ic: 0..1
      %1 = scf.while (%ic = %c0) : (i32) -> i32 {
        %c = arith.cmpi slt, %ic, %c1 : i32
        scf.condition(%c) %ic : i32
      } do {
      ^bb0(%ic: i32):
        // oh: 0..3
        %2 = scf.while (%oh = %c0) : (i32) -> i32 {
          %c = arith.cmpi slt, %oh, %c3 : i32
          scf.condition(%c) %oh : i32
        } do {
        ^bb0(%oh: i32):
          // ow: 0..3
          %3 = scf.while (%ow = %c0) : (i32) -> i32 {
            %c = arith.cmpi slt, %ow, %c3 : i32
            scf.condition(%c) %ow : i32
          } do {
          ^bb0(%ow: i32):
            // kh: 0..2
            %4 = scf.while (%kh = %c0) : (i32) -> i32 {
              %c = arith.cmpi slt, %kh, %c2 : i32
              scf.condition(%c) %kh : i32
            } do {
            ^bb0(%kh: i32):
              // kw: 0..2
              %5 = scf.while (%kw = %c0) : (i32) -> i32 {
                %c = arith.cmpi slt, %kw, %c2 : i32
                scf.condition(%c) %kw : i32
              } do {
              ^bb0(%kw: i32):
                // in_addr = (oh+kh)*4 + (ow+kw)   (ic=0, dim=1)
                %ih = arith.addi %oh, %kh : i32
                %iw = arith.addi %ow, %kw : i32
                %ih4 = arith.muli %ih, %c4 : i32
                %in_addr = arith.addi %ih4, %iw : i32
                // w_addr = kh*4 + kw*2 + oc      (ic=0)
                %kh4 = arith.muli %kh, %c4 : i32
                %kw2 = arith.muli %kw, %c2 : i32
                %w_tmp = arith.addi %kh4, %kw2 : i32
                %w_addr = arith.addi %w_tmp, %oc : i32
                // out_addr = oh*6 + ow*2 + oc
                %oh6 = arith.muli %oh, %c6 : i32
                %ow2 = arith.muli %ow, %c2 : i32
                %out_tmp = arith.addi %oh6, %ow2 : i32
                %out_addr = arith.addi %out_tmp, %oc : i32
                %in_idx = arith.index_cast %in_addr : i32 to index
                %w_idx = arith.index_cast %w_addr : i32 to index
                %out_idx = arith.index_cast %out_addr : i32 to index
                %iv = memref.load %in[%in_idx] : memref<16xi32>
                %wv = memref.load %w[%w_idx] : memref<8xi32>
                %prod = arith.muli %iv, %wv : i32
                %acc = memref.load %out[%out_idx] : memref<18xi32>
                %sum = arith.addi %acc, %prod : i32
                memref.store %sum, %out[%out_idx] : memref<18xi32>
                %kwn = arith.addi %kw, %c1 : i32
                scf.yield %kwn : i32
              }
              %khn = arith.addi %kh, %c1 : i32
              scf.yield %khn : i32
            }
            %own = arith.addi %ow, %c1 : i32
            scf.yield %own : i32
          }
          %ohn = arith.addi %oh, %c1 : i32
          scf.yield %ohn : i32
        }
        %icn = arith.addi %ic, %c1 : i32
        scf.yield %icn : i32
      }
      %ocn = arith.addi %oc, %c1 : i32
      scf.yield %ocn : i32
    }
    return
  }
}
