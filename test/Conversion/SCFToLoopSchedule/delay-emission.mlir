// RUN: circt-opt --pass-pipeline="builtin.module(func.func(scf-while-loop-flattening,canonicalize,mark-memory-accesses,construct-memory-dependencies,convert-memref-to-loopschedule,index-removal,convert-scf-to-loopschedule{enable-delay-merging=true}))" %s | FileCheck %s

// Verify that with `enable-delay-merging=true`, the SCFToLoopSchedule pass
// coalesces ops with overlapping start times into a single multi-cycle
// `loopschedule.step`, wrapping later-offset ops in `loopschedule.delay`
// regions. The kernel below performs a multiply-then-load-then-add chain
// where the multiply has multi-cycle latency, leaving room for the load to
// be issued mid-step.

// CHECK-LABEL: func.func @bucket_merge
// CHECK: loopschedule.sequential
// CHECK: loopschedule.step
// CHECK: arith.muli
// CHECK: loopschedule.delay
// CHECK: loopschedule.load
// CHECK: loopschedule.register
// CHECK: -> {{.*}}i32
module {
  func.func @bucket_merge(%in: memref<16xi32>,
                          %w:  memref<8xi32>,
                          %out: memref<18xi32>) attributes {top} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %0 = scf.while (%oc = %c0) : (i32) -> i32 {
      %c = arith.cmpi slt, %oc, %c2 : i32
      scf.condition(%c) %oc : i32
    } do {
    ^bb0(%oc: i32):
      %1 = scf.while (%ic = %c0) : (i32) -> i32 {
        %c = arith.cmpi slt, %ic, %c1 : i32
        scf.condition(%c) %ic : i32
      } do {
      ^bb0(%ic: i32):
        %2 = scf.while (%oh = %c0) : (i32) -> i32 {
          %c = arith.cmpi slt, %oh, %c3 : i32
          scf.condition(%c) %oh : i32
        } do {
        ^bb0(%oh: i32):
          %3 = scf.while (%ow = %c0) : (i32) -> i32 {
            %c = arith.cmpi slt, %ow, %c3 : i32
            scf.condition(%c) %ow : i32
          } do {
          ^bb0(%ow: i32):
            %4 = scf.while (%kh = %c0) : (i32) -> i32 {
              %c = arith.cmpi slt, %kh, %c2 : i32
              scf.condition(%c) %kh : i32
            } do {
            ^bb0(%kh: i32):
              %5 = scf.while (%kw = %c0) : (i32) -> i32 {
                %c = arith.cmpi slt, %kw, %c2 : i32
                scf.condition(%c) %kw : i32
              } do {
              ^bb0(%kw: i32):
                %ih = arith.addi %oh, %kh : i32
                %iw = arith.addi %ow, %kw : i32
                %ih4 = arith.muli %ih, %c4 : i32
                %in_addr = arith.addi %ih4, %iw : i32
                %kh4 = arith.muli %kh, %c4 : i32
                %kw2 = arith.muli %kw, %c2 : i32
                %w_tmp = arith.addi %kh4, %kw2 : i32
                %w_addr = arith.addi %w_tmp, %oc : i32
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
