// RUN: circt-opt --loopschedule-binding %s | FileCheck %s

// Memory ops are bound through the same mechanism as arith ops: each
// memref has an `oplib.operator` entry with `limit = 2`, and each load
// or store carries `loopschedule.operator = @<name>`. `OperatorAllocation`
// emits these entries/tags in the real flow; this dialect-level test
// spells them explicitly so it can run without the full pipeline.

oplib.library @lib {
  // Two memory operators, one per memref. No `TargetOp` — memory
  // operators don't participate in op-name-based library matching; the
  // scheduler sets its own `mem_<hash>` resource per memref, and the
  // binder keys off the `loopschedule.operator` symbol.
  oplib.operator @mem_a latency<1> , limit<2> {}
  oplib.operator @mem_b latency<1> , limit<2> {}
}

// Three loads on a 2-port memref in a single frame.
//
// Start times: two at cycle 0, one at cycle 1. With only 2 ports:
//   - The two cycle-0 loads conflict → distinct ports 0 and 1.
//   - The cycle-1 load has occupancy [1, 1], disjoint from [0, 0], so
//     it shares port 0 (lowest free).

// CHECK-LABEL: loopschedule.func_sequential @three_loads
loopschedule.func_sequential @three_loads(%a: memref<16xi32>)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %c2 = arith.constant 2 : i4
  loopschedule.frame {
    loopschedule.at 0 {
      // CHECK: loopschedule.load %arg0[%c0_i4
      // CHECK-SAME: loopschedule.binding = 0
      %v1 = loopschedule.load %a[%c0 : i4]
          {loopschedule.operator = @mem_a} : memref<16xi32>
      // CHECK: loopschedule.load %arg0[%c1_i4
      // CHECK-SAME: loopschedule.binding = 1
      %v2 = loopschedule.load %a[%c1 : i4]
          {loopschedule.operator = @mem_a} : memref<16xi32>
    }
    loopschedule.at 1 {
      // CHECK: loopschedule.load %arg0[%c2_i4
      // CHECK-SAME: loopschedule.binding = 0
      %v3 = loopschedule.load %a[%c2 : i4]
          {loopschedule.operator = @mem_a} : memref<16xi32>
    }
  }
  loopschedule.return
}

// Pipelined accesses with II=2, three stages accessing the same 2-port
// memory. Residues mod 2: stage 0 → {0}, stage 1 → {1}, stage 2 → {0}.
// Stages 0 and 2 share a residue and conflict. Stage 1 is disjoint.
// Binding: 0, 0, 1.
// CHECK-LABEL: loopschedule.func_sequential @pipelined_ii2
loopschedule.func_sequential @pipelined_ii2(%a: memref<64xi32>)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i7
  %c1 = arith.constant 1 : i7
  %zero_i32 = arith.constant 0 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %s = loopschedule.at 0 -> !loopschedule.handle {
      %l = loopschedule.launch : !loopschedule.handle {
        %p = loopschedule.pipeline II = 2 trip_count = 16 iter_args(%i = %c0) : (i7) -> i7 {
          %r:3 = loopschedule.at 0 -> (i7, i32, i1) {
            // CHECK: loopschedule.load
            // CHECK-SAME: loopschedule.binding = 0
            %idx = arith.trunci %i : i7 to i6
            %v = loopschedule.load %a[%idx : i6]
                {loopschedule.operator = @mem_a} : memref<64xi32>
            %next = arith.addi %i, %c1 : i7
            loopschedule.iter_arg_update %i = %next : i7
            %cond = arith.cmpi slt, %next, %c0 : i7
            loopschedule.yield %next, %v, %cond : i7, i32, i1
          }
          %s1:2 = loopschedule.at 1 -> (i7, i32) {
            // CHECK: loopschedule.store
            // CHECK-SAME: loopschedule.binding = 0
            %idx = arith.trunci %r#0 : i7 to i6
            loopschedule.store %zero_i32, %a[%idx : i6]
                {loopschedule.operator = @mem_a} : memref<64xi32>
            loopschedule.yield %r#0, %r#1 : i7, i32
          }
          %s2 = loopschedule.at 2 -> i7 {
            // CHECK: loopschedule.load
            // CHECK-SAME: loopschedule.binding = 1
            %idx = arith.trunci %s1#0 : i7 to i6
            %v = loopschedule.load %a[%idx : i6]
                {loopschedule.operator = @mem_a} : memref<64xi32>
            loopschedule.yield %s1#0 : i7
          }
          loopschedule.terminator condition(%r#2), results(%r#0) : i7
        }
        loopschedule.yield %p : i7
      }
      loopschedule.yield %l : !loopschedule.handle
    }
    loopschedule.yield %s : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return
}

// Two memrefs, distinct operators — separate pools.
// CHECK-LABEL: loopschedule.func_sequential @two_memrefs
loopschedule.func_sequential @two_memrefs(%a: memref<16xi32>, %b: memref<16xi32>)
    attributes {oplib.library = @lib} {
  %c0 = arith.constant 0 : i4
  %c1 = arith.constant 1 : i4
  %zero = arith.constant 0 : i32
  loopschedule.frame {
    loopschedule.at 0 {
      // CHECK: loopschedule.store %c0_i32, %arg0
      // CHECK-SAME: loopschedule.binding = 0
      loopschedule.store %zero, %a[%c0 : i4]
          {loopschedule.operator = @mem_a} : memref<16xi32>
      // CHECK: loopschedule.store %c0_i32, %arg1
      // CHECK-SAME: loopschedule.binding = 0
      loopschedule.store %zero, %b[%c0 : i4]
          {loopschedule.operator = @mem_b} : memref<16xi32>
    }
    loopschedule.at 1 {
      // CHECK: loopschedule.load %arg0[%c1_i4
      // CHECK-SAME: loopschedule.binding = 0
      %va = loopschedule.load %a[%c1 : i4]
          {loopschedule.operator = @mem_a} : memref<16xi32>
      // CHECK: loopschedule.load %arg1[%c1_i4
      // CHECK-SAME: loopschedule.binding = 0
      %vb = loopschedule.load %b[%c1 : i4]
          {loopschedule.operator = @mem_b} : memref<16xi32>
    }
  }
  loopschedule.return
}
