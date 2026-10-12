// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{enable-pipeline-prearm=true})" --split-input-file %s | FileCheck %s

// A frame that contains a LAUNCH is lowered by the wait-frame path, which
// holds any at-op result a later at-op reads in a capture register: a static
// load's `rd_data` is only meaningful while the frame owns the port, and the
// launched child takes the port over during WAIT_i.
//
// That register's clock enable is the at-offset+1 cycle gate, so its value is
// READABLE from at-offset+2. Handing it to the at-op that runs AT
// at-offset+1 gives that at-op the value the PREVIOUS outer iteration left
// behind -- which is how `lo = rowptr[i]` came back as `rowptr[i-1]`, one row
// late, on the AXI bus itself (the second address of every row, not the data
// that came back for it).
//
// So: at-offset K+1 keeps the combinational value, K+2 and beyond get the
// register. The combinational value is good for the whole frame -- a
// single-access port holds its address frame-wide, a contended port's load
// result is already a live-bypassed capture register -- so nothing else has
// to move.

// THE BUG SHAPE: `%idx` (at-result 2) is produced in `at 0` and read by
// `at 1`'s load address, in a frame that launches a child at `at 2`. No hold
// register may stand between them: `at 1` runs in the cycle such a register
// would be CLOCKED, so it drives the live truncation of the iteration
// argument straight onto the port.
//
// `%hi` (at-result 3) is the control: it is read by the launched child, which
// runs during WAIT_0 with the port handed over, so IT still gets a hold.
// The loop now inlines into the function module itself (no hw.module @loop0);
// the hold registers and muxes live in @cross_at_addr's body, gated by the
// machine's loop0_frame_cycle_* results.
// CHECK-LABEL: hw.module @cross_at_addr(
// CHECK-NOT:     loop0_f0_at0_r2_latched
// CHECK:         %[[IDX:.+]] = arith.trunci %{{.+}} : i64 to i5
// CHECK-NOT:     loop0_f0_at0_r2_latched
// CHECK:         seq.compreg.ce sym @loop0_f0_at0_r3_latched
// CHECK:         comb.mux %{{.+}}, %[[IDX]], %{{.+}} : i5
// CHECK-NOT:     loop0_f0_at0_r2_latched
//
// The cross-at hold shape also refuses drain-edge pre-arming (gap-9):
// the inner pipeline reads same-frame at results, so its stage 0 cannot
// issue in the child_start cycle — the issue gate stays plain `active`.
// CHECK-NOT:     loop0_pip0_done_prev
// CHECK:         fsm.machine @cross_at_addr_fsm
loopschedule.func_sequential @cross_at_addr(%mem: memref<32xi64>, %n: i64) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1_i5 = arith.constant 1 : i5
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0_i64) : (i64) -> i64 {
          %f:5 = loopschedule.frame -> (i1, i64, i5, i64, !loopschedule.handle) {
            %s0:4 = loopschedule.at 0 -> (i1, i64, i5, i64) {
              %cond = arith.cmpi slt, %i, %n : i64
              %inext = arith.addi %i, %c1_i64 : i64
              %idx = arith.trunci %i : i64 to i5
              %idx1 = arith.addi %idx, %c1_i5 : i5
              %hi = loopschedule.load %mem[%idx1 : i5] : memref<32xi64>
              loopschedule.iter_arg_update %i = %inext : i64
              loopschedule.yield %cond, %inext, %idx, %hi : i1, i64, i5, i64
            }
            %s1 = loopschedule.at 1 -> i64 {
              %lo = loopschedule.load %mem[%s0#2 : i5] : memref<32xi64>
              loopschedule.yield %lo : i64
            }
            %s2 = loopschedule.at 2 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %pip = loopschedule.pipeline II = 1 iter_args(%k = %c0_i64) : (i64) -> i64 {
                  %p0:2 = loopschedule.at 0 -> (i64, i1) {
                    %c = arith.cmpi slt, %k, %s0#3 : i64
                    %knext = arith.addi %k, %c1_i64 : i64
                    loopschedule.iter_arg_update %k = %knext : i64
                    loopschedule.yield %knext, %c : i64, i1
                  }
                  loopschedule.terminator condition(%p0#1), results(%p0#0) : i64
                }
                loopschedule.yield %pip : i64
              }
              loopschedule.yield %inner : !loopschedule.handle
            }
            loopschedule.yield %s0#0, %s0#1, %s0#2, %s1, %s2 : i1, i64, i5, i64, !loopschedule.handle
          }
          loopschedule.frame await {
            loopschedule.await %f#4
          }
          loopschedule.terminator condition(%f#0), results(%f#1) : i64
        }
        loopschedule.yield %seq : i64
      }
      loopschedule.yield %launch : !loopschedule.handle
    }
    loopschedule.yield %outer : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return
}

// -----

// THE NEGATIVE CONTROL: same frame, same launch, but the consumer sits at
// `at 2` -- two cycles after the producer, which is exactly when the hold
// register becomes readable. The register must still be there; the fix
// narrows who may read it, it does not delete the mechanism.
// CHECK-LABEL: hw.module @cross_at_addr_far(
// CHECK:         %[[HOLD:.+]] = seq.compreg.ce sym @loop0_f0_at0_r2_latched
// CHECK:         comb.mux %{{.+}}, %[[HOLD]], %{{.+}} : i5
// Same-frame reads refuse pre-arming here too.
// CHECK-NOT:     loop0_pip0_done_prev
// CHECK:         fsm.machine @cross_at_addr_far_fsm
loopschedule.func_sequential @cross_at_addr_far(%mem: memref<32xi64>, %n: i64) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c1_i5 = arith.constant 1 : i5
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0_i64) : (i64) -> i64 {
          %f:5 = loopschedule.frame -> (i1, i64, i5, i64, !loopschedule.handle) {
            %s0:4 = loopschedule.at 0 -> (i1, i64, i5, i64) {
              %cond = arith.cmpi slt, %i, %n : i64
              %inext = arith.addi %i, %c1_i64 : i64
              %idx = arith.trunci %i : i64 to i5
              %idx1 = arith.addi %idx, %c1_i5 : i5
              %hi = loopschedule.load %mem[%idx1 : i5] : memref<32xi64>
              loopschedule.iter_arg_update %i = %inext : i64
              loopschedule.yield %cond, %inext, %idx, %hi : i1, i64, i5, i64
            }
            %s1 = loopschedule.at 2 -> i64 {
              %lo = loopschedule.load %mem[%s0#2 : i5] : memref<32xi64>
              loopschedule.yield %lo : i64
            }
            %s2 = loopschedule.at 3 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %pip = loopschedule.pipeline II = 1 iter_args(%k = %c0_i64) : (i64) -> i64 {
                  %p0:2 = loopschedule.at 0 -> (i64, i1) {
                    %c = arith.cmpi slt, %k, %s0#3 : i64
                    %knext = arith.addi %k, %c1_i64 : i64
                    loopschedule.iter_arg_update %k = %knext : i64
                    loopschedule.yield %knext, %c : i64, i1
                  }
                  loopschedule.terminator condition(%p0#1), results(%p0#0) : i64
                }
                loopschedule.yield %pip : i64
              }
              loopschedule.yield %inner : !loopschedule.handle
            }
            loopschedule.yield %s0#0, %s0#1, %s0#2, %s1, %s2 : i1, i64, i5, i64, !loopschedule.handle
          }
          loopschedule.frame await {
            loopschedule.await %f#4
          }
          loopschedule.terminator condition(%f#0), results(%f#1) : i64
        }
        loopschedule.yield %seq : i64
      }
      loopschedule.yield %launch : !loopschedule.handle
    }
    loopschedule.yield %outer : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return
}

// -----

// A wait frame's RESULTS are read after the launched child has run, so an
// at-result the frame yields needs the hold as much as one the child reads:
// the child walks the same port during WAIT_0 and leaves rd_data holding its
// own last load. This is trisolv's shape: `L[i][i]`, loaded ahead of the
// `j` loop over row i of L and read after it, came back as `L[i][i-1]`.
// CHECK-LABEL: hw.module @frame_result_hold(
// CHECK:         %[[HOLD:.+]] = seq.compreg.ce sym @loop0_f0_at0_r2_latched
// CHECK:         arith.addi %{{.+}}, %[[HOLD]] : i32
// CHECK:         fsm.machine @frame_result_hold_fsm
loopschedule.func_sequential @frame_result_hold(%L: memref<16xi32>, %x: memref<16xi32>) {
  %c0_i4 = arith.constant 0 : i4
  %c1_i4 = arith.constant 1 : i4
  %c2_i4 = arith.constant 2 : i4
  %c4_i4 = arith.constant 4 : i4
  %c0_i32 = arith.constant 0 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential trip_count = 4 iter_args(%i = %c0_i4) : (i4) -> i4 {
          %f:4 = loopschedule.frame -> (i1, i4, i32, !loopschedule.handle) {
            %s0:3 = loopschedule.at 0 -> (i1, i4, i32) {
              %cond = arith.cmpi slt, %i, %c4_i4 : i4
              %row = arith.shli %i, %c2_i4 : i4
              %didx = arith.addi %row, %i : i4
              %diag = loopschedule.load %L[%didx : i4] : memref<16xi32>
              loopschedule.yield %cond, %row, %diag : i1, i4, i32
            }
            %s1 = loopschedule.at 1 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %sq:2 = loopschedule.sequential iter_args(%j = %c0_i4, %acc = %c0_i32) : (i4, i32) -> (i4, i32) {
                  %g:3 = loopschedule.frame -> (i1, i4, i32) {
                    %p0:3 = loopschedule.at 0 -> (i1, i4, i32) {
                      %c = arith.cmpi slt, %j, %i : i4
                      %jnext = arith.addi %j, %c1_i4 : i4
                      %eidx = arith.addi %s0#1, %j : i4
                      %e = loopschedule.load %L[%eidx : i4] : memref<16xi32>
                      loopschedule.iter_arg_update %j = %jnext : i4
                      loopschedule.yield %c, %jnext, %e : i1, i4, i32
                    }
                    %p1 = loopschedule.at 1 -> i32 {
                      %sum = arith.addi %acc, %p0#2 : i32
                      loopschedule.iter_arg_update %acc = %sum : i32
                      loopschedule.yield %sum : i32
                    }
                    loopschedule.yield %p0#0, %p0#1, %p1 : i1, i4, i32
                  }
                  loopschedule.terminator condition(%g#0), results(%g#1, %g#2) : i4, i32
                }
                loopschedule.yield %sq#0, %sq#1 : i4, i32
              }
              loopschedule.yield %inner : !loopschedule.handle
            }
            loopschedule.yield %s0#0, %s0#1, %s0#2, %s1 : i1, i4, i32, !loopschedule.handle
          }
          %post = loopschedule.frame -> (i4) {
            %r = loopschedule.await %f#3 -> i32
            loopschedule.yield %r : i32
          } do (%acc: i32) {
            %n = loopschedule.at 0 -> i4 {
              %inext = arith.addi %i, %c1_i4 : i4
              %v = arith.addi %acc, %f#2 : i32
              loopschedule.store %v, %x[%i : i4] : memref<16xi32>
              loopschedule.iter_arg_update %i = %inext : i4
              loopschedule.yield %inext : i4
            }
            loopschedule.yield %n : i4
          }
          loopschedule.terminator condition(%f#0), results(%post) : i4
        }
        loopschedule.yield %seq : i4
      }
      loopschedule.yield %launch : !loopschedule.handle
    }
    loopschedule.yield %outer : !loopschedule.handle
  }
  loopschedule.frame await {
    loopschedule.await %h
  }
  loopschedule.return
}
