// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" --split-input-file %s | FileCheck %s

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
// CHECK-LABEL: hw.module @cross_at_addr(
// CHECK:         hw.module @loop0(
// CHECK-NOT:     loop0_f0_at0_r2_latched
// CHECK:         %[[IDX:.+]] = arith.trunci %{{.+}} : i64 to i5
// CHECK-NOT:     loop0_f0_at0_r2_latched
// CHECK:         seq.compreg.ce sym @loop0_f0_at0_r3_latched
// CHECK:         comb.mux %{{.+}}, %[[IDX]], %{{.+}} : i5
// CHECK-NOT:     loop0_f0_at0_r2_latched
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
// CHECK:         hw.module @loop0(
// CHECK:         %[[HOLD:.+]] = seq.compreg.ce sym @loop0_f0_at0_r2_latched
// CHECK:         comb.mux %{{.+}}, %[[HOLD]], %{{.+}} : i5
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
