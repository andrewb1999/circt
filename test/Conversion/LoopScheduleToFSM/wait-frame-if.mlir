// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A `loopschedule.if` in a frame that also launches a child (the wait-frame
// path) is lowered like one in a leaf frame: its ops issue under the frame
// gate AND the condition. The wait-frame path used to clone the `if` whole,
// leaving the store inside it on the original memory (a crash: the memory
// was erased with the function while the clone still used it).

// CHECK-LABEL: hw.module @wait_frame_if(
// CHECK:         %[[SMALL:.+]] = arith.cmpi slt, %{{.+}}, %c2_i64 : i64
// CHECK:         %[[EN:.+]] = comb.and %{{.+}}, %[[SMALL]] : i1
// CHECK:         comb.and %{{.+}}, %[[EN]] : i1
// CHECK:         fsm.machine @wait_frame_if_fsm
loopschedule.func_sequential @wait_frame_if(%mem: memref<16xi64>, %n: i64) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0_i64) : (i64) -> i64 {
          %f:3 = loopschedule.frame -> (i1, i64, !loopschedule.handle) {
            %s0:2 = loopschedule.at 0 -> (i1, i64) {
              %cond = arith.cmpi slt, %i, %n : i64
              %inext = arith.addi %i, %c1_i64 : i64
              %small = arith.cmpi slt, %i, %c2_i64 : i64
              %idx = arith.trunci %i : i64 to i4
              loopschedule.if %small {
                loopschedule.store %i, %mem[%idx : i4] : memref<16xi64>
              }
              loopschedule.iter_arg_update %i = %inext : i64
              loopschedule.yield %cond, %inext : i1, i64
            }
            %s1 = loopschedule.at 1 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %pip = loopschedule.pipeline II = 1 iter_args(%k = %c0_i64) : (i64) -> i64 {
                  %p0:2 = loopschedule.at 0 -> (i64, i1) {
                    %c = arith.cmpi slt, %k, %c2_i64 : i64
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
            loopschedule.yield %s0#0, %s0#1, %s1 : i1, i64, !loopschedule.handle
          }
          loopschedule.frame await {
            loopschedule.await %f#2
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
