// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s

// A pipeline's stage registers are delays: each carries its value into the
// next stage, which reads it the next cycle the pipeline moves, so it
// shifts on the stall gate alone and has no reset, like the rest of its
// chain (which lets synthesis put a long chain in shift-register LUTs).
// Only a register read later holds its value between its stage's firings,
// on the stage's clock enable: an iter-arg's updated value, which the next
// iteration reads up to II cycles on, and a loop result, read after the
// drain.

// CHECK-LABEL: hw.module @delays
// The iter-arg's update (stage 0) holds.
// CHECK:       %loop0_s0_r0 = seq.compreg.ce sym @loop0_s0_r0 %{{[0-9]+}}, %clk, %{{[0-9]+}} reset %rst, %{{.+}} : index
// The value stage 2 reads from stage 0, and the condition, shift.
// CHECK-NEXT:  %loop0_s0_r1 = seq.compreg.ce sym @loop0_s0_r1 %{{[0-9]+}}, %clk, %[[SHIFT:[0-9]+]] : i32
// CHECK-NEXT:  %loop0_s0_r2 = seq.compreg.ce sym @loop0_s0_r2 %{{[0-9]+}}, %clk, %[[SHIFT]] : i1
// CHECK:       %loop0_s1_r0 = seq.compreg.ce sym @loop0_s1_r0 %loop0_s0_r1, %clk, %[[SHIFT]] : i32
// The loop's result (stage 2) holds.
// CHECK:       %loop0_s2_r0 = seq.compreg.ce sym @loop0_s2_r0 %{{[0-9]+}}, %clk, %{{[0-9]+}} reset %rst, %{{.+}} : i32

loopschedule.func_sequential @delays(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 2 iter_args(%i = %c0) : (index) -> i32 {
          %0:3 = loopschedule.at 0 -> (index, i32, i1) {
            %cond = arith.cmpi ult, %i, %c10 : index
            %next_i = arith.addi %i, %c1 : index
            %ii = arith.index_cast %i : index to i32
            %v = arith.addi %ii, %arg0 : i32
            loopschedule.iter_arg_update %i = %next_i : index
            loopschedule.yield %next_i, %v, %cond : index, i32, i1
          }
          %1 = loopschedule.at 1 -> i32 {
            loopschedule.yield %0#1 : i32
          }
          %2 = loopschedule.at 2 -> i32 {
            %w = arith.addi %1, %c1_i32 : i32
            loopschedule.yield %w : i32
          }
          loopschedule.terminator condition(%0#2), results(%2) : i32
        }
        loopschedule.yield %pip : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    loopschedule.yield %hp : !loopschedule.handle
  }
  %result = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %r = loopschedule.at 0 -> i32 {
      loopschedule.yield %v : i32
    }
    loopschedule.yield %r : i32
  }
  loopschedule.return %result : i32
}
