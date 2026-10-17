// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s --implicit-check-not=loop0_s1_r0 --implicit-check-not=loop0_s3_r

// A pipeline's stage registers are delays: each carries its value into the
// next stages and has no reset, and its copies shift with it, so
// synthesis can put a long chain in shift-register LUTs. Only a register
// read later holds its value between its stage's firings, on the stage's
// clock enable: an iter-arg's updated value (stage 0 here), which the next
// iteration reads up to II cycles on, and a loop result (stage 4), read
// after the drain.
//
// Stages 1 and 3 pass `v` and `w` on (as SCFToLoopSchedule forwards a
// value through every stage up to its last use): those results are the
// values themselves, read through their copies, with no register of their
// own.
//
// At II 2 a register read past the next cycle keeps its value until the
// next iteration's overwrites it, two cycles on: the registers of the
// stages in phase 0 of the II (0, 2, 4) are written in its cycles (when an
// iteration is in one of them, stage 0 counted from its issue slot) and
// read for the two after. `v` (stage 0), read in stages 2 and 4, takes its
// register and a copy for the next two cycles; `w` (stage 2), read in
// stage 4, its register. `u` and `y`, read the next cycle, and the
// condition shift on the stall gate every cycle, as at II 1.

// CHECK-LABEL: hw.module @ii2
// CHECK:       %[[PHASE0_ANY:[0-9]+]] = comb.or %{{[0-9]+}}, %loop0_ce_stage_2, %loop0_ce_stage_4 : i1
// CHECK-NEXT:  %[[PHASE0:[0-9]+]] = comb.and %[[PHASE0_ANY]], %[[NOT_STALL:[0-9]+]] : i1
// CHECK:       %loop0_s0_r0 = seq.compreg.ce sym @loop0_s0_r0 %{{[0-9]+}}, %clk, %{{[0-9]+}} reset %rst, %{{.+}} : index
// CHECK-NEXT:  %loop0_s0_r1 = seq.compreg.ce sym @loop0_s0_r1 %{{[0-9]+}}, %clk, %[[PHASE0]] : i32
// CHECK-NEXT:  %loop0_s0_r2 = seq.compreg.ce sym @loop0_s0_r2 %{{[0-9]+}}, %clk, %[[NOT_STALL]] : i1
// CHECK-NEXT:  %loop0_s0_r3 = seq.compreg.ce sym @loop0_s0_r3 %{{[0-9]+}}, %clk, %[[NOT_STALL]] : i32
// CHECK-NEXT:  %[[Y:[0-9]+]] = arith.addi %loop0_s0_r3, %c1_i32 : i32
// CHECK-NEXT:  %loop0_s1_r1 = seq.compreg.ce sym @loop0_s1_r1 %[[Y]], %clk, %[[NOT_STALL]] : i32
// CHECK-NEXT:  %[[W:[0-9]+]] = arith.addi %loop0_s0_r1, %loop0_s1_r1 : i32
// CHECK-NEXT:  %loop0_s2_r1 = seq.compreg.ce sym @loop0_s2_r1 %[[W]], %clk, %[[PHASE0]] : i32
// CHECK-NEXT:  %loop0_s2_dly_r1_from_s0 = seq.compreg.ce sym @loop0_s2_dly_r1_from_s0 %loop0_s0_r1, %clk, %[[PHASE0]] : i32
// CHECK-NEXT:  %[[X:[0-9]+]] = arith.addi %loop0_s2_dly_r1_from_s0, %loop0_s2_r1 : i32
// CHECK-NOT:   _dly_
// CHECK:       %loop0_s4_r0 = seq.compreg.ce sym @loop0_s4_r0 %[[X]], %clk, %{{[0-9]+}} reset %rst, %{{.+}} : i32

loopschedule.func_sequential @ii2(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 2 iter_args(%i = %c0) : (index) -> i32 {
          %0:4 = loopschedule.at 0 -> (index, i32, i1, i32) {
            %cond = arith.cmpi ult, %i, %c10 : index
            %next_i = arith.addi %i, %c1 : index
            %ii = arith.index_cast %i : index to i32
            %v = arith.addi %ii, %arg0 : i32
            %u = arith.xori %ii, %arg0 : i32
            loopschedule.iter_arg_update %i = %next_i : index
            loopschedule.yield %next_i, %v, %cond, %u : index, i32, i1, i32
          }
          %1:2 = loopschedule.at 1 -> (i32, i32) {
            %y = arith.addi %0#3, %c1_i32 : i32
            loopschedule.yield %0#1, %y : i32, i32
          }
          %2:2 = loopschedule.at 2 -> (i32, i32) {
            %w = arith.addi %1#0, %1#1 : i32
            loopschedule.yield %1#0, %w : i32, i32
          }
          %3:2 = loopschedule.at 3 -> (i32, i32) {
            loopschedule.yield %2#0, %2#1 : i32, i32
          }
          %4 = loopschedule.at 4 -> i32 {
            %x = arith.addi %3#0, %3#1 : i32
            loopschedule.yield %x : i32
          }
          loopschedule.terminator condition(%0#2), results(%4) : i32
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

// At II 1 the pipeline moves every cycle (on the stall gate), and a value
// takes a register per cycle it is read after its stage.

// CHECK-LABEL: hw.module @ii1
// CHECK:       %loop0_s0_r0 = seq.compreg.ce sym @loop0_s0_r0 %{{[0-9]+}}, %clk, %{{[0-9]+}} reset %rst, %{{.+}} : index
// CHECK-NEXT:  %loop0_s0_r1 = seq.compreg.ce sym @loop0_s0_r1 %{{[0-9]+}}, %clk, %[[SHIFT:[0-9]+]] : i32
// CHECK-NEXT:  %loop0_s0_r2 = seq.compreg.ce sym @loop0_s0_r2 %{{[0-9]+}}, %clk, %[[SHIFT]] : i1
// CHECK-NEXT:  %loop0_s0_r3 = seq.compreg.ce sym @loop0_s0_r3 %{{[0-9]+}}, %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %[[Y1:[0-9]+]] = arith.addi %loop0_s0_r3, %c1_i32 : i32
// CHECK-NEXT:  %loop0_s1_r1 = seq.compreg.ce sym @loop0_s1_r1 %[[Y1]], %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %loop0_s1_dly_r1_from_s0 = seq.compreg.ce sym @loop0_s1_dly_r1_from_s0 %loop0_s0_r1, %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %[[W1:[0-9]+]] = arith.addi %loop0_s1_dly_r1_from_s0, %loop0_s1_r1 : i32
// CHECK-NEXT:  %loop0_s2_r1 = seq.compreg.ce sym @loop0_s2_r1 %[[W1]], %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %loop0_s2_dly_r1_from_s0 = seq.compreg.ce sym @loop0_s2_dly_r1_from_s0 %loop0_s1_dly_r1_from_s0, %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %loop0_s3_dly_r1_from_s0 = seq.compreg.ce sym @loop0_s3_dly_r1_from_s0 %loop0_s2_dly_r1_from_s0, %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %loop0_s3_dly_r1_from_s2 = seq.compreg.ce sym @loop0_s3_dly_r1_from_s2 %loop0_s2_r1, %clk, %[[SHIFT]] : i32
// CHECK-NEXT:  %[[X1:[0-9]+]] = arith.addi %loop0_s3_dly_r1_from_s0, %loop0_s3_dly_r1_from_s2 : i32
// CHECK:       %loop0_s4_r0 = seq.compreg.ce sym @loop0_s4_r0 %[[X1]], %clk, %{{[0-9]+}} reset %rst, %{{.+}} : i32

loopschedule.func_sequential @ii1(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c1_i32 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip = loopschedule.pipeline II = 1 iter_args(%i = %c0) : (index) -> i32 {
          %0:4 = loopschedule.at 0 -> (index, i32, i1, i32) {
            %cond = arith.cmpi ult, %i, %c10 : index
            %next_i = arith.addi %i, %c1 : index
            %ii = arith.index_cast %i : index to i32
            %v = arith.addi %ii, %arg0 : i32
            %u = arith.xori %ii, %arg0 : i32
            loopschedule.iter_arg_update %i = %next_i : index
            loopschedule.yield %next_i, %v, %cond, %u : index, i32, i1, i32
          }
          %1:2 = loopschedule.at 1 -> (i32, i32) {
            %y = arith.addi %0#3, %c1_i32 : i32
            loopschedule.yield %0#1, %y : i32, i32
          }
          %2:2 = loopschedule.at 2 -> (i32, i32) {
            %w = arith.addi %1#0, %1#1 : i32
            loopschedule.yield %1#0, %w : i32, i32
          }
          %3:2 = loopschedule.at 3 -> (i32, i32) {
            loopschedule.yield %2#0, %2#1 : i32, i32
          }
          %4 = loopschedule.at 4 -> i32 {
            %x = arith.addi %3#0, %3#1 : i32
            loopschedule.yield %x : i32
          }
          loopschedule.terminator condition(%0#2), results(%4) : i32
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
