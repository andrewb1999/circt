// RUN: amc-opt --lower-loopschedule-to-fsm %s | FileCheck %s --implicit-check-not=epilogue_delay

// A pipeline is done once the loop has exited (`epilogue`) and nothing is
// left in it: no iteration in stages 1 to 3 (the stage-CE chain), and no
// iter-arg init capture pending. The trip count here is a runtime value, so
// the loop can run no iteration and then returns %acc's init, captured when
// stage 2, its first reader, would first have read it: two cycles after the
// start (`preload_start_d2`). Each of these is a shift register, so `busy`
// registers the OR of what they shift in (stage 0 is idle once the loop has
// exited), and done is the AND of two registers.

// CHECK-LABEL: hw.module @sum
// CHECK-DAG: %loop0_ce_stage_1 = seq.compreg.ce sym @loop0_ce_stage_1
// CHECK-DAG: %loop0_ce_stage_2 = seq.compreg.ce sym @loop0_ce_stage_2 %loop0_ce_stage_1,
// CHECK-DAG: %loop0_ce_stage_3 = seq.compreg.ce sym @loop0_ce_stage_3 %loop0_ce_stage_2,
// CHECK-DAG: %loop0_preload_start_d2 = seq.compreg.ce sym @loop0_preload_start_d2 %loop0_preload_start_d1,
// CHECK-DAG: %loop0_init_r1 = seq.compreg.ce sym @loop0_init_r1 %arg1, %clk, %loop0_preload_start_d2
// CHECK-DAG: %[[AHEAD:.+]] = comb.or %loop0_ce_stage_1, %loop0_ce_stage_2, %loop0_preload_start_d1 : i1
// CHECK-DAG: %loop0_busy = seq.compreg.ce sym @loop0_busy %[[AHEAD]],
// CHECK-DAG: %[[IDLE:.+]] = comb.xor %loop0_busy, %true
// CHECK-DAG: comb.and %loop0_epilogue, %[[IDLE]] : i1

loopschedule.func_sequential @sum(%x: i32, %a: i32, %n: i32) -> i32 attributes {top} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %hp = loopschedule.at 0 -> !loopschedule.handle {
      %hp_launch = loopschedule.launch : !loopschedule.handle {
        %pip:2 = loopschedule.pipeline II = 1 latency = 4 iter_args(%i = %c0, %acc = %a) : (i32, i32) -> (i32, i32) {
          %r:2 = loopschedule.at 0 -> (i32, i1) {
            %cond = arith.cmpi slt, %i, %n : i32
            %next = arith.addi %i, %c1 : i32
            loopschedule.iter_arg_update %i = %next : i32
            loopschedule.yield %next, %cond : i32, i1
          }
          %s1 = loopschedule.at 1 -> i32 {
            loopschedule.yield %r#0 : i32
          }
          %s2 = loopschedule.at 2 -> i32 {
            %y = arith.addi %acc, %x : i32
            loopschedule.iter_arg_update %acc = %y : i32
            loopschedule.yield %y : i32
          }
          loopschedule.at 3 {
            loopschedule.yield
          }
          loopschedule.terminator condition(%r#1), results(%s1, %s2) : i32, i32
        }
        loopschedule.yield %pip#1 : i32
      }
      loopschedule.yield %hp_launch : !loopschedule.handle
    }
    loopschedule.yield %hp : !loopschedule.handle
  }
  %res = loopschedule.frame -> (i32) {
    %v = loopschedule.await %h -> i32
    loopschedule.yield %v : i32
  } do (%v: i32) {
    %r = loopschedule.at 0 -> i32 {
      loopschedule.yield %v : i32
    }
    loopschedule.yield %r : i32
  }
  loopschedule.return %res : i32
}
