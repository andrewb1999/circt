// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm{enable-pipeline-prearm=true})" --split-input-file %s | FileCheck %s

// A launched child reads its frame-produced operands out of the
// launch-consumer capture registers, which latch during cycle O+1 for a
// producer in `at O` and are register-readable from O+2. The SCHEDULE places a
// launch as soon as its operands are COMBINATIONALLY available, which is too
// early for an iter_arg INIT whose preload depth is small -- and the induction
// variable's depth is forced to 0, because the loop condition reads it
// unregistered.
//
// A loop with a runtime LOWER bound is exactly that shape:
// `for k in range(rowptr[i], rowptr[i+1])` inits its IV from a load in the
// enclosing frame. Started at its scheduled offset, the preload latched the
// PREVIOUS outer iteration's base and every row walked from the previous row's
// start -- plausible numbers, silently wrong. The child_start pulse is pushed
// to the first legal cycle instead, and the frame's cycle states extend to
// cover it.

// The load is in `at 0`, the launch was SCHEDULED at `at 1`, and the IV init is
// read at stage 0 => the pulse must wait until cycle 0 + 2. So the frame has
// THREE cycle states and child_start fires in the last of them, not the middle.
//
// This shape must also REFUSE drain-edge pre-arming (gap-9): the IV init
// is loaded fresh from the frame each iteration, so it cannot sit
// pre-armed in the feedback register through the idle window. The stage-0
// issue gate stays plain `active` — no `or(active, child_start)` exists.
// The loop now inlines into the single per-function machine, so the frame's
// cycle states are the prefixed loop0_FRAME_0_0..2.
// CHECK-NOT: loop0_pip0_done_prev
// CHECK-LABEL: fsm.machine @dyn_lower_bound_fsm
// CHECK-SAME:    resNames = [{{.*}}"loop0_child_start_0", "loop0_child_active_0", "loop0_frame_cycle_0_0", "loop0_frame_cycle_0_1", "loop0_frame_cycle_0_2"]
// CHECK:         fsm.state @loop0_FRAME_0_0 output
// CHECK-NEXT:      fsm.output %false, %false, %true, %false, %loop0_first_iter, %false, %true, %false, %false, %false, %true, %false, %false
// CHECK:         fsm.state @loop0_FRAME_0_1 output
// CHECK-NEXT:      fsm.output %false, %false, %true, %false, %loop0_first_iter, %false, %true, %false, %false, %false, %false, %true, %false
// loop0_child_start_0 (result 8) and loop0_child_active_0 (result 9) rise
// here, in cycle 2 — the last of the THREE cycle states.
// CHECK:         fsm.state @loop0_FRAME_0_2 output
// CHECK-NEXT:      fsm.output %false, %false, %true, %false, %loop0_first_iter, %false, %true, %false, %true, %true, %false, %false, %true
loopschedule.func_sequential @dyn_lower_bound(%rowptr: memref<17xi64>, %n: i64) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0_i64) : (i64) -> i64 {
          %f:4 = loopschedule.frame -> (i1, i64, i64, !loopschedule.handle) {
            %s0:3 = loopschedule.at 0 -> (i1, i64, i64) {
              %cond = arith.cmpi slt, %i, %n : i64
              %inext = arith.addi %i, %c1_i64 : i64
              %idx = arith.trunci %i : i64 to i5
              %lo = loopschedule.load %rowptr[%idx : i5] : memref<17xi64>
              loopschedule.iter_arg_update %i = %inext : i64
              loopschedule.yield %cond, %inext, %lo : i1, i64, i64
            }
            %s1 = loopschedule.at 1 -> i64 {
              %hi = arith.addi %s0#2, %c1_i64 : i64
              loopschedule.yield %hi : i64
            }
            %s2 = loopschedule.at 1 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %pip = loopschedule.pipeline II = 1 iter_args(%k = %s0#2) : (i64) -> i64 {
                  %p0:2 = loopschedule.at 0 -> (i64, i1) {
                    %c = arith.cmpi slt, %k, %s1 : i64
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
            loopschedule.yield %s0#0, %s0#1, %s1, %s2 : i1, i64, i64, !loopschedule.handle
          }
          loopschedule.frame await {
            loopschedule.await %f#3
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

// NEGATIVE CONTROL, and the reason this is not a blanket delay: the same nest
// with a STATIC lower bound inits its IV from a constant, which is stable long
// before the frame starts. Nothing constrains the pulse, the launch keeps its
// scheduled cycle 1, and the frame keeps two cycle states -- so every existing
// design (all of which have static lower bounds) lowers bit-identically.
//
// Pre-arming is still refused here — not by the init (constant) but by
// the launch-cycle timing bound: the pipeline body reads the same-frame
// captured `hi` bound, whose capture register is not yet readable in the
// cycle child_start pulses, so stage 0 cannot issue that cycle.
// CHECK-NOT: loop0_pip0_done_prev

// CHECK-LABEL: fsm.machine @static_lower_bound_fsm
// CHECK-SAME:    resNames = [{{.*}}"loop0_frame_cycle_0_0", "loop0_frame_cycle_0_1"]
// CHECK-NOT:     frame_cycle_0_2
// CHECK:         fsm.state @loop0_FRAME_0_0 output
// CHECK-NEXT:      fsm.output %false, %false, %true, %false, %loop0_first_iter, %false, %true, %false, %false, %false, %true, %false
// loop0_child_start_0 (result 8) rises in cycle 1, its scheduled offset.
// CHECK:         fsm.state @loop0_FRAME_0_1 output
// CHECK-NEXT:      fsm.output %false, %false, %true, %false, %loop0_first_iter, %false, %true, %false, %true, %true, %false, %true
loopschedule.func_sequential @static_lower_bound(%rowptr: memref<17xi64>, %n: i64) {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %outer = loopschedule.at 0 -> !loopschedule.handle {
      %launch = loopschedule.launch : !loopschedule.handle {
        %seq = loopschedule.sequential iter_args(%i = %c0_i64) : (i64) -> i64 {
          %f:4 = loopschedule.frame -> (i1, i64, i64, !loopschedule.handle) {
            %s0:3 = loopschedule.at 0 -> (i1, i64, i64) {
              %cond = arith.cmpi slt, %i, %n : i64
              %inext = arith.addi %i, %c1_i64 : i64
              %idx = arith.trunci %i : i64 to i5
              %lo = loopschedule.load %rowptr[%idx : i5] : memref<17xi64>
              loopschedule.iter_arg_update %i = %inext : i64
              loopschedule.yield %cond, %inext, %lo : i1, i64, i64
            }
            %s1 = loopschedule.at 1 -> i64 {
              %hi = arith.addi %s0#2, %c1_i64 : i64
              loopschedule.yield %hi : i64
            }
            %s2 = loopschedule.at 1 -> !loopschedule.handle {
              %inner = loopschedule.launch : !loopschedule.handle {
                %pip = loopschedule.pipeline II = 1 iter_args(%k = %c0_i64) : (i64) -> i64 {
                  %p0:2 = loopschedule.at 0 -> (i64, i1) {
                    %c = arith.cmpi slt, %k, %s1 : i64
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
            loopschedule.yield %s0#0, %s0#1, %s1, %s2 : i1, i64, i64, !loopschedule.handle
          }
          loopschedule.frame await {
            loopschedule.await %f#3
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
