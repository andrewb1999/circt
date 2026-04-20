// RUN: circt-opt -canonicalize %s | FileCheck %s

// Sequential loop with one pass-through iter-arg: the terminator yields the
// iter-arg block argument at the same position as its init, meaning the
// value is loop-invariant. The canonicalizer should drop it.

// CHECK-LABEL: func.func @passthrough
// CHECK-SAME:  %[[BASE:.*]]: i32, %[[ACC0:.*]]: i32
func.func @passthrough(%init_base: i32, %init_acc: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : i32
  // After canonicalization, only the `acc` iter-arg should survive.
  // CHECK:      %[[RES:.*]] = loopschedule.sequential
  // CHECK-SAME:   iter_args(%[[ACC:.*]] = %[[ACC0]])
  // CHECK-SAME:   : (i32) -> i32
  %r:2 = loopschedule.sequential iter_args(%ba = %init_base, %acc = %init_acc)
                                 : (i32, i32) -> (i32, i32) {
    %cond, %next = loopschedule.frame -> (i1, i32) {
      %at:2 = loopschedule.at 0 -> (i1, i32) {
        %c = arith.cmpi slt, %acc, %c1 : i32
        %n = arith.addi %acc, %c1 : i32
        loopschedule.iter_arg_update %acc = %n : i32
        // Self-update for the pass-through iter-arg satisfies the current
        // verifier; the canonicalizer elides both the iter-arg and this
        // update.
        loopschedule.iter_arg_update %ba = %ba : i32
        loopschedule.yield %c, %n : i1, i32
      }
      loopschedule.yield %at#0, %at#1 : i1, i32
    }
    loopschedule.terminator condition(%cond), results(%ba, %next) : i32, i32
  }
  // External use of the base result should become the init directly.
  // CHECK: return %[[BASE]], %[[RES]] : i32, i32
  return %r#0, %r#1 : i32, i32
}
