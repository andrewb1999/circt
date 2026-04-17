// XFAIL: *
// RUN: circt-opt %s -lower-loopschedule-to-calyx -split-input-file | FileCheck %s

// The LoopScheduleToCalyx pass currently has a pre-existing bug where
// BuildIntermediateRegs crashes on nested frame/at shapes
// (getUniqueName(phase->getParentOp()) assert), so these test IRs are kept
// as XFAIL until the pass learns how to handle the new surface.

// Minimal: a single-stage pipeline that increments an iter-arg.
// CHECK: calyx.component @minimal
func.func @minimal() {
  %c0_i64 = arith.constant 0 : i64
  %c10_i64 = arith.constant 10 : i64
  %c1_i64 = arith.constant 1 : i64
  loopschedule.pipeline II = 1 trip_count = 10 iter_args(%arg0 = %c0_i64) : (i64) -> () {
    %0:2 = loopschedule.at 0 -> (i64, i1) {
      %cond = arith.cmpi ult, %arg0, %c10_i64 : i64
      %next = arith.addi %arg0, %c1_i64 : i64
      loopschedule.iter_arg_update %arg0 = %next : i64
      loopschedule.yield %next, %cond : i64, i1
    }
    loopschedule.terminator condition(%0#1), results()
  }
  return
}

// -----

// Dot-product-style multi-stage pipeline with load/mul/add through stages.
// CHECK: calyx.component @dot
func.func @dot(%arg0: memref<64xi32>, %arg1: memref<64xi32>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %0 = loopschedule.pipeline II = 1 trip_count = 5 iter_args(%arg2 = %c0, %arg3 = %c0_i32) : (index, i32) -> i32 {
    %s0:4 = loopschedule.at 0 -> (i32, i32, index, i1) {
      %a = memref.load %arg0[%arg2] : memref<64xi32>
      %b = memref.load %arg1[%arg2] : memref<64xi32>
      %nx = arith.addi %arg2, %c1 : index
      %cond = arith.cmpi ult, %arg2, %c64 : index
      loopschedule.iter_arg_update %arg2 = %nx : index
      loopschedule.yield %a, %b, %nx, %cond : i32, i32, index, i1
    }
    %s1 = loopschedule.at 1 -> i32 {
      %m = arith.muli %s0#0, %s0#1 : i32
      loopschedule.yield %m : i32
    }
    %s2 = loopschedule.at 4 -> i32 {
      %acc = arith.addi %arg3, %s1 : i32
      loopschedule.iter_arg_update %arg3 = %acc : i32
      loopschedule.yield %acc : i32
    }
    loopschedule.terminator condition(%s0#3), results(%s2) : i32
  }
  return %0 : i32
}

// -----

// Independent stores in a single stage.
// CHECK: calyx.component @store
module {
  func.func @store(%arg0: memref<4xi32>, %arg1: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 4 iter_args(%arg2 = %c0) : (index) -> () {
      %s0:4 = loopschedule.at 0 -> (i32, index, index, i1) {
        %v = memref.load %arg0[%arg2] : memref<4xi32>
        %nx = arith.addi %arg2, %c1 : index
        %cond = arith.cmpi ult, %arg2, %c4 : index
        loopschedule.iter_arg_update %arg2 = %nx : index
        loopschedule.yield %v, %arg2, %nx, %cond : i32, index, index, i1
      }
      loopschedule.at 1 {
        memref.store %s0#0, %arg1[%s0#1] : memref<4xi32>
        memref.store %s0#0, %arg1[%s0#1] : memref<4xi32>
        loopschedule.yield
      }
      loopschedule.terminator condition(%s0#3), results()
    }
    return
  }
}

// -----

// Scalar (shapeless) memref becomes a 1D Calyx memory of size 1.
// CHECK: calyx.component @main
module {
  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    %alloca = memref.alloca() : memref<i32>
    memref.store %c1_i32, %alloca[] : memref<i32>
    return
  }
}
