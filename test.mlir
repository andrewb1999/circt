module {
  func.func @kernel(%arg0: memref<20xi32>, %arg1: memref<20xi32>, %arg2: memref<20xi32>, %arg3: memref<20xi32>, %arg4: memref<20xi32>) attributes {itypes = "uuuuu", otypes = ""} {
    %c0_5 = arith.constant 0 : i5
    %c20_5 = arith.constant 20 : i5
    %c1_5 = arith.constant 1 : i5
    loopschedule.step {
      loopschedule.sequential trip_count = 20 iter_args(%arg5 = %c0_5) : (i5) -> () {
        %0 = arith.cmpi ult, %arg5, %c20_5 : i5
        loopschedule.register %0 : i1
      } do {
        %0:2 = loopschedule.step {
          %6 = loopschedule.load %arg2[%arg5 : i5] : memref<20xi32>
          %7 = loopschedule.load %arg1[%arg5 : i5] : memref<20xi32>
          loopschedule.register %6, %7 : i32, i32
        } : i32, i32
        %1:4 = loopschedule.step {
          %6 = loopschedule.load %arg0[%arg5 : i5] : memref<20xi32>
          %7 = arith.trunci %0#0 : i32 to i5
          %8 = loopschedule.load %arg3[%7 : i5] {from = "denseVec"} : memref<20xi32>
          %9 = arith.trunci %0#1 : i32 to i5
          loopschedule.register %6, %8, %9, %0#1 : i32, i32, i5, i32
        } : i32, i32, i5, i32
        %2 = loopschedule.step {
          %6 = arith.extui %1#0 {unsigned} : i32 to i64
          %7 = arith.extui %1#1 {unsigned} : i32 to i64
          %8 = arith.muli %6, %7 : i64
          loopschedule.register %8 : i64
        } : i64
        %3 = loopschedule.step {
          %6 = arith.trunci %1#3 : i32 to i5
          %7 = loopschedule.load %arg4[%6 : i5] {from = "outputVec"} : memref<20xi32>
          loopschedule.register %7 : i32
        } : i32
        %4 = loopschedule.step {
          %6 = arith.trunci %2 {unsigned} : i64 to i32
          %7 = arith.addi %3, %6 : i32
          loopschedule.register %7 : i32
        } : i32
        %5 = loopschedule.step {
          loopschedule.store %4, %arg4[%1#2 : i5] {to = "outputVec"} : memref<20xi32>
          %6 = arith.addi %arg5, %c1_5 : i5
          loopschedule.register %6 : i5
        } : i5
        loopschedule.terminator iter_args(%5), results() : (i5) -> ()
      }
    }
    return
  }
}

