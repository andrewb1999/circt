module {
  loopschedule.dependencies @kernel_matmul_deps {
    loopschedule.depends_on "load2" {
      loopschedule.access "store0"
      loopschedule.access "store1" dist<1>
    }
    loopschedule.depends_on "store1" {
      loopschedule.access "store0"
      loopschedule.access "load2" dist<1>
      loopschedule.access "load2"
    }
  }
  func.func @kernel_matmul(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) -> memref<16x16xi32> attributes {itypes = "ss", loopschedule.dependencies = @kernel_matmul_deps, otypes = "s", top} {
    %c0_i5 = arith.constant 0 : i5
    %c-16_i5 = arith.constant -16 : i5
    %c1_i5 = arith.constant 1 : i5
    %c0_i5_0 = arith.constant 0 : i5
    %c-16_i5_1 = arith.constant -16 : i5
    %c1_i5_2 = arith.constant 1 : i5
    %c0_i5_3 = arith.constant 0 : i5
    %c-16_i5_4 = arith.constant -16 : i5
    %c1_i5_5 = arith.constant 1 : i5
    %c0_i5_6 = arith.constant 0 : i5
    %c-16_i5_7 = arith.constant -16 : i5
    %c1_i5_8 = arith.constant 1 : i5
    %c0_i5_9 = arith.constant 0 : i5
    %c-16_i5_10 = arith.constant -16 : i5
    %c1_i5_11 = arith.constant 1 : i5
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<16x16xi32>
    loopschedule.step {
      loopschedule.sequential trip_count = 16 iter_args(%arg2 = %c0_i5_3) : (i5) -> () {
        %0 = arith.cmpi ult, %arg2, %c-16_i5_4 : i5
        loopschedule.register %0 : i1
      } do {
        loopschedule.step {
          loopschedule.sequential trip_count = 16 iter_args(%arg3 = %c0_i5_6) : (i5) -> () {
            %1 = arith.cmpi ult, %arg3, %c-16_i5_7 : i5
            loopschedule.register %1 : i1
          } do {
            %1 = loopschedule.step {
              loopschedule.store %c0_i32, %alloc[%arg2, %arg3 : i5, i5] : memref<16x16xi32>
              %2 = arith.addi %arg3, %c1_i5_8 : i5
              loopschedule.register %2 : i5
            } : index
            loopschedule.terminator iter_args(%1), results() : (index) -> ()
          }
        }
        %0 = loopschedule.step {
          %1 = arith.addi %arg2, %c1_i5_5 : i5
          loopschedule.register %1 : i5
        } : index
        loopschedule.terminator iter_args(%0), results() : (index) -> ()
      }
    }
    loopschedule.step {
      loopschedule.sequential trip_count = 16 iter_args(%arg2 = %c0_i5) : (i5) -> () {
        %0 = arith.cmpi ult, %arg2, %c-16_i5 : i5
        loopschedule.register %0 : i1
      } do {
        loopschedule.step {
          loopschedule.sequential trip_count = 16 iter_args(%arg3 = %c0_i5_0) : (i5) -> () {
            %1 = arith.cmpi ult, %arg3, %c-16_i5_1 : i5
            loopschedule.register %1 : i1
          } do {
            loopschedule.step {
              loopschedule.pipeline II = 2 trip_count = 16 iter_args(%arg4 = %c0_i5_9) : (i5) -> () {
                %2 = arith.cmpi ult, %arg4, %c-16_i5_10 : i5
                loopschedule.register %2 : i1
              } do {
                %2:5 = loopschedule.pipeline.stage start = 0 end = 1 {
                  %7 = arith.trunci %arg2 : i5 to i4
                  %8 = arith.trunci %arg4 : i5 to i4
                  %9 = loopschedule.load %arg0[%7, %8 : i4, i4] : memref<16x16xi32>
                  %10 = arith.trunci %arg4 : i5 to i4
                  %11 = arith.trunci %arg3 : i5 to i4
                  %12 = loopschedule.load %arg1[%10, %11 : i4, i4] : memref<16x16xi32>
                  %13 = arith.trunci %arg2 : i5 to i4
                  %14 = arith.trunci %arg3 : i5 to i4
                  %15 = arith.addi %arg4, %c1_i5_11 : i5
                  loopschedule.register %9, %12, %13, %14, %15 : i32, i32, i4, i4, i5
                } : i32, i32, i4, i4, i5
                %3:3 = loopschedule.pipeline.stage start = 1 end = 2 {
                  %7 = arith.muli %2#0, %2#1 : i32
                  loopschedule.register %2#2, %2#3, %7 : i4, i4, i32
                } : i4, i4, i32
                %4:2 = loopschedule.pipeline.stage start = 2 end = 3 {
                  loopschedule.register %3#0, %3#1 : i4, i4
                } : i4, i4
                %5:2 = loopschedule.pipeline.stage start = 3 end = 4 {
                  loopschedule.register %4#0, %4#1 : i4, i4
                } : i4, i4
                %6 = loopschedule.pipeline.stage start = 4 end = 5 {
                  %7 = loopschedule.load %alloc[%5#0, %5#1 : i4, i4] : memref<16x16xi32>
                  loopschedule.register %7 : i32
                } : i32
                loopschedule.pipeline.stage start = 5 end = 6 {
                  %7 = arith.addi %6, %3#2 : i32
                  loopschedule.store %7, %alloc[%arg2, %arg3 : i5, i5] : memref<16x16xi32>
                }
                loopschedule.terminator iter_args(%2#4), results() : (i5) -> ()
              }
            }
            %1 = loopschedule.step {
              %2 = arith.addi %arg3, %c1_i5_2 : i5
              loopschedule.register %2 : i5
            } : index
            loopschedule.terminator iter_args(%1), results() : (index) -> ()
          }
        }
        %0 = loopschedule.step {
          %1 = arith.addi %arg2, %c1_i5 : i5
          loopschedule.register %1 : i5
        } : index
        loopschedule.terminator iter_args(%0), results() : (index) -> ()
      }
    }
    return %alloc : memref<16x16xi32>
  }
}

