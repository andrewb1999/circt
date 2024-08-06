module {
  func.func @kernel_if_select(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>, %arg3: memref<16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %true = hw.constant true
    loopschedule.pipeline II = 1 trip_count = 16 iter_args(%arg4 = %c0) : (index) -> () {
      %0 = arith.cmpi slt, %arg4, %c16 : index
      loopschedule.register %0 : i1
    } do {
      %0:3 = loopschedule.pipeline.stage start = 0 end = 1 {
        %2 = loopschedule.load %arg2[%arg4 : index] {loopschedule.access_name = "load0"} : memref<16xi32>
        %3 = arith.addi %arg4, %c1 : index
        loopschedule.register %2, %arg4, %3 : i32, index, index
      } : i32, index, index
      %1:4 = loopschedule.pipeline.stage start = 1 end = 2 {
        %2 = arith.cmpi eq, %0#0, %c0_i32 : i32
        %3 = loopschedule.if %2 -> (i32) {
          %6 = loopschedule.load %arg0[%0#1 : index] {loopschedule.access_name = "load1"} : memref<16xi32>
          loopschedule.yield %6 : i32
        }
        %4 = arith.xori %2, %true : i1
        %5 = loopschedule.if %4 -> (i32) {
          %6 = loopschedule.load %arg1[%0#1 : index] {loopschedule.access_name = "load2"} : memref<16xi32>
          loopschedule.yield %6 : i32
        }
        loopschedule.register %2, %3, %5, %0#1 : i1, i32, i32, index
      } : i1, i32, i32, index
      loopschedule.pipeline.stage start = 2 end = 3 {
        %2 = loopschedule.if %1#0 -> (i32) {
          loopschedule.store %1#1, %arg3[%1#3 : index] {loopschedule.access_name = "store0"} : memref<16xi32>
          loopschedule.yield
        }
        %3 = arith.xori %1#0, %true : i1
        %4 = loopschedule.if %3 -> (i32) {
          loopschedule.store %1#2, %arg3[%1#3 : index] {loopschedule.access_name = "store1"} : memref<16xi32>
          loopschedule.yield
        }
        %5 = arith.select %1#0, %1#1, %1#2 : i32
        loopschedule.store %5, %arg3[%1#3 : index] {loopschedule.access_name = "store2"} : memref<16xi32>
      }
      loopschedule.terminator iter_args(%0#2), results() : (index) -> ()
    }
    return
  }
}
