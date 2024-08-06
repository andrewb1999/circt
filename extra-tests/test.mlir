module {
  func.func @kernel_if_select(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<16xi32>, %arg3: memref<16xi32>) {
    %0 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 0 : index}> : () -> index
    %2 = "arith.constant"() <{value = 16 : index}> : () -> index
    %3 = "arith.constant"() <{value = 1 : index}> : () -> index
    "loopschedule.pipeline"(%1) <{II = 1 : i64, tripCount = 16 : i64}> ({
    ^bb0(%arg4: index):
      %4 = "arith.cmpi"(%arg4, %2) <{predicate = 2 : i64}> : (index, index) -> i1
      "loopschedule.register"(%4) : (i1) -> ()
    }, {
    ^bb0(%arg4: index):
      %4:3 = "loopschedule.pipeline.stage"() <{end = 1 : i64, start = 0 : i64}> ({
        %6 = "loopschedule.load"(%arg2, %arg4) <{nontemporal = false}> {loopschedule.access_name = "load0"} : (memref<16xi32>, index) -> i32
        %7 = "arith.addi"(%arg4, %3) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
        "loopschedule.register"(%6, %arg4, %7) : (i32, index, index) -> ()
      }) : () -> (i32, index, index)
      %5:4 = "loopschedule.pipeline.stage"() <{end = 2 : i64, start = 1 : i64}> ({
        %6 = "arith.cmpi"(%4#0, %0) <{predicate = 0 : i64}> : (i32, i32) -> i1
        %7 = "loopschedule.load"(%arg0, %4#1) <{nontemporal = false}> {loopschedule.access_name = "load1"} : (memref<16xi32>, index) -> i32
        %8 = "loopschedule.load"(%arg1, %4#1) <{nontemporal = false}> {loopschedule.access_name = "load2"} : (memref<16xi32>, index) -> i32
        "loopschedule.register"(%6, %7, %8, %4#1) : (i1, i32, i32, index) -> ()
      }) : () -> (i1, i32, i32, index)
      "loopschedule.pipeline.stage"() <{end = 3 : i64, start = 2 : i64}> ({
        "loopschedule.store"(%5#1, %arg3, %5#3) <{nontemporal = false}> {loopschedule.access_name = "store0"} : (i32, memref<16xi32>, index) -> ()
        "loopschedule.store"(%5#2, %arg3, %5#3) <{nontemporal = false}> {loopschedule.access_name = "store1"} : (i32, memref<16xi32>, index) -> ()
        "loopschedule.store"(%5#2, %arg3, %5#3) <{nontemporal = false}> {loopschedule.access_name = "store2"} : (i32, memref<16xi32>, index) -> ()
        "loopschedule.register"() : () -> ()
      }) : () -> ()
      "loopschedule.terminator"(%4#2) <{operandSegmentSizes = array<i32: 1, 0>}> : (index) -> ()
    }) : (index) -> ()
    return
  }
}
