"func.func"() <{function_type = (memref<20xi32>) -> (), sym_name = "kernel_fib"}> ({
^bb0(%arg0: memref<20xi32>):
  %0 = "arith.constant"() <{value = 2 : i6}> : () -> i6
  %1 = "arith.constant"() <{value = 1 : i6}> : () -> i6
  %2 = "arith.constant"() <{value = 1 : i5}> : () -> i5
  %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %4 = "arith.constant"() <{value = 0 : i5}> : () -> i5
  %5 = "arith.constant"() <{value = 18 : i5}> : () -> i5
  "loopschedule.store"(%3, %arg0, %2) <{nontemporal = false}> : (i32, memref<20xi32>, i5) -> ()
  "scf.for"(%4, %5, %2) ({
  ^bb0(%arg1: i5):
    %6 = "arith.extsi"(%arg1) : (i5) -> i6
    %7 = "arith.addi"(%6, %1) <{overflowFlags = #arith.overflow<none>}> : (i6, i6) -> i6
    %8 = "arith.trunci"(%7) : (i6) -> i5
    %9 = "loopschedule.load"(%arg0, %8) <{nontemporal = false}> : (memref<20xi32>, i5) -> i32
    %10 = "loopschedule.load"(%arg0, %arg1) <{nontemporal = false}> : (memref<20xi32>, i5) -> i32
    %11 = "arith.addi"(%9, %10) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
    %12 = "arith.extsi"(%arg1) : (i5) -> i6
    %13 = "arith.addi"(%12, %0) <{overflowFlags = #arith.overflow<none>}> : (i6, i6) -> i6
    "loopschedule.store"(%11, %arg0, %13) <{nontemporal = false}> : (i32, memref<20xi32>, i6) -> ()
    "scf.yield"() : () -> ()
  }) {loopschedule.trip_count = 18 : i64} : (i5, i5, i5) -> ()
  "func.return"() : () -> ()
}) {itypes = "u", loopschedule.dependencies = @kernel_fib_deps, otypes = "", top} : () -> ()
