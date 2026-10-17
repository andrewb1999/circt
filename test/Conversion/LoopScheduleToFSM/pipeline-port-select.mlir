// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A port's address and write data only matter in a cycle one of its
// accesses enables it, and no two do in one cycle, so they are selected
// among the accesses with nothing defaulting to zero: the last access drives
// them outside the others' selects, a lone one as it is. Only the enables
// are gated. Here an II-2 pipeline reads A[i] in stage 0 and writes it back
// in stage 1 through one port: the address selects the load's in stage 0's
// cycles and is the store's otherwise, and the store, the port's only
// writer, drives its write data as is, up through the loop's frame to the
// module's port. That no two accesses enable a port in one cycle is checked
// in simulation.

// CHECK-LABEL: hw.module @increment
// CHECK-SAME:    out mem0_addr : i4, out mem0_rd_en : i1, out mem0_wr_data : i32, out mem0_wr_en : i1
// CHECK:         %[[DATA:.+]] = arith.addi %mem0_rd_data, %c1_i32 : i32
// CHECK:         %[[BOTH:.+]] = comb.and %{{.+}}, %{{.+}} : i1
// CHECK:         %[[ONE:.+]] = comb.xor %[[BOTH]], %true
// CHECK:         verif.clocked_assert %[[ONE]] if %{{.+}}, posedge %{{.+}} label "mem_port_one_accessor"
// CHECK:         %[[ADDR:.+]] = comb.mux %{{.+}}, %{{.+}}, %{{.+}} : i4
// CHECK:         hw.output %[[ADDR]], %{{.+}}, %[[DATA]], %{{.+}},

loopschedule.func_sequential @increment(%A: memref<16xi32>) attributes {top} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %h = loopschedule.frame -> (!loopschedule.handle) {
    %lh = loopschedule.at 0 -> !loopschedule.handle {
      %lh_launch = loopschedule.launch : !loopschedule.handle {
        loopschedule.pipeline II = 2 latency = 2 iter_args(%i = %c0_i32) : (i32) -> () {
          %s0:3 = loopschedule.at 0 -> (i32, i32, i1) {
            %v = loopschedule.load %A[%i : i32] : memref<16xi32>
            %ni = arith.addi %i, %c1_i32 : i32
            %cond = arith.cmpi slt, %ni, %c16_i32 : i32
            loopschedule.iter_arg_update %i = %ni : i32
            loopschedule.yield %v, %i, %cond : i32, i32, i1
          }
          loopschedule.at 1 {
            %w = arith.addi %s0#0, %c1_i32 : i32
            loopschedule.store %w, %A[%s0#1 : i32] : memref<16xi32>
            loopschedule.yield
          }
          loopschedule.terminator condition(%s0#2), results()
        }
        loopschedule.yield
      }
      loopschedule.yield %lh_launch : !loopschedule.handle
    }
    loopschedule.yield %lh : !loopschedule.handle
  }
  loopschedule.frame {
    loopschedule.await %h
    loopschedule.yield
  } do {
    loopschedule.yield
  }
  loopschedule.return
}
