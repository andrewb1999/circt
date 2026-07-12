// RUN: amc-opt --pass-pipeline="builtin.module(operator-allocation{target-device=xcv80},lower-loopschedule-to-fsm)" %s | FileCheck %s

// A sequential frame with THREE static accesses to the same memory port:
// fib-style `A[i] = A[i-1] + A[i-2]`, with the two loads at cycles 0 and 1
// and the store at cycle 3. Regression test for the contended-port
// lowering: without SeqPortMuxCtx each access overwrote the port's single
// address slot (last writer wins), so only the store's address ever
// reached the memory and both load results aliased the raw rd_data wire
// (the loop ran but every store wrote garbage).
module {
  loopschedule.func_sequential @fib(%arg0: memref<32xi32>) attributes {top} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i6 = arith.constant 0 : i6
    %c1_i6 = arith.constant 1 : i6
    %c30_i6 = arith.constant 30 : i6
    %c1_i5 = arith.constant 1 : i5
    %c2_i5 = arith.constant 2 : i5
    %h = loopschedule.frame -> (!loopschedule.handle) {
      %lh = loopschedule.at 0 -> !loopschedule.handle {
        %launch = loopschedule.launch : !loopschedule.handle {
          loopschedule.sequential trip_count = 30 iter_args(%i = %c0_i6) : (i6) -> () {
            %f:2 = loopschedule.frame -> (i1, i6) {
              %s0:3 = loopschedule.at 0 -> (i1, i32, i5) {
                %cond = arith.cmpi slt, %i, %c30_i6 : i6
                %it = arith.trunci %i : i6 to i5
                %am1 = arith.addi %it, %c1_i5 : i5
                %v1 = loopschedule.load %arg0[%am1 : i5] : memref<32xi32>
                %ast = arith.addi %it, %c2_i5 : i5
                loopschedule.yield %cond, %v1, %ast : i1, i32, i5
              }
              %s1 = loopschedule.at 1 -> i32 {
                %v2 = loopschedule.load %arg0[%i : i6] : memref<32xi32>
                loopschedule.yield %v2 : i32
              }
              %s2 = loopschedule.at 2 -> i6 {
                %n = arith.addi %i, %c1_i6 : i6
                loopschedule.iter_arg_update %i = %n : i6
                loopschedule.yield %n : i6
              }
              loopschedule.at 3 {
                %sum = arith.addi %s0#1, %s1 : i32
                loopschedule.store %sum, %arg0[%s0#2 : i5] : memref<32xi32>
              }
              loopschedule.yield %s0#0, %s2 : i1, i6
            }
            loopschedule.terminator condition(%f#0), results()
          }
          loopschedule.yield
        }
        loopschedule.yield %launch : !loopschedule.handle
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
}

// CHECK: hw.module @loop0
// FSM outputs: #3 = frame_active_0, #4..#7 = frame_cycle_0_0..3.
// CHECK: %[[FSM:.+]]:8 = fsm.hw_instance "loop0_fsm_inst"

// Load at cycle 0 (A[i-1]): address gated in under frame_cycle_0_0. The
// external port is a latency-1 registered BRAM read, so the data is
// captured one cycle after issue (frame_cycle_0_1) and consumers get a
// live-cycle bypass mux.
// CHECK: %[[A0:.+]] = comb.mux %[[FSM]]#4,
// CHECK: %loop0_f0_ldcap_0 = seq.compreg.ce sym @loop0_f0_ldcap_0 %mem0_rd_data, %clk, %[[FSM]]#5
// CHECK: %[[V0:.+]] = comb.mux %[[FSM]]#5, %mem0_rd_data, %loop0_f0_ldcap_0

// Load at cycle 1 (A[i-2]): address chains over the cycle-0 address; data
// captured at cycle 2.
// CHECK: %[[A1:.+]] = comb.mux %[[FSM]]#5, %{{.+}}, %[[A0]]
// CHECK: %loop0_f0_ldcap_1 = seq.compreg.ce sym @loop0_f0_ldcap_1 %mem0_rd_data, %clk, %[[FSM]]#6
// CHECK: %[[V1:.+]] = comb.mux %[[FSM]]#6, %mem0_rd_data, %loop0_f0_ldcap_1

// The cycle-3 adder consumes the captured/bypassed values, never two
// aliases of the raw rd_data wire.
// CHECK: %[[SUM:.+]] = arith.addi %[[V0]], %[[V1]]

// Store at cycle 3: its address has lowest priority in the mux chain and
// wr_en fires only in its own cycle.
// CHECK: %[[ADDR:.+]] = comb.mux %[[FSM]]#7, %{{.+}}, %[[A1]]
// CHECK: comb.mux %[[FSM]]#3, %[[ADDR]],
// CHECK: comb.mux %[[FSM]]#3, %[[SUM]],
// CHECK: %[[WREN:.+]] = comb.mux %[[FSM]]#3, %[[FSM]]#7,
// (done is the FSM's done OR'd with the early-done advance-edge term)
// CHECK: hw.output %{{.+}}, %{{.+}}, %{{.+}}, %[[WREN]]
