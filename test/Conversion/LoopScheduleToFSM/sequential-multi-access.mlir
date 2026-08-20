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

// The loop now inlines into the function module: all port muxing lives in
// @fib's body, gated by the single machine's prefixed results.
// FSM results: #6 = loop0_frame_active_0, #7..#10 = loop0_frame_cycle_0_0..3.
// CHECK: hw.module @fib
// CHECK: %[[FSM:.+]]:11 = fsm.hw_instance "fib_fsm_inst" @fib_fsm

// Load at cycle 0 (A[i-1]): address gated in under frame_cycle_0_0. The
// external port is a latency-1 registered BRAM read, so the data is
// captured one cycle after issue (frame_cycle_0_1) and consumers get a
// live-cycle bypass mux.
// CHECK: %[[A0:.+]] = comb.mux %[[FSM]]#7,
// CHECK: %loop0_f0_ldcap_0 = seq.compreg.ce sym @loop0_f0_ldcap_0 %mem0_rd_data, %clk, %[[FSM]]#8
// CHECK: %[[V0:.+]] = comb.mux %[[FSM]]#8, %mem0_rd_data, %loop0_f0_ldcap_0

// Load at cycle 1 (A[i-2]): address chains over the cycle-0 address; data
// captured at cycle 2.
// CHECK: %[[A1:.+]] = comb.mux %[[FSM]]#8, %{{.+}}, %[[A0]]
// CHECK: %loop0_f0_ldcap_1 = seq.compreg.ce sym @loop0_f0_ldcap_1 %mem0_rd_data, %clk, %[[FSM]]#9
// CHECK: %[[V1:.+]] = comb.mux %[[FSM]]#9, %mem0_rd_data, %loop0_f0_ldcap_1

// The cycle-3 adder consumes the captured/bypassed values, never two
// aliases of the raw rd_data wire.
// CHECK: %[[SUM:.+]] = arith.addi %[[V0]], %[[V1]]

// Store at cycle 3: its address has lowest priority in the mux chain,
// and wr_en fires only in its own cycle. Enables merge as gate-qualified
// OR terms (concurrent-sibling-safe), addr/data keep the gate-keyed mux
// (this port is a passive-read memref).
// CHECK: %[[ADDR:.+]] = comb.mux %[[FSM]]#10, %{{.+}}, %[[A1]]
// CHECK: %[[GWREN:.+]] = comb.and %[[FSM]]#6, %[[FSM]]#10
// CHECK: comb.mux %[[FSM]]#6, %[[ADDR]],
// CHECK: comb.mux %[[FSM]]#6, %[[SUM]],
// CHECK: %[[WREN0:.+]] = comb.or %[[GWREN]], %false
// The loop-level enable then merges up through the function-level frame
// gate (frame_running_0, result #2) before reaching the port.
// CHECK: %[[WRENF:.+]] = comb.and %[[FSM]]#2, %[[WREN0]]
// CHECK: %[[WREN:.+]] = comb.or %[[WRENF]],
// CHECK: hw.output %{{.+}}, %{{.+}}, %{{.+}}, %[[WREN]]
