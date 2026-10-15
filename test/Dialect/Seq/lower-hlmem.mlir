// RUN: circt-opt --lower-seq-hlmem %s | FileCheck %s

// CHECK-LABEL:   hw.module @d1(
// CHECK-SAME:                  in %[[CLOCK:.*]] : !seq.clock,
// CHECK-SAME:                  in %[[VAL_1:.*]] : i1) {
// CHECK:           %[[VAL_2:.*]] = sv.reg  : !hw.inout<uarray<4xi32>>
// CHECK:           %[[CLK:.+]] = seq.from_clock %[[CLOCK]]
// CHECK:           sv.alwaysff(posedge %[[CLK]]) {
// CHECK:             sv.if %[[VAL_3:.*]] {
// CHECK:               %[[VAL_4:.*]] = sv.array_index_inout %[[VAL_2]]{{\[}}%[[VAL_5:.*]]] : !hw.inout<uarray<4xi32>>, i2
// CHECK:               sv.passign %[[VAL_4]], %[[VAL_6:.*]] : i32
// CHECK:             }
// CHECK:           }(syncreset : posedge %[[VAL_1]]) {
// CHECK:           }
// CHECK:           %[[VAL_7:.*]] = hw.constant 0 : i2
// CHECK:           %[[VAL_8:.*]] = hw.constant true
// CHECK:           %[[VAL_9:.*]] = hw.constant 42 : i32
// CHECK:           %[[VAL_10:.*]] = sv.array_index_inout %[[VAL_2]]{{\[}}%[[VAL_7]]] : !hw.inout<uarray<4xi32>>, i2
// CHECK:           %[[VAL_11:.*]] = sv.read_inout %[[VAL_10]] : !hw.inout<i32>
// CHECK:           %[[VAL_12:.*]] = seq.compreg sym @myMemory_rdaddr0_dly0 %[[VAL_7]], %[[CLOCK]] : i2
// CHECK:           %[[VAL_13:.*]] = sv.array_index_inout %[[VAL_2]]{{\[}}%[[VAL_12]]] : !hw.inout<uarray<4xi32>>, i2
// CHECK:           %[[VAL_14:.*]] = sv.read_inout %[[VAL_13]] : !hw.inout<i32>
// CHECK:           %[[VAL_15:.*]] = seq.compreg sym @myMemory_rd0_reg %[[VAL_14]], %[[CLOCK]] : i32
// CHECK:           hw.output
// CHECK:         }
hw.module @d1(in %clk : !seq.clock, in %rst : i1) {
  %myMemory = seq.hlmem @myMemory %clk, %rst : <4xi32>
    seq.write %myMemory[%c0_i2] %c42_i32 wren %c1_i1 { latency = 1 } : !seq.hlmem<4xi32>

  %c0_i2 = hw.constant 0 : i2
  %c1_i1 = hw.constant 1 : i1
  %c42_i32 = hw.constant 42 : i32

  %myMemory_rdata = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 0} : !seq.hlmem<4xi32>
  %myMemory_rdata2 = seq.read %myMemory[%c0_i2] rden %c1_i1 { latency = 2} : !seq.hlmem<4xi32>
  hw.output
}

// Each write port gets its own always_ff block: synthesis infers a
// multi-port RAM only from one write port per process.
// CHECK-LABEL:   hw.module @two_writes(
// CHECK:           %[[MEM:.+]] = sv.reg : !hw.inout<uarray<8xi16>>
// CHECK:           sv.alwaysff(posedge %{{.+}}) {
// CHECK-NEXT:        sv.if %{{.+}} {
// CHECK-NEXT:          %[[LOC0:.+]] = sv.array_index_inout %[[MEM]][%{{.+}}]
// CHECK-NEXT:          sv.passign %[[LOC0]], %{{.+}} : i16
// CHECK-NEXT:        }
// CHECK-NEXT:      }(syncreset : posedge %{{.+}}) {
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.alwaysff(posedge %{{.+}}) {
// CHECK-NEXT:        sv.if %{{.+}} {
// CHECK-NEXT:          %[[LOC1:.+]] = sv.array_index_inout %[[MEM]][%{{.+}}]
// CHECK-NEXT:          sv.passign %[[LOC1]], %{{.+}} : i16
// CHECK-NEXT:        }
// CHECK-NEXT:      }(syncreset : posedge %{{.+}}) {
// CHECK-NEXT:      }
// No two ports may write one address in the same cycle (checked out of
// reset): which write would win is unspecified.
// CHECK-DAG:       %[[SAME:.+]] = comb.icmp eq %a1, %a0 : i3
// CHECK-DAG:       %[[BOTH:.+]] = comb.and %en1, %en0 : i1
// CHECK-DAG:       %[[COLLISION:.+]] = comb.and %[[BOTH]], %[[SAME]] : i1
// CHECK-DAG:       %[[NOT_RST:.+]] = comb.xor %rst, %{{.+}} : i1
// CHECK-DAG:       %[[OK:.+]] = comb.xor %[[COLLISION]], %{{.+}} : i1
// CHECK:           verif.clocked_assert %[[OK]] if %[[NOT_RST]], posedge %{{.+}} label "mem_write_collision" : i1
// CHECK-NEXT:      hw.output
hw.module @two_writes(in %clk : !seq.clock, in %rst : i1, in %a0 : i3, in %d0 : i16, in %en0 : i1, in %a1 : i3, in %d1 : i16, in %en1 : i1) {
  %mem = seq.hlmem @mem %clk, %rst : <8xi16>
  seq.write %mem[%a0] %d0 wren %en0 { latency = 1 } : !seq.hlmem<8xi16>
  seq.write %mem[%a1] %d1 wren %en1 { latency = 1 } : !seq.hlmem<8xi16>
  hw.output
}

// Three write ports: one check over every pair.
// CHECK-LABEL:   hw.module @three_writes(
// CHECK-COUNT-3:   comb.icmp eq
// CHECK:           verif.clocked_assert {{.+}} label "mem3_write_collision"
// CHECK-NOT:       verif.clocked_assert
// CHECK:           hw.output
hw.module @three_writes(in %clk : !seq.clock, in %rst : i1, in %a0 : i3, in %a1 : i3, in %a2 : i3, in %d : i16, in %en : i1) {
  %mem3 = seq.hlmem @mem3 %clk, %rst : <8xi16>
  seq.write %mem3[%a0] %d wren %en { latency = 1 } : !seq.hlmem<8xi16>
  seq.write %mem3[%a1] %d wren %en { latency = 1 } : !seq.hlmem<8xi16>
  seq.write %mem3[%a2] %d wren %en { latency = 1 } : !seq.hlmem<8xi16>
  hw.output
}
