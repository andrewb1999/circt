//===- LoopScheduleTestbenchGeneration.cpp - Testbench generation -*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates an hw.module @top testbench that wraps a DUT hw.module produced by
// LoopScheduleToFSM. Two modes:
//
// 1. Attribute mode (no data-dir): Uses testbench.inputs and
//    testbench.expected_outputs attributes for scalar I/O with PASS/FAIL.
//
// 2. Data-dir mode (--data-dir=<path>): Loads all inputs (scalar and memory)
//    from hex files via $readmemh. After done, dumps all outputs (scalar and
//    memory) to stdout for external comparison.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "circt/Dialect/SV/SVOps.h"

namespace circt {
namespace loopschedule {
#define GEN_PASS_DECL_LOOPSCHEDULETESTBENCHGENERATION
#define GEN_PASS_DEF_LOOPSCHEDULETESTBENCHGENERATION
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"
} // namespace loopschedule
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

namespace {

/// A group of memory ports sharing a common prefix.
struct MemPortGroup {
  std::string name;       // e.g. "mem0"
  hw::PortInfo rdData;    // input: read data
  hw::PortInfo addr;      // output: address
  hw::PortInfo wrData;    // output: write data
  hw::PortInfo wrEn;      // output: write enable
  unsigned addrWidth;     // address bits
  unsigned depth;         // 2^addrWidth
  Type dataType;          // element type
};

struct LoopScheduleTestbenchGenerationPass
    : public circt::loopschedule::impl::LoopScheduleTestbenchGenerationBase<
          LoopScheduleTestbenchGenerationPass> {
  using LoopScheduleTestbenchGenerationBase::
      LoopScheduleTestbenchGenerationBase;
  void runOnOperation() override;

private:
  void generateAttributeMode(hw::HWModuleOp dutMod, ModuleOp moduleOp);
  void generateDataDirMode(hw::HWModuleOp dutMod, ModuleOp moduleOp);
};

} // namespace

//===----------------------------------------------------------------------===//
// Port classification helpers
//===----------------------------------------------------------------------===//

static bool isControlPort(StringRef name) {
  return name == "clk" || name == "rst" || name == "start" || name == "done";
}

/// Try to detect memory port groups from DUT ports.
/// Memory ports follow the naming convention: {name}_rd_data, {name}_addr,
/// {name}_wr_data, {name}_wr_en.
static SmallVector<MemPortGroup>
classifyMemoryPorts(const SmallVector<hw::PortInfo> &dutPorts) {
  // Collect candidate prefixes from _rd_data ports.
  SmallVector<std::string> prefixes;
  for (auto &port : dutPorts) {
    StringRef pname = port.getName();
    if (pname.ends_with("_rd_data"))
      prefixes.push_back(pname.drop_back(8).str()); // remove "_rd_data"
  }

  SmallVector<MemPortGroup> groups;
  for (auto &prefix : prefixes) {
    MemPortGroup group;
    group.name = prefix;

    for (auto &port : dutPorts) {
      StringRef pname = port.getName();
      if (pname == prefix + "_rd_data")
        group.rdData = port;
      else if (pname == prefix + "_addr")
        group.addr = port;
      else if (pname == prefix + "_wr_data")
        group.wrData = port;
      else if (pname == prefix + "_wr_en")
        group.wrEn = port;
    }

    // Verify all four ports exist.
    if (group.rdData.getName().empty() || group.addr.getName().empty() ||
        group.wrData.getName().empty() || group.wrEn.getName().empty())
      continue;

    group.dataType = group.rdData.type;
    group.addrWidth = cast<IntegerType>(group.addr.type).getWidth();
    group.depth = 1u << group.addrWidth;
    groups.push_back(group);
  }
  return groups;
}

/// Check if a port belongs to any memory group.
static bool isMemoryPort(StringRef name,
                         const SmallVector<MemPortGroup> &memGroups) {
  for (auto &g : memGroups) {
    if (name == g.name + "_rd_data" || name == g.name + "_addr" ||
        name == g.name + "_wr_data" || name == g.name + "_wr_en")
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Attribute mode (legacy — scalar I/O from MLIR attributes)
//===----------------------------------------------------------------------===//

void LoopScheduleTestbenchGenerationPass::generateAttributeMode(
    hw::HWModuleOp dutMod, ModuleOp moduleOp) {
  auto *ctx = moduleOp.getContext();
  OpBuilder builder(ctx);

  auto inputsAttr = dutMod->getAttrOfType<ArrayAttr>("testbench.inputs");
  auto expectedAttr =
      dutMod->getAttrOfType<ArrayAttr>("testbench.expected_outputs");
  if (!inputsAttr || !expectedAttr) {
    dutMod.emitError("DUT must have both testbench.inputs and "
                     "testbench.expected_outputs attributes");
    return signalPassFailure();
  }

  auto dutPorts = dutMod.getPortList();

  SmallVector<hw::PortInfo> scalarInputs;
  SmallVector<hw::PortInfo> scalarOutputs;

  for (auto &port : dutPorts) {
    if (port.isInput()) {
      if (!isControlPort(port.getName()))
        scalarInputs.push_back(port);
    } else if (port.isOutput()) {
      if (port.getName() != "done")
        scalarOutputs.push_back(port);
    }
  }

  if (scalarInputs.size() != inputsAttr.size()) {
    dutMod.emitError("testbench.inputs size mismatch: expected ")
        << scalarInputs.size() << " but got " << inputsAttr.size();
    return signalPassFailure();
  }
  if (scalarOutputs.size() != expectedAttr.size()) {
    dutMod.emitError("testbench.expected_outputs size mismatch: expected ")
        << scalarOutputs.size() << " but got " << expectedAttr.size();
    return signalPassFailure();
  }

  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto loc = dutMod.getLoc();

  SmallVector<hw::PortInfo> topPorts;
  topPorts.push_back({{builder.getStringAttr("clk"), builder.getI1Type(),
                        hw::ModulePort::Direction::Input}});
  topPorts.push_back({{builder.getStringAttr("rst"), builder.getI1Type(),
                        hw::ModulePort::Direction::Input}});

  auto topMod = hw::HWModuleOp::create(builder, loc,
                                        builder.getStringAttr("top"),
                                        hw::ModulePortInfo(topPorts));

  Block *body = topMod.getBodyBlock();
  builder.setInsertionPoint(body->getTerminator());

  Value clk = body->getArgument(0);
  Value rst = body->getArgument(1);

  Value falseVal =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
  Value trueVal =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
  auto i8Type = builder.getIntegerType(8);
  Value c0_i8 = hw::ConstantOp::create(builder, loc, i8Type, 0);
  Value c1_i8 = hw::ConstantOp::create(builder, loc, i8Type, 1);
  Value cFF_i8 = hw::ConstantOp::create(builder, loc, i8Type, 255);
  Value fd = hw::ConstantOp::create(builder, loc, builder.getIntegerType(32),
                                    0x80000002);

  SmallVector<Value> inputValues;
  for (auto attr : inputsAttr) {
    auto intAttr = cast<IntegerAttr>(attr);
    inputValues.push_back(hw::ConstantOp::create(builder, loc, intAttr));
  }

  SmallVector<Value> expectedValues;
  for (auto attr : expectedAttr) {
    auto intAttr = cast<IntegerAttr>(attr);
    expectedValues.push_back(hw::ConstantOp::create(builder, loc, intAttr));
  }

  auto startReg = sv::RegOp::create(builder, loc, builder.getI1Type(),
                                    builder.getStringAttr("tb_start"));
  Value startVal = sv::ReadInOutOp::create(builder, loc, startReg);
  auto ctrReg = sv::RegOp::create(builder, loc, i8Type,
                                  builder.getStringAttr("tb_counter"));
  Value ctrVal = sv::ReadInOutOp::create(builder, loc, ctrReg);
  auto doneSeenReg = sv::RegOp::create(builder, loc, builder.getI1Type(),
                                       builder.getStringAttr("tb_done_seen"));
  Value doneSeenVal = sv::ReadInOutOp::create(builder, loc, doneSeenReg);

  SmallVector<Value> dutInputs;
  unsigned scalarIdx = 0;
  for (auto &port : dutPorts) {
    if (!port.isInput())
      continue;
    if (port.getName() == "clk")
      dutInputs.push_back(clk);
    else if (port.getName() == "rst")
      dutInputs.push_back(rst);
    else if (port.getName() == "start")
      dutInputs.push_back(startVal);
    else
      dutInputs.push_back(inputValues[scalarIdx++]);
  }

  auto dutInst = hw::InstanceOp::create(builder, loc, dutMod,
                                        builder.getStringAttr("dut"),
                                        dutInputs);

  Value doneVal;
  SmallVector<Value> outputValues;
  unsigned resultIdx = 0;
  for (auto &port : dutPorts) {
    if (!port.isOutput())
      continue;
    if (port.getName() == "done")
      doneVal = dutInst.getResult(resultIdx);
    else
      outputValues.push_back(dutInst.getResult(resultIdx));
    resultIdx++;
  }

  sv::AlwaysFFOp::create(
      builder, loc, sv::EventControl::AtPosEdge, clk, [&] {
        sv::IfOp::create(
            builder, loc, rst,
            [&] {
              sv::PAssignOp::create(builder, loc, startReg, falseVal);
              sv::PAssignOp::create(builder, loc, ctrReg, c0_i8);
              sv::PAssignOp::create(builder, loc, doneSeenReg, falseVal);
            },
            [&] {
              Value nextCtr =
                  comb::AddOp::create(builder, loc, ctrVal, c1_i8);
              sv::PAssignOp::create(builder, loc, ctrReg, nextCtr);

              Value isCycle0 = comb::ICmpOp::create(
                  builder, loc, comb::ICmpPredicate::eq, ctrVal, c0_i8);
              sv::PAssignOp::create(builder, loc, startReg, isCycle0);

              sv::IfOp::create(builder, loc, doneVal, [&] {
                Value notDoneSeen =
                    comb::XorOp::create(builder, loc, doneSeenVal, trueVal);
                sv::IfOp::create(builder, loc, notDoneSeen, [&] {
                  sv::PAssignOp::create(builder, loc, doneSeenReg, trueVal);

                  if (scalarOutputs.empty()) {
                    sv::FWriteOp::create(builder, loc, fd, "PASS\n",
                                         ValueRange{});
                  } else {
                    Value allMatch = trueVal;
                    for (auto [outVal, expVal] :
                         llvm::zip(outputValues, expectedValues)) {
                      Value match = comb::ICmpOp::create(
                          builder, loc, comb::ICmpPredicate::eq, outVal,
                          expVal);
                      allMatch =
                          comb::AndOp::create(builder, loc, allMatch, match);
                    }
                    sv::IfOp::create(
                        builder, loc, allMatch,
                        [&] {
                          sv::FWriteOp::create(builder, loc, fd, "PASS\n",
                                               ValueRange{});
                        },
                        [&] {
                          sv::FWriteOp::create(builder, loc, fd, "FAIL\n",
                                               ValueRange{});
                        });
                  }
                  sv::FinishOp::create(builder, loc, 0);
                });
              });

              Value isTimeout = comb::ICmpOp::create(
                  builder, loc, comb::ICmpPredicate::eq, ctrVal, cFF_i8);
              sv::IfOp::create(builder, loc, isTimeout, [&] {
                sv::FWriteOp::create(builder, loc, fd, "TIMEOUT\n",
                                     ValueRange{});
                sv::FinishOp::create(builder, loc, 1);
              });
            });
      });

  dutMod->removeAttr("testbench.inputs");
  dutMod->removeAttr("testbench.expected_outputs");
}

//===----------------------------------------------------------------------===//
// Data-dir mode (hex files for all I/O, stdout dump)
//===----------------------------------------------------------------------===//

void LoopScheduleTestbenchGenerationPass::generateDataDirMode(
    hw::HWModuleOp dutMod, ModuleOp moduleOp) {
  auto *ctx = moduleOp.getContext();
  OpBuilder builder(ctx);
  auto loc = dutMod.getLoc();

  auto dutPorts = dutMod.getPortList();
  auto memGroups = classifyMemoryPorts(dutPorts);

  // Classify scalar ports (excluding control and memory ports).
  SmallVector<hw::PortInfo> scalarInputs;
  SmallVector<hw::PortInfo> scalarOutputs;
  for (auto &port : dutPorts) {
    if (isControlPort(port.getName()) || isMemoryPort(port.getName(), memGroups))
      continue;
    if (port.isInput())
      scalarInputs.push_back(port);
    else if (port.isOutput())
      scalarOutputs.push_back(port);
  }

  // Build hw.module @top(in %clk: i1, in %rst: i1).
  builder.setInsertionPointToEnd(moduleOp.getBody());

  SmallVector<hw::PortInfo> topPorts;
  topPorts.push_back({{builder.getStringAttr("clk"), builder.getI1Type(),
                        hw::ModulePort::Direction::Input}});
  topPorts.push_back({{builder.getStringAttr("rst"), builder.getI1Type(),
                        hw::ModulePort::Direction::Input}});

  auto topMod = hw::HWModuleOp::create(builder, loc,
                                        builder.getStringAttr("top"),
                                        hw::ModulePortInfo(topPorts));

  Block *body = topMod.getBodyBlock();
  builder.setInsertionPoint(body->getTerminator());

  Value clk = body->getArgument(0);
  Value rst = body->getArgument(1);

  // Constants.
  Value falseVal =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
  Value trueVal =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
  auto i8Type = builder.getIntegerType(8);
  Value c0_i8 = hw::ConstantOp::create(builder, loc, i8Type, 0);
  Value c1_i8 = hw::ConstantOp::create(builder, loc, i8Type, 1);
  Value cFF_i8 = hw::ConstantOp::create(builder, loc, i8Type, 255);
  Value fd = hw::ConstantOp::create(builder, loc, builder.getIntegerType(32),
                                    0x80000002);

  // --- Create memory arrays and initialize from hex files ---
  struct MemInfo {
    sv::RegOp reg;
    Value readVal; // combinational read: mem[addr]
  };
  DenseMap<StringRef, MemInfo> memInfoMap;

  for (auto &group : memGroups) {
    auto arrayType =
        hw::UnpackedArrayType::get(group.dataType, group.depth);
    auto memReg = sv::RegOp::create(builder, loc, arrayType,
                                    builder.getStringAttr(group.name));

    // Initialize via sv.initial { sv.readmem }.
    sv::InitialOp::create(builder, loc, [&] {
      sv::ReadMemOp::create(builder, loc, memReg,
                            dataDir + "/" + group.name + ".hex",
                            MemBaseTypeAttr::MemBaseHex);
    });

    memInfoMap[group.name] = {memReg, Value()};
  }

  // --- Create scalar input arrays and initialize from hex files ---
  SmallVector<Value> scalarInputValues;
  for (auto &port : scalarInputs) {
    auto arrayType = hw::UnpackedArrayType::get(port.type, 1);
    auto scalarReg = sv::RegOp::create(
        builder, loc, arrayType,
        builder.getStringAttr(port.getName().str() + "_data"));

    sv::InitialOp::create(builder, loc, [&] {
      sv::ReadMemOp::create(builder, loc, scalarReg,
                            dataDir + "/" + port.getName().str() + ".hex",
                            MemBaseTypeAttr::MemBaseHex);
    });

    // Read element [0] for the scalar value.
    auto addrType = builder.getIntegerType(1);
    Value idx = hw::ConstantOp::create(builder, loc, addrType, 0);
    Value elemRef =
        sv::ArrayIndexInOutOp::create(builder, loc, scalarReg, idx);
    Value scalarVal = sv::ReadInOutOp::create(builder, loc, elemRef);
    scalarInputValues.push_back(scalarVal);
  }

  // --- Control registers ---
  auto startReg = sv::RegOp::create(builder, loc, builder.getI1Type(),
                                    builder.getStringAttr("tb_start"));
  Value startVal = sv::ReadInOutOp::create(builder, loc, startReg);
  auto ctrReg = sv::RegOp::create(builder, loc, i8Type,
                                  builder.getStringAttr("tb_counter"));
  Value ctrVal = sv::ReadInOutOp::create(builder, loc, ctrReg);
  auto doneSeenReg = sv::RegOp::create(builder, loc, builder.getI1Type(),
                                       builder.getStringAttr("tb_done_seen"));
  Value doneSeenVal = sv::ReadInOutOp::create(builder, loc, doneSeenReg);

  // --- Build DUT instance operands ---
  // We need to wire: clk, rst, start, scalar inputs, memory rd_data.
  // Memory rd_data needs the DUT's addr output, which creates a dependency.
  // We read mem[addr] combinationally after the instance is created.
  // For the instance, we use a read from index 0 as placeholder and fix later.
  // Actually: we need to provide rd_data BEFORE the instance. Use
  // sv.read_inout of the array indexed by the addr (which comes from instance
  // output). This is a combinational cycle in the testbench, which is fine for
  // simulation — the memory model is combinational read, clocked write.
  //
  // Solution: create the read wiring AFTER the instance using the addr output.
  // For hw.instance inputs, we need the rd_data value upfront. Use a wire.

  // For each memory group, we need a combinational read that depends on
  // the DUT's address output. We'll build these after the instance.
  // For now, collect what we need for the DUT inputs.

  SmallVector<Value> dutInputs;
  unsigned scalarIdx = 0;

  // We'll collect memory read values to connect after instance creation.
  // Use DenseMap to track which DUT input index corresponds to which mem group.
  DenseMap<StringRef, unsigned> memRdDataInputIdx;

  for (auto &port : dutPorts) {
    if (!port.isInput())
      continue;
    if (port.getName() == "clk") {
      dutInputs.push_back(clk);
    } else if (port.getName() == "rst") {
      dutInputs.push_back(rst);
    } else if (port.getName() == "start") {
      dutInputs.push_back(startVal);
    } else if (port.getName().ends_with("_rd_data")) {
      // Memory read data — placeholder; will be wired after instance.
      // We need a zero constant of the right type as placeholder.
      // Actually, we can't modify instance inputs after creation.
      // Instead, use the combinational memory read path.
      // The read depends on the addr output of the instance, but we
      // haven't created the instance yet.
      //
      // The correct approach: create the memory read wire first,
      // and use a backedge-like pattern. But hw.instance doesn't
      // support backedges.
      //
      // Standard solution: use an sv.wire for the read data, feed it
      // to the instance, then assign it combinationally from mem[addr].
      StringRef prefix = port.getName().drop_back(8); // remove "_rd_data"
      auto rdWire = sv::WireOp::create(builder, loc, port.type,
                                       builder.getStringAttr(
                                           prefix.str() + "_rd_wire"));
      Value rdVal = sv::ReadInOutOp::create(builder, loc, rdWire);
      dutInputs.push_back(rdVal);
      memRdDataInputIdx[prefix] = dutInputs.size() - 1;

      // Store the wire so we can assign to it later.
      memInfoMap[prefix].readVal = rdWire; // reuse readVal to store the wire
    } else {
      // Scalar input.
      dutInputs.push_back(scalarInputValues[scalarIdx++]);
    }
  }

  // --- Instantiate DUT ---
  auto dutInst = hw::InstanceOp::create(builder, loc, dutMod,
                                        builder.getStringAttr("dut"),
                                        dutInputs);

  // --- Extract DUT output values ---
  Value doneVal;
  SmallVector<Value> scalarOutputValues;
  DenseMap<StringRef, Value> memAddrValues;
  DenseMap<StringRef, Value> memWrDataValues;
  DenseMap<StringRef, Value> memWrEnValues;

  unsigned outIdx = 0;
  for (auto &port : dutPorts) {
    if (!port.isOutput())
      continue;
    Value result = dutInst.getResult(outIdx++);
    if (port.getName() == "done") {
      doneVal = result;
    } else if (port.getName().ends_with("_addr")) {
      StringRef prefix = port.getName().drop_back(5); // remove "_addr"
      memAddrValues[prefix] = result;
    } else if (port.getName().ends_with("_wr_data")) {
      StringRef prefix = port.getName().drop_back(8); // remove "_wr_data"
      memWrDataValues[prefix] = result;
    } else if (port.getName().ends_with("_wr_en")) {
      StringRef prefix = port.getName().drop_back(6); // remove "_wr_en"
      memWrEnValues[prefix] = result;
    } else {
      scalarOutputValues.push_back(result);
    }
  }

  // --- Wire memory combinational reads: mem[addr] → rd_data wire ---
  for (auto &group : memGroups) {
    auto &info = memInfoMap[group.name];
    Value addr = memAddrValues[group.name];
    Value elemRef =
        sv::ArrayIndexInOutOp::create(builder, loc, info.reg, addr);
    Value rdData = sv::ReadInOutOp::create(builder, loc, elemRef);
    // Assign the wire.
    sv::AssignOp::create(builder, loc, info.readVal, rdData);
  }

  // --- always_ff: control, memory writes, output dump ---
  sv::AlwaysFFOp::create(
      builder, loc, sv::EventControl::AtPosEdge, clk, [&] {
        sv::IfOp::create(
            builder, loc, rst,
            // Reset.
            [&] {
              sv::PAssignOp::create(builder, loc, startReg, falseVal);
              sv::PAssignOp::create(builder, loc, ctrReg, c0_i8);
              sv::PAssignOp::create(builder, loc, doneSeenReg, falseVal);
            },
            // Normal operation.
            [&] {
              Value nextCtr =
                  comb::AddOp::create(builder, loc, ctrVal, c1_i8);
              sv::PAssignOp::create(builder, loc, ctrReg, nextCtr);

              Value isCycle0 = comb::ICmpOp::create(
                  builder, loc, comb::ICmpPredicate::eq, ctrVal, c0_i8);
              sv::PAssignOp::create(builder, loc, startReg, isCycle0);

              // Memory writes: when wr_en high, write to mem[addr].
              for (auto &group : memGroups) {
                auto &info = memInfoMap[group.name];
                Value wrEn = memWrEnValues[group.name];
                Value addr = memAddrValues[group.name];
                Value wrData = memWrDataValues[group.name];

                sv::IfOp::create(builder, loc, wrEn, [&] {
                  Value elemRef = sv::ArrayIndexInOutOp::create(builder, loc,
                                                                info.reg, addr);
                  sv::PAssignOp::create(builder, loc, elemRef, wrData);
                });
              }

              // Done handling: dump outputs and finish.
              sv::IfOp::create(builder, loc, doneVal, [&] {
                Value notDoneSeen =
                    comb::XorOp::create(builder, loc, doneSeenVal, trueVal);
                sv::IfOp::create(builder, loc, notDoneSeen, [&] {
                  sv::PAssignOp::create(builder, loc, doneSeenReg, trueVal);

                  // Dump scalar outputs to stdout.
                  for (auto [idx, outVal] :
                       llvm::enumerate(scalarOutputValues)) {
                    StringRef outName = scalarOutputs[idx].getName();
                    unsigned width =
                        cast<IntegerType>(outVal.getType()).getWidth();
                    unsigned hexDigits = (width + 3) / 4;
                    std::string fmtStr = "@OUT " + outName.str() + " %0" +
                                          std::to_string(hexDigits) + "h\n";
                    sv::FWriteOp::create(builder, loc, fd, fmtStr,
                                         ValueRange{outVal});
                  }

                  // Dump memory contents to stdout.
                  for (auto &group : memGroups) {
                    auto &info = memInfoMap[group.name];
                    unsigned width =
                        cast<IntegerType>(group.dataType).getWidth();
                    unsigned hexDigits = (width + 3) / 4;

                    // Print header: @MEM name depth
                    std::string headerStr =
                        "@MEM " + group.name + " " +
                         std::to_string(group.depth) + "\n";
                    sv::FWriteOp::create(builder, loc, fd, headerStr,
                                         ValueRange{});

                    // Iterate with sv.for and print each element.
                    unsigned idxWidth = group.addrWidth + 1; // +1 to avoid
                                                             // overflow
                    auto idxType = builder.getIntegerType(idxWidth);
                    Value lb =
                        hw::ConstantOp::create(builder, loc, idxType, 0);
                    Value ub = hw::ConstantOp::create(builder, loc, idxType,
                                                      group.depth);
                    Value step =
                        hw::ConstantOp::create(builder, loc, idxType, 1);

                    sv::ForOp::create(
                        builder, loc, lb, ub, step, "i", [&](BlockArgument iv) {
                          // Truncate iv to addr width for indexing.
                          Value truncIV = comb::ExtractOp::create(
                              builder, loc,
                              builder.getIntegerType(group.addrWidth), iv, 0);
                          Value elemRef = sv::ArrayIndexInOutOp::create(
                              builder, loc, info.reg, truncIV);
                          Value elem =
                              sv::ReadInOutOp::create(builder, loc, elemRef);
                          std::string elemFmtStr = std::string("%0") +
                                                    std::to_string(hexDigits) +
                                                    "h\n";
                          sv::FWriteOp::create(builder, loc, fd, elemFmtStr,
                                               ValueRange{elem});
                        });
                  }

                  // Print end marker.
                  sv::FWriteOp::create(builder, loc, fd, "DONE\n",
                                       ValueRange{});
                  sv::FinishOp::create(builder, loc, 0);
                });
              });

              // Timeout.
              Value isTimeout = comb::ICmpOp::create(
                  builder, loc, comb::ICmpPredicate::eq, ctrVal, cFF_i8);
              sv::IfOp::create(builder, loc, isTimeout, [&] {
                sv::FWriteOp::create(builder, loc, fd, "TIMEOUT\n",
                                     ValueRange{});
                sv::FinishOp::create(builder, loc, 1);
              });
            });
      });

  // Clean up testbench attributes if present.
  dutMod->removeAttr("testbench.inputs");
  dutMod->removeAttr("testbench.expected_outputs");
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

void LoopScheduleTestbenchGenerationPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Find the DUT: an hw.module marked with 'top' or testbench attributes.
  hw::HWModuleOp dutMod;
  moduleOp.walk([&](hw::HWModuleOp mod) {
    if (mod->hasAttr("testbench.inputs") || mod->hasAttr("top"))
      dutMod = mod;
  });

  if (!dutMod)
    return;

  if (dataDir.empty()) {
    // Legacy attribute mode.
    if (!dutMod->hasAttr("testbench.inputs"))
      return; // Nothing to do without attributes and no data-dir.
    generateAttributeMode(dutMod, moduleOp);
  } else {
    // Data-dir mode: hex files for all I/O.
    generateDataDirMode(dutMod, moduleOp);
  }
}

std::unique_ptr<mlir::Pass>
circt::loopschedule::createLoopScheduleTestbenchGenerationPass() {
  return std::make_unique<LoopScheduleTestbenchGenerationPass>();
}
