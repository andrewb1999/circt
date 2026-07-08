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
// Options struct is brought in by LoopSchedulePasses.h via GEN_PASS_DECL.
#define GEN_PASS_DEF_LOOPSCHEDULETESTBENCHGENERATION
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"
} // namespace loopschedule
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

namespace {

/// A group of memory ports sharing a common prefix. Only 1-D I/O memories
/// are supported here — multi-dim memrefs would need a richer testbench
/// model and are intentionally rejected.
struct MemPortGroup {
  std::string name;       // e.g. "mem0"
  hw::PortInfo rdData;    // input: read data
  hw::PortInfo addr;      // output: address (mem<i>_addr)
  hw::PortInfo wrData;    // output: write data
  hw::PortInfo wrEn;      // output: write enable
  // Optional read enable (mem<i>_rd_en). When present, the behavioral
  // memory only updates its read register on enabled cycles (BRAM hold
  // semantics); absent (older DUTs) means always-enabled.
  hw::PortInfo rdEn;
  unsigned addrWidth;     // address bits
  unsigned depth;         // 2^addrWidth
  Type dataType;          // element type
};

struct LoopScheduleTestbenchGenerationPass
    : public circt::loopschedule::impl::LoopScheduleTestbenchGenerationBase<
          LoopScheduleTestbenchGenerationPass> {
  using LoopScheduleTestbenchGenerationBase::
      LoopScheduleTestbenchGenerationBase;
  // Expose the TableGen-generated typedef so out-of-class callers can
  // construct the pass with options.
  using Options = ::circt::loopschedule::LoopScheduleTestbenchGenerationOptions;
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
  return name == "clk" || name == "rst" || name == "start" || name == "done" ||
         name == "ready";
}

/// Try to detect memory port groups from DUT ports.
/// Memory ports follow the naming convention: {name}_rd_data, {name}_addr,
/// {name}_wr_data, {name}_wr_en. Only 1-D I/O memories are supported —
/// a group containing `_addr_<d>` (the multi-dim naming) is rejected.
static SmallVector<MemPortGroup, 2>
classifyMemoryPorts(const SmallVector<hw::PortInfo> &dutPorts) {
  // Collect candidate prefixes from _rd_data ports.
  SmallVector<std::string> prefixes;
  for (auto &port : dutPorts) {
    StringRef pname = port.getName();
    if (pname.ends_with("_rd_data"))
      prefixes.push_back(pname.drop_back(8).str()); // remove "_rd_data"
  }

  SmallVector<MemPortGroup, 2> groups;
  for (auto &prefix : prefixes) {
    MemPortGroup group;
    group.name = prefix;
    bool multiDim = false;

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
      else if (pname == prefix + "_rd_en")
        group.rdEn = port;
      else if (pname.starts_with(prefix + "_addr_"))
        multiDim = true;
    }

    if (multiDim)
      continue;

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
                         const SmallVectorImpl<MemPortGroup> &memGroups) {
  for (auto &g : memGroups) {
    if (name == g.name + "_rd_data" || name == g.name + "_addr" ||
        name == g.name + "_wr_data" || name == g.name + "_wr_en" ||
        name == g.name + "_rd_en")
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// AXI bundle classification
//===----------------------------------------------------------------------===//

namespace {
/// One m_axi bundle on the DUT boundary: the `<name>_m_axi_<sig>` port group
/// plus the slave-sizing metadata the FSM lowering published via the
/// `amc.axi_bundles` module attribute (the AXI face itself carries no depth).
struct AxiBundleInfo {
  std::string name;
  uint64_t depth = 0;
  unsigned dataW = 32;
  unsigned addrShift = 2;
  /// DUT ports keyed by bare signal name ("araddr", "rdata", ...).
  llvm::StringMap<hw::PortInfo> ports;
};
} // namespace

/// The m_axi signal set, in the port-declaration order of
/// hdl/systemverilog/axi_slave_mem.sv. `dutOutput` marks signals the kernel
/// master DRIVES (= slave inputs); the rest are slave outputs feeding the
/// kernel.
struct AxiSigDesc {
  const char *sig;
  bool dutOutput;
};
static const AxiSigDesc kAxiSigs[] = {
    {"araddr", true},   {"arvalid", true},  {"arlen", true},
    {"arsize", true},   {"arburst", true},  {"arid", true},
    {"arprot", true},   {"arcache", true},  {"arlock", true},
    {"arqos", true},    {"arregion", true}, {"arready", false},
    {"rdata", false},   {"rvalid", false},  {"rresp", false},
    {"rlast", false},   {"rid", false},     {"rready", true},
    {"awaddr", true},   {"awvalid", true},  {"awlen", true},
    {"awsize", true},   {"awburst", true},  {"awid", true},
    {"awprot", true},   {"awcache", true},  {"awlock", true},
    {"awqos", true},    {"awregion", true}, {"awready", false},
    {"wdata", true},    {"wstrb", true},    {"wvalid", true},
    {"wlast", true},    {"wready", false},  {"bvalid", false},
    {"bresp", false},   {"bid", false},     {"bready", true},
};

static bool isAxiPort(StringRef name) { return name.contains("_m_axi_"); }

/// Group the DUT's `<bundle>_m_axi_<sig>` ports into per-bundle infos and
/// resolve each bundle's depth from the `amc.axi_bundles` attribute.
static LogicalResult
classifyAxiBundles(hw::HWModuleOp dutMod,
                   const SmallVector<hw::PortInfo> &dutPorts,
                   SmallVector<AxiBundleInfo> &bundles) {
  llvm::MapVector<StringRef, AxiBundleInfo> byName;
  for (auto &port : dutPorts) {
    StringRef pname = port.getName();
    size_t pos = pname.find("_m_axi_");
    if (pos == StringRef::npos)
      continue;
    StringRef bundle = pname.take_front(pos);
    StringRef sig = pname.drop_front(pos + strlen("_m_axi_"));
    auto &info = byName[bundle];
    info.name = bundle.str();
    info.ports[sig] = port;
  }
  if (byName.empty())
    return success();

  auto bundlesAttr = dutMod->getAttrOfType<ArrayAttr>("amc.axi_bundles");
  for (auto &kv : byName) {
    auto &info = kv.second;
    for (auto &desc : kAxiSigs)
      if (!info.ports.count(desc.sig))
        return dutMod.emitError("m_axi bundle '")
               << kv.first << "' is missing signal '" << desc.sig << "'";
    info.dataW = cast<IntegerType>(info.ports["rdata"].type).getWidth();
    unsigned strbW = cast<IntegerType>(info.ports["wstrb"].type).getWidth();
    info.addrShift = llvm::Log2_32(std::max(1u, strbW));

    bool found = false;
    if (bundlesAttr) {
      for (auto attr : bundlesAttr) {
        auto dict = dyn_cast<DictionaryAttr>(attr);
        if (!dict)
          continue;
        auto nameAttr = dict.getAs<StringAttr>("name");
        auto depthAttr = dict.getAs<IntegerAttr>("depth");
        if (nameAttr && depthAttr && nameAttr.getValue() == kv.first) {
          info.depth = (uint64_t)depthAttr.getInt();
          found = true;
          break;
        }
      }
    }
    if (!found)
      return dutMod.emitError(
                 "m_axi bundle '")
             << kv.first
             << "' has no amc.axi_bundles metadata (slave depth unknown)";
    bundles.push_back(info);
  }
  return success();
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
  // Use a 24-bit timeout counter (16M cycles) to comfortably handle deeply
  // nested sequential loops, where multi-step bodies make per-iteration
  // cycle counts grow super-linearly with nesting depth.
  auto i8Type = builder.getIntegerType(24);
  Value c0_i8 = hw::ConstantOp::create(builder, loc, i8Type, 0);
  Value c1_i8 = hw::ConstantOp::create(builder, loc, i8Type, 1);
  Value cFF_i8 = hw::ConstantOp::create(builder, loc, i8Type, 0xFFFFFF);
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

  // AXI bundles get a behavioral slave instance each (axi_slave_mem.sv).
  SmallVector<AxiBundleInfo> axiBundles;
  if (failed(classifyAxiBundles(dutMod, dutPorts, axiBundles)))
    return signalPassFailure();

  // Classify scalar ports (excluding control, memory, and AXI ports).
  SmallVector<hw::PortInfo> scalarInputs;
  SmallVector<hw::PortInfo> scalarOutputs;
  for (auto &port : dutPorts) {
    if (isControlPort(port.getName()) ||
        isMemoryPort(port.getName(), memGroups) || isAxiPort(port.getName()))
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
  // Use a 24-bit timeout counter (16M cycles) to comfortably handle deeply
  // nested sequential loops, where multi-step bodies make per-iteration
  // cycle counts grow super-linearly with nesting depth.
  auto i8Type = builder.getIntegerType(24);
  Value c0_i8 = hw::ConstantOp::create(builder, loc, i8Type, 0);
  Value c1_i8 = hw::ConstantOp::create(builder, loc, i8Type, 1);
  Value cFF_i8 = hw::ConstantOp::create(builder, loc, i8Type, 0xFFFFFF);
  Value fd = hw::ConstantOp::create(builder, loc, builder.getIntegerType(32),
                                    0x80000002);

  // --- Create memory arrays and initialize from hex files ---
  // Memories are sized to fit MAX_N_TXNS transactions worth of data
  // (`MAX_N_TXNS * group.depth`). For each transaction T's view of
  // memory (size group.depth), the testbench shifts the DUT's address
  // by `T * group.depth`. This lets a single verilator run process N
  // transactions with per-transaction-distinct inputs/outputs without
  // needing per-transaction tags from the DUT — write attribution is
  // recovered via per-address write counters (each transaction's write
  // to address A is the (T+1)th write to A overall, since pipeline
  // ordering preserves per-(mem,addr) issue order).
  //
  // MAX_N_TXNS is a fixed compile-time bound on batch size. Tests need
  // num_transactions <= MAX_N_TXNS. Sim-only memory; not synthesized.
  const unsigned MAX_N_TXNS = 1024;
  auto i32TypeMem = builder.getIntegerType(32);

  struct MemInfo {
    sv::RegOp reg;          // partitioned: MAX_N_TXNS * group.depth
    Value readVal;          // combinational read wire (set later)
    sv::RegOp writeCntReg;  // i32[group.depth]: per-addr write counter
    unsigned perTxnDepth;   // = group.depth
  };
  DenseMap<StringRef, MemInfo> memInfoMap;

  for (auto &group : memGroups) {
    unsigned partitionedDepth = MAX_N_TXNS * group.depth;
    auto arrayType =
        hw::UnpackedArrayType::get(group.dataType, partitionedDepth);
    auto memReg = sv::RegOp::create(builder, loc, arrayType,
                                    builder.getStringAttr(group.name));

    // Per-address write counter array. Tracks how many transactions have
    // written each address; the next write to address A goes into slot
    // `count[A] * group.depth + A`. Initialized to zero at sim start.
    auto cntArrayType =
        hw::UnpackedArrayType::get(i32TypeMem, group.depth);
    auto cntReg = sv::RegOp::create(
        builder, loc, cntArrayType,
        builder.getStringAttr(group.name + "_addr_wcnt"));

    // Init the memory contents from hex AND zero the per-addr counter in
    // the same initial block. We use blocking assignment in a for-loop
    // here (instead of NBA inside always_ff) because verilator rejects
    // delayed array writes inside loops (BLKLOOPINIT).
    sv::InitialOp::create(builder, loc, [&] {
      sv::ReadMemOp::create(builder, loc, memReg,
                            dataDir + "/" + group.name + ".hex",
                            MemBaseTypeAttr::MemBaseHex);
      Value zero32 = hw::ConstantOp::create(builder, loc, i32TypeMem, 0);
      unsigned w = std::max(1u, group.addrWidth);
      auto idxTy = builder.getIntegerType(w + 1);
      Value lb = hw::ConstantOp::create(builder, loc, idxTy, 0);
      Value ub =
          hw::ConstantOp::create(builder, loc, idxTy, group.depth);
      Value step = hw::ConstantOp::create(builder, loc, idxTy, 1);
      sv::ForOp::create(
          builder, loc, lb, ub, step, "ai", [&](BlockArgument iv) {
            Value tIv = comb::ExtractOp::create(
                builder, loc,
                builder.getIntegerType(std::max(1u, group.addrWidth)), iv, 0);
            Value cntRef = sv::ArrayIndexInOutOp::create(
                builder, loc, cntReg, tIv);
            sv::BPAssignOp::create(builder, loc, cntRef, zero32);
          });
    });

    memInfoMap[group.name] =
        {memReg, Value(), cntReg, group.depth};
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
  // `tb_start` is driven combinationally (a wire, not a reg) so it stays
  // in sync with `_dut_ready` on the same cycle. A registered start would
  // be one cycle late: the DUT's `ready` could have dropped between the
  // cycle we sampled it and the cycle the registered start is presented,
  // and the start would be silently rejected.
  auto startWire = sv::WireOp::create(builder, loc, builder.getI1Type(),
                                      builder.getStringAttr("tb_start"));
  Value startVal = sv::ReadInOutOp::create(builder, loc, startWire);
  auto ctrReg = sv::RegOp::create(builder, loc, i8Type,
                                  builder.getStringAttr("tb_counter"));
  Value ctrVal = sv::ReadInOutOp::create(builder, loc, ctrReg);
  auto doneSeenReg = sv::RegOp::create(builder, loc, builder.getI1Type(),
                                       builder.getStringAttr("tb_done_seen"));
  Value doneSeenVal = sv::ReadInOutOp::create(builder, loc, doneSeenReg);

  // Multi-transaction handshake. The number of transactions to drive is
  // loaded at simulation init from `num_transactions.hex` (one i32 entry).
  // Falls back to 1 when the file is absent (verilator emits a warning but
  // leaves the array zero-initialized — we treat 0 as "1" so the legacy
  // single-shot path still works for tests written before Phase 2).
  auto i32Type = builder.getIntegerType(32);
  auto numTxnsArrayType = hw::UnpackedArrayType::get(i32Type, 1);
  auto numTxnsReg = sv::RegOp::create(builder, loc, numTxnsArrayType,
                                      builder.getStringAttr("tb_num_txns_arr"));
  sv::InitialOp::create(builder, loc, [&] {
    sv::ReadMemOp::create(builder, loc, numTxnsReg,
                          dataDir + "/num_transactions.hex",
                          MemBaseTypeAttr::MemBaseHex);
  });
  auto numTxnsAddrTy = builder.getIntegerType(1);
  Value numTxnsZeroIdx =
      hw::ConstantOp::create(builder, loc, numTxnsAddrTy, 0);
  Value numTxnsElemRef = sv::ArrayIndexInOutOp::create(builder, loc, numTxnsReg,
                                                        numTxnsZeroIdx);
  Value numTxnsRaw = sv::ReadInOutOp::create(builder, loc, numTxnsElemRef);
  // Treat 0 as 1 for backwards compat with tests that don't write the file.
  Value c1_i32 = hw::ConstantOp::create(builder, loc, i32Type, 1);
  Value isZeroNumTxns = comb::ICmpOp::create(
      builder, loc, comb::ICmpPredicate::eq, numTxnsRaw,
      hw::ConstantOp::create(builder, loc, i32Type, 0));
  Value numTxns =
      comb::MuxOp::create(builder, loc, isZeroNumTxns, c1_i32, numTxnsRaw);

  // Per-transaction counters. Issue counter ticks on each accepted start;
  // done counter ticks on each done pulse. Memory dump fires once after
  // tb_done_count == numTxns.
  auto issueCntReg = sv::RegOp::create(
      builder, loc, i32Type, builder.getStringAttr("tb_issue_count"));
  Value issueCntVal = sv::ReadInOutOp::create(builder, loc, issueCntReg);
  auto doneCntReg = sv::RegOp::create(builder, loc, i32Type,
                                      builder.getStringAttr("tb_done_count"));
  Value doneCntVal = sv::ReadInOutOp::create(builder, loc, doneCntReg);

  // Per-memory read-shift register chain. Each memory's read shift is
  // taken from `chain[max_read_stage_for_that_mem]`, where:
  //   chain[0] : increments on every accepted `start`. At cycle T it
  //              equals the txn currently entering stage 0.
  //   chain[s] : registered copy of chain[s-1] (1-cycle delay). At cycle
  //              T it equals the txn currently at stage s.
  // This works for any II as long as the kernel is structurally valid
  // (max read stage < II + min read stage), which the scheduler enforces.
  //
  // The DUT carries a `loopschedule.mem_read_stages` attribute (an int
  // array, one entry per mem<i> in port order) that names the max read
  // stage for each memory. Older DUTs (e.g. sequential funcs lowered
  // before this attr existed) fall back to chain[0] for everything.
  ArrayAttr memReadStagesAttr =
      dutMod->getAttrOfType<ArrayAttr>("loopschedule.mem_read_stages");
  unsigned maxReadStage = 0;
  if (memReadStagesAttr) {
    for (auto attr : memReadStagesAttr) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        continue;
      maxReadStage = std::max(maxReadStage, (unsigned)intAttr.getInt());
    }
  }

  SmallVector<sv::RegOp> shiftChainRegs(maxReadStage + 1);
  SmallVector<Value> shiftChainVals(maxReadStage + 1);
  for (unsigned s = 0; s <= maxReadStage; ++s) {
    shiftChainRegs[s] = sv::RegOp::create(
        builder, loc, i32Type,
        builder.getStringAttr("tb_read_shift_s" + std::to_string(s)));
    shiftChainVals[s] =
        sv::ReadInOutOp::create(builder, loc, shiftChainRegs[s]);
  }

  // Map mem<i> name → chain index (= the memory's max read stage).
  DenseMap<StringRef, unsigned> memShiftStage;
  if (memReadStagesAttr) {
    for (auto [i, attr] : llvm::enumerate(memReadStagesAttr)) {
      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (!intAttr)
        continue;
      std::string name = "mem" + std::to_string(i);
      memShiftStage[StringAttr::get(ctx, name).strref()] =
          (unsigned)intAttr.getInt();
    }
  }
  auto shiftForMem = [&](StringRef memName) -> Value {
    auto it = memShiftStage.find(memName);
    unsigned s = (it != memShiftStage.end()) ? it->second : 0;
    if (s >= shiftChainVals.size())
      s = shiftChainVals.size() - 1;
    return shiftChainVals[s];
  };

  // --- Build DUT instance operands ---
  // We need to wire: clk, rst, start, scalar inputs, memory rd_data.
  // Memory rd_data needs the DUT's addr output, which creates a dependency:
  // the rd_data value must exist BEFORE the instance is created. Use an
  // sv.wire for the read data, feed it to the instance, and assign it after
  // the instance from a REGISTERED read of the array (BRAM-port contract:
  // rd_data is valid the cycle after the DUT presents the address; writes
  // are clocked as before).
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

  // AXI: wires feeding the DUT's slave->master inputs (assigned from the
  // slave instances below) and the DUT's master->slave output values.
  llvm::StringMap<Value> axiInWires;
  llvm::StringMap<Value> axiOutVals;

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
    } else if (isAxiPort(port.getName())) {
      // AXI slave->master signal: a wire assigned from the per-bundle slave
      // instance after the DUT instance exists.
      auto axiWire = sv::WireOp::create(
          builder, loc, port.type,
          builder.getStringAttr(port.getName().str() + "_wire"));
      Value axiVal = sv::ReadInOutOp::create(builder, loc, axiWire);
      dutInputs.push_back(axiVal);
      axiInWires[port.getName()] = axiWire;
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
  Value readyVal;
  SmallVector<Value> scalarOutputValues;
  DenseMap<StringRef, Value> memAddrValues;
  DenseMap<StringRef, Value> memWrDataValues;
  DenseMap<StringRef, Value> memWrEnValues;
  DenseMap<StringRef, Value> memRdEnValues;

  unsigned outIdx = 0;
  for (auto &port : dutPorts) {
    if (!port.isOutput())
      continue;
    Value result = dutInst.getResult(outIdx++);
    if (port.getName() == "done") {
      doneVal = result;
    } else if (port.getName() == "ready") {
      readyVal = result;
    } else if (isAxiPort(port.getName())) {
      axiOutVals[port.getName()] = result;
    } else if (port.getName().ends_with("_addr")) {
      StringRef prefix = port.getName().drop_back(5); // remove "_addr"
      memAddrValues[prefix] = result;
    } else if (port.getName().ends_with("_wr_data")) {
      StringRef prefix = port.getName().drop_back(8); // remove "_wr_data"
      memWrDataValues[prefix] = result;
    } else if (port.getName().ends_with("_wr_en")) {
      StringRef prefix = port.getName().drop_back(6); // remove "_wr_en"
      memWrEnValues[prefix] = result;
    } else if (port.getName().ends_with("_rd_en")) {
      StringRef prefix = port.getName().drop_back(6); // remove "_rd_en"
      memRdEnValues[prefix] = result;
    } else {
      scalarOutputValues.push_back(result);
    }
  }

  // --- Wire memory combinational reads: mem[txn*K + addr] → rd_data ---
  // The shift `tb_issue_count * K` selects the slot for the transaction
  // currently entering stage 0. tb_issue_count is the pre-increment
  // register value at the cycle a `start` is accepted (the DUT samples
  // mem<i>_rd_data at that cycle); after the cycle it ticks to T+1.
  unsigned partAddrWidth = std::max(
      1u, llvm::Log2_64_Ceil(MAX_N_TXNS) +
              [&]() {
                unsigned m = 0;
                for (auto &g : memGroups)
                  m = std::max<unsigned>(m, g.addrWidth);
                return m;
              }());
  auto partAddrType = builder.getIntegerType(partAddrWidth);
  for (auto &group : memGroups) {
    auto &info = memInfoMap[group.name];
    Value addr = memAddrValues[group.name];
    // Width-extend the DUT addr (group.addrWidth) and the issue count
    // (i32) to a common width that fits MAX_N_TXNS * group.depth.
    Value addrExt = comb::ConcatOp::create(
        builder, loc,
        ValueRange{hw::ConstantOp::create(
                       builder, loc,
                       builder.getIntegerType(partAddrWidth - group.addrWidth),
                       0),
                   addr});
    Value txnExt = comb::ExtractOp::create(builder, loc, partAddrType,
                                            shiftForMem(group.name), 0);
    Value kConst = hw::ConstantOp::create(
        builder, loc, partAddrType, group.depth);
    Value shift = comb::MulOp::create(builder, loc, txnExt, kConst);
    Value effectiveAddr =
        comb::AddOp::create(builder, loc, shift, addrExt);
    Value elemRef =
        sv::ArrayIndexInOutOp::create(builder, loc, info.reg, effectiveAddr);
    Value rdData = sv::ReadInOutOp::create(builder, loc, elemRef);
    // BRAM-port contract: external memories present latency-1 registered
    // reads. Register the (transaction-shifted) array read so rd_data is
    // valid the cycle AFTER the DUT presents the address, exactly like a
    // synchronous BRAM. When the DUT exposes a read enable, the register
    // only updates on enabled cycles, so the last read value HOLDS between
    // reads (ram_1rw-style content_en semantics). The index and address
    // are sampled together in the issue cycle, so the read-shift
    // transaction selection stays coherent.
    auto rdReg = sv::RegOp::create(
        builder, loc, group.dataType,
        builder.getStringAttr(group.name + "_rd_reg"));
    auto rdEnIt = memRdEnValues.find(group.name);
    Value rdEnVal = rdEnIt != memRdEnValues.end() ? rdEnIt->second : Value();
    sv::AlwaysFFOp::create(builder, loc, sv::EventControl::AtPosEdge, clk,
                           [&] {
                             if (rdEnVal) {
                               sv::IfOp::create(builder, loc, rdEnVal, [&] {
                                 sv::PAssignOp::create(builder, loc, rdReg,
                                                       rdData);
                               });
                             } else {
                               sv::PAssignOp::create(builder, loc, rdReg,
                                                     rdData);
                             }
                           });
    Value rdRegVal = sv::ReadInOutOp::create(builder, loc, rdReg);
    sv::AssignOp::create(builder, loc, info.readVal, rdRegVal);
  }

  // --- AXI slave instances: one behavioral slave per bundle ---
  // The slave model lives in hdl/systemverilog/axi_slave_mem.sv (compiled
  // into every Verilator build). Each instance is hex-initialized from
  // <bundle>.hex and dumps its memory in @MEM format on the first
  // tb_axi_dump pulse; the TB delays $finish two cycles so the slaves flush
  // before DONE.
  Value axiDumpVal, axiFinishWaitVal;
  sv::RegOp axiDumpReg, axiFinishWaitReg;
  auto i2Type = builder.getIntegerType(2);
  if (!axiBundles.empty()) {
    axiDumpReg = sv::RegOp::create(builder, loc, builder.getI1Type(),
                                   builder.getStringAttr("tb_axi_dump"));
    axiDumpVal = sv::ReadInOutOp::create(builder, loc, axiDumpReg);
    axiFinishWaitReg = sv::RegOp::create(
        builder, loc, i2Type, builder.getStringAttr("tb_axi_finish_wait"));
    axiFinishWaitVal = sv::ReadInOutOp::create(builder, loc, axiFinishWaitReg);

    // One extern decl per distinct data width (the SV module is
    // width-generic via DATA_W, so all map to the same verilog name).
    auto noneType = builder.getNoneType();
    auto i32ParamType = builder.getIntegerType(32);
    DenseMap<unsigned, hw::HWModuleExternOp> externs;
    OpBuilder externBuilder = OpBuilder::atBlockEnd(moduleOp.getBody());
    for (auto &b : axiBundles) {
      auto &ext = externs[b.dataW];
      if (!ext) {
        SmallVector<hw::PortInfo> extPorts;
        auto addPort = [&](StringRef pname, Type ty,
                           hw::ModulePort::Direction dir) {
          extPorts.push_back({{builder.getStringAttr(pname), ty, dir}});
        };
        addPort("clk", builder.getI1Type(), hw::ModulePort::Direction::Input);
        addPort("rst", builder.getI1Type(), hw::ModulePort::Direction::Input);
        addPort("dump", builder.getI1Type(), hw::ModulePort::Direction::Input);
        for (auto &desc : kAxiSigs)
          addPort(("m_axi_" + StringRef(desc.sig)).str(),
                  b.ports[desc.sig].type,
                  desc.dutOutput ? hw::ModulePort::Direction::Input
                                 : hw::ModulePort::Direction::Output);
        SmallVector<Attribute> paramDecls = {
            hw::ParamDeclAttr::get("NAME", noneType),
            hw::ParamDeclAttr::get("INIT_FILE", noneType),
            hw::ParamDeclAttr::get("DEPTH", i32ParamType),
            hw::ParamDeclAttr::get("DATA_W", i32ParamType),
            hw::ParamDeclAttr::get("ADDR_SHIFT", i32ParamType)};
        ext = hw::HWModuleExternOp::create(
            externBuilder, loc,
            builder.getStringAttr("axi_slave_mem_w" +
                                  std::to_string(b.dataW)),
            ArrayRef<hw::PortInfo>(extPorts), "axi_slave_mem",
            builder.getArrayAttr(paramDecls));
      }

      SmallVector<Attribute> paramVals = {
          hw::ParamDeclAttr::get("NAME", builder.getStringAttr(b.name)),
          hw::ParamDeclAttr::get(
              "INIT_FILE",
              builder.getStringAttr(dataDir + "/" + b.name + ".hex")),
          hw::ParamDeclAttr::get(
              "DEPTH", builder.getI32IntegerAttr((int32_t)b.depth)),
          hw::ParamDeclAttr::get("DATA_W",
                                 builder.getI32IntegerAttr(b.dataW)),
          hw::ParamDeclAttr::get("ADDR_SHIFT",
                                 builder.getI32IntegerAttr(b.addrShift))};
      SmallVector<Value> operands = {clk, rst, axiDumpVal};
      for (auto &desc : kAxiSigs)
        if (desc.dutOutput)
          operands.push_back(
              axiOutVals[(b.name + "_m_axi_" + desc.sig).c_str()]);
      auto slaveInst = hw::InstanceOp::create(
          builder, loc, ext, builder.getStringAttr(b.name + "_slave"),
          operands, builder.getArrayAttr(paramVals));
      unsigned resIdx = 0;
      for (auto &desc : kAxiSigs)
        if (!desc.dutOutput)
          sv::AssignOp::create(
              builder, loc,
              axiInWires[(b.name + "_m_axi_" + desc.sig).c_str()],
              slaveInst.getResult(resIdx++));
    }
  }

  // --- always_ff: control, memory writes, output dump ---
  sv::AlwaysFFOp::create(
      builder, loc, sv::EventControl::AtPosEdge, clk, [&] {
        sv::IfOp::create(
            builder, loc, rst,
            // Reset.
            [&] {
              sv::PAssignOp::create(builder, loc, ctrReg, c0_i8);
              sv::PAssignOp::create(builder, loc, doneSeenReg, falseVal);
              Value c0_i32 =
                  hw::ConstantOp::create(builder, loc, i32Type, 0);
              sv::PAssignOp::create(builder, loc, issueCntReg, c0_i32);
              sv::PAssignOp::create(builder, loc, doneCntReg, c0_i32);
              for (auto &reg : shiftChainRegs)
                sv::PAssignOp::create(builder, loc, reg, c0_i32);
              if (!axiBundles.empty()) {
                sv::PAssignOp::create(builder, loc, axiDumpReg, falseVal);
                sv::PAssignOp::create(
                    builder, loc, axiFinishWaitReg,
                    hw::ConstantOp::create(builder, loc, i2Type, 0));
              }
              // Per-(mem, addr) write counters are zero-initialized at
              // sim start in their `initial` block (alongside $readmemh).
              // We don't re-zero them on rst because verilator rejects
              // delayed (NBA) array writes inside for-loops, and tests
              // only ever pulse rst once per simulation.
            },
            // Normal operation.
            [&] {
              Value nextCtr =
                  comb::AddOp::create(builder, loc, ctrVal, c1_i8);
              sv::PAssignOp::create(builder, loc, ctrReg, nextCtr);

              // tb_start is a *wire* assigned combinationally (see assign
              // below the always_ff) so it tracks the DUT's `ready` on
              // the same cycle. Here we just track the issue count by
              // observing the same combinational predicate.
              sv::IfOp::create(builder, loc, startVal, [&] {
                Value c1 =
                    hw::ConstantOp::create(builder, loc, i32Type, 1);
                Value nextIssue =
                    comb::AddOp::create(builder, loc, issueCntVal, c1);
                sv::PAssignOp::create(builder, loc, issueCntReg, nextIssue);
              });
              // Read-shift chain. chain[0] increments on accepted start
              // (clamped to numTxns-1 so it stays valid for last-txn
              // reads). chain[s>=1] copies chain[s-1] every cycle, so
              // chain[s] at cycle T = chain[0] at cycle T-s = txn ID
              // currently at stage s.
              {
                Value c1 =
                    hw::ConstantOp::create(builder, loc, i32Type, 1);
                Value nextChain0 = comb::AddOp::create(
                    builder, loc, shiftChainVals[0], c1);
                Value moreAfter = comb::ICmpOp::create(
                    builder, loc, comb::ICmpPredicate::ult, nextChain0,
                    numTxns);
                Value chain0Bumped = comb::MuxOp::create(
                    builder, loc, moreAfter, nextChain0,
                    shiftChainVals[0]);
                Value nextChain0Final = comb::MuxOp::create(
                    builder, loc, startVal, chain0Bumped,
                    shiftChainVals[0]);
                sv::PAssignOp::create(builder, loc, shiftChainRegs[0],
                                       nextChain0Final);
                for (unsigned s = 1; s < shiftChainRegs.size(); ++s)
                  sv::PAssignOp::create(builder, loc, shiftChainRegs[s],
                                         shiftChainVals[s - 1]);
              }

              // Memory writes: when wr_en high, look up per-addr write
              // counter, place the value in slot `count*K + addr`, and
              // bump the counter. The Tth write to address A goes to
              // transaction T's partition. Pipeline ordering preserves
              // per-(mem,addr) issue order, so the count = the txn id
              // for that write.
              for (auto &group : memGroups) {
                auto &info = memInfoMap[group.name];
                Value wrEn = memWrEnValues[group.name];
                Value addr = memAddrValues[group.name];
                Value wrData = memWrDataValues[group.name];

                sv::IfOp::create(builder, loc, wrEn, [&] {
                  // Lookup current write count for this address.
                  Value cntElemRef = sv::ArrayIndexInOutOp::create(
                      builder, loc, info.writeCntReg, addr);
                  Value cntVal =
                      sv::ReadInOutOp::create(builder, loc, cntElemRef);
                  // Compute effective address = cnt * K + addr.
                  Value cntTrunc = comb::ExtractOp::create(
                      builder, loc, partAddrType, cntVal, 0);
                  Value kConst = hw::ConstantOp::create(
                      builder, loc, partAddrType, group.depth);
                  Value shift =
                      comb::MulOp::create(builder, loc, cntTrunc, kConst);
                  Value addrExt = comb::ConcatOp::create(
                      builder, loc,
                      ValueRange{hw::ConstantOp::create(
                                     builder, loc,
                                     builder.getIntegerType(partAddrWidth -
                                                             group.addrWidth),
                                     0),
                                 addr});
                  Value effectiveAddr =
                      comb::AddOp::create(builder, loc, shift, addrExt);
                  Value elemRef = sv::ArrayIndexInOutOp::create(
                      builder, loc, info.reg, effectiveAddr);
                  sv::PAssignOp::create(builder, loc, elemRef, wrData);
                  // Increment the counter for this address.
                  Value cntPlus1 = comb::AddOp::create(
                      builder, loc, cntVal,
                      hw::ConstantOp::create(builder, loc, i32Type, 1));
                  sv::PAssignOp::create(builder, loc, cntElemRef, cntPlus1);
                });
              }

              // Done handling: count pulses; only dump+finish on the
              // *final* done. Each done pulse increments tb_done_count;
              // the dump fires when tb_done_count reaches num_txns. This
              // preserves N=1 behavior (immediate dump on first done).
              sv::IfOp::create(builder, loc, doneVal, [&] {
                Value nextDone = comb::AddOp::create(
                    builder, loc, doneCntVal,
                    hw::ConstantOp::create(builder, loc, i32Type, 1));
                sv::PAssignOp::create(builder, loc, doneCntReg, nextDone);
                Value isLastDone = comb::ICmpOp::create(
                    builder, loc, comb::ICmpPredicate::eq, nextDone, numTxns);
                Value notDoneSeen =
                    comb::XorOp::create(builder, loc, doneSeenVal, trueVal);
                Value shouldDump =
                    comb::AndOp::create(builder, loc, isLastDone, notDoneSeen);
                sv::IfOp::create(builder, loc, shouldDump, [&] {
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

                  // Dump memory contents to stdout. With multi-txn
                  // partitioning the memory has `numTxns * group.depth`
                  // live entries — the rest is unused (X). Dump only
                  // the live region. Header reports `numTxns*depth` so
                  // the parser knows how many entries to read.
                  for (auto &group : memGroups) {
                    auto &info = memInfoMap[group.name];
                    unsigned width =
                        cast<IntegerType>(group.dataType).getWidth();
                    unsigned hexDigits = (width + 3) / 4;

                    // Header: @MEM name <numTxns * group.depth> <group.depth>
                    // (Both values printed so the Python parser can
                    // demux per-txn outputs without separate metadata.)
                    std::string headerFmt = "@MEM " + group.name + " %0d " +
                                             std::to_string(group.depth) +
                                             "\n";
                    Value kConst =
                        hw::ConstantOp::create(builder, loc, i32Type,
                                               group.depth);
                    Value totalLive =
                        comb::MulOp::create(builder, loc, numTxns, kConst);
                    sv::FWriteOp::create(builder, loc, fd, headerFmt,
                                         ValueRange{totalLive});

                    // Iterate from 0 to numTxns * group.depth.
                    unsigned idxWidth = std::max(
                        1u, llvm::Log2_64_Ceil(MAX_N_TXNS * group.depth) + 1);
                    auto idxType = builder.getIntegerType(idxWidth);
                    Value lb =
                        hw::ConstantOp::create(builder, loc, idxType, 0);
                    Value ubExt = comb::ExtractOp::create(
                        builder, loc, idxType, totalLive, 0);
                    Value step =
                        hw::ConstantOp::create(builder, loc, idxType, 1);

                    sv::ForOp::create(
                        builder, loc, lb, ubExt, step, "i",
                        [&](BlockArgument iv) {
                          // `iv` is `idxType` wide, which already covers
                          // `MAX_N_TXNS * group.depth` slots — the partitioned
                          // memory's full address space. Don't try to coerce
                          // it through `partAddrType` (a *global* width based
                          // on the largest mem in the group): if this group
                          // is smaller, `partAddrType` has more bits than
                          // `iv` and the extract would fail.
                          Value elemRef = sv::ArrayIndexInOutOp::create(
                              builder, loc, info.reg, iv);
                          Value elem =
                              sv::ReadInOutOp::create(builder, loc, elemRef);
                          std::string elemFmtStr = std::string("%0") +
                                                    std::to_string(hexDigits) +
                                                    "h\n";
                          sv::FWriteOp::create(builder, loc, fd, elemFmtStr,
                                               ValueRange{elem});
                        });
                  }

                  // Print cycle count so downstream tooling can report it.
                  sv::FWriteOp::create(builder, loc, fd, "@CYCLES %0d\n",
                                       ValueRange{ctrVal});
                  if (axiBundles.empty()) {
                    // Print end marker.
                    sv::FWriteOp::create(builder, loc, fd, "DONE\n",
                                         ValueRange{});
                    sv::FinishOp::create(builder, loc, 0);
                  } else {
                    // AXI slaves dump their memories on the NEXT posedge
                    // (they see tb_axi_dump at T+1); DONE/$finish are
                    // deferred two cycles via tb_axi_finish_wait so the
                    // end marker stays last in the stream.
                    sv::PAssignOp::create(builder, loc, axiDumpReg, trueVal);
                  }
                });
              });

              // AXI epilogue: T (dump set) -> T+1 (slaves flush, wait 0->1)
              // -> T+2 (DONE + $finish).
              if (!axiBundles.empty()) {
                sv::IfOp::create(builder, loc, axiDumpVal, [&] {
                  Value c1_i2 =
                      hw::ConstantOp::create(builder, loc, i2Type, 1);
                  Value nextWait = comb::AddOp::create(
                      builder, loc, axiFinishWaitVal, c1_i2);
                  sv::PAssignOp::create(builder, loc, axiFinishWaitReg,
                                        nextWait);
                  Value waitDone = comb::ICmpOp::create(
                      builder, loc, comb::ICmpPredicate::eq, axiFinishWaitVal,
                      c1_i2);
                  sv::IfOp::create(builder, loc, waitDone, [&] {
                    sv::FWriteOp::create(builder, loc, fd, "DONE\n",
                                         ValueRange{});
                    sv::FinishOp::create(builder, loc, 0);
                  });
                });
              }

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

  // Combinational drive of tb_start: the DUT's `ready` line gates start
  // and the issue counter limits how many starts we issue. Using
  // `assign` (not a registered drive) keeps tb_start in sync with
  // `_dut_ready` on the same cycle — a registered start would lag one
  // cycle and could be silently rejected if `ready` flips.
  {
    Value readyHigh = readyVal ? readyVal : trueVal;
    Value moreToIssue = comb::ICmpOp::create(
        builder, loc, comb::ICmpPredicate::ult, issueCntVal, numTxns);
    Value shouldStart =
        comb::AndOp::create(builder, loc, readyHigh, moreToIssue);
    sv::AssignOp::create(builder, loc, startWire, shouldStart);
  }

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

std::unique_ptr<mlir::Pass>
circt::loopschedule::createLoopScheduleTestbenchGenerationPass(
    const LoopScheduleTestbenchGenerationOptions &options) {
  return std::make_unique<LoopScheduleTestbenchGenerationPass>(options);
}
