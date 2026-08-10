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

/// One physical access port of a memory. A memory with `count > 1` (a
/// dual-port BRAM arg) exposes several of these over the same backing array,
/// so the read/write drive is generated per physical port while the array,
/// hex init, and `@MEM` dump stay per logical memory.
struct MemPort {
  std::string physPrefix; // "mem0" (AMC) or "mem0_bram0"/"mem0_bram1" (BRAM)
  hw::PortInfo rdData;     // input: read data (AMC: _rd_data, BRAM: _dout)
  hw::PortInfo addr;       // output: address
  hw::PortInfo wrData;     // output: write data (AMC: _wr_data, BRAM: _din)
  hw::PortInfo wrEn;       // output: write enable (AMC: _wr_en, BRAM: _we)
  // Optional AMC read enable (mem<i>_rd_en). When present, the behavioral
  // memory only updates its read register on enabled cycles (BRAM hold
  // semantics); absent (older DUTs) means always-enabled.
  hw::PortInfo rdEn;
  // Xilinx-BRAM chip-enable (= rd_en | wr_en); the effective read-enable is
  // recovered as `en & ~we`.
  hw::PortInfo en;
};

/// A logical memory at the DUT boundary and its one-or-more physical ports.
/// Only 1-D I/O memories are supported — multi-dim memrefs would need a richer
/// testbench model and are intentionally rejected.
struct MemPortGroup {
  std::string name;       // e.g. "mem0"
  // Xilinx-BRAM adapter interface (see lib/Support/BramInterfaceGen.cpp):
  // the boundary speaks addr/en/we/din/dout instead of the AMC memory-port
  // protocol. `isBram` selects that decoding.
  bool isBram = false;
  llvm::SmallVector<MemPort, 2> ports;
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
    MemPort mp;
    mp.physPrefix = prefix;
    bool multiDim = false;

    for (auto &port : dutPorts) {
      StringRef pname = port.getName();
      if (pname == prefix + "_rd_data")
        mp.rdData = port;
      else if (pname == prefix + "_addr")
        mp.addr = port;
      else if (pname == prefix + "_wr_data")
        mp.wrData = port;
      else if (pname == prefix + "_wr_en")
        mp.wrEn = port;
      else if (pname == prefix + "_rd_en")
        mp.rdEn = port;
      else if (pname.starts_with(prefix + "_addr_"))
        multiDim = true;
    }

    if (multiDim)
      continue;

    // Verify all four ports exist (a null port name means not found).
    if (!mp.rdData.name || !mp.addr.name || !mp.wrData.name || !mp.wrEn.name)
      continue;

    group.dataType = mp.rdData.type;
    group.addrWidth = cast<IntegerType>(mp.addr.type).getWidth();
    group.depth = 1u << group.addrWidth;
    group.ports.push_back(mp);
    groups.push_back(group);
  }
  return groups;
}

/// Detect external Block-RAM adapter port groups from DUT ports. The
/// `amc.memory_ref` → `amc.expand_ref` → `buildBramInterface` path exposes a
/// single logical memory `mem<i>` as one or more physical BRAM ports named
/// `mem<i>_bram<k>_{addr,en,we,din,dout}` (see BramInterfaceGen.cpp). Each
/// physical port drives the shared backing array independently; a dual-port
/// (multi-access) arg yields two ports over one memory.
static SmallVector<MemPortGroup, 2>
classifyBramMemoryPorts(const SmallVector<hw::PortInfo> &dutPorts) {
  // Candidate logical names from `bram0`'s data ports: `_bram0_dout`
  // (readable) or, for write-only memories, `_bram0_din`.
  SmallVector<std::string> names;
  llvm::SmallDenseSet<StringRef> seenName;
  for (auto &port : dutPorts) {
    StringRef pname = port.getName();
    StringRef base;
    if (pname.ends_with("_bram0_dout"))
      base = pname.drop_back(11); // "_bram0_dout"
    else if (pname.ends_with("_bram0_din"))
      base = pname.drop_back(10); // "_bram0_din"
    else
      continue;
    if (seenName.insert(base).second)
      names.push_back(base.str());
  }

  SmallVector<MemPortGroup, 2> groups;
  for (auto &name : names) {
    MemPortGroup group;
    group.name = name;
    group.isBram = true;
    // Gather every physical port `<name>_bram<k>_*`, in ascending k order.
    for (unsigned k = 0;; ++k) {
      std::string p = name + "_bram" + std::to_string(k);
      MemPort mp;
      mp.physPrefix = p;
      for (auto &port : dutPorts) {
        StringRef pn = port.getName();
        if (pn == p + "_addr")
          mp.addr = port;
        else if (pn == p + "_dout")
          mp.rdData = port;
        else if (pn == p + "_din")
          mp.wrData = port;
        else if (pn == p + "_we")
          mp.wrEn = port;
        else if (pn == p + "_en")
          mp.en = port;
      }
      // Address + chip-enable are mandatory for a real port; their absence
      // (a null port name on the default-constructed PortInfo) means we've
      // run past the last physical port.
      if (!mp.addr.name || !mp.en.name)
        break;
      if (group.ports.empty()) {
        // Element type / depth are memory-wide; take them from the first port.
        group.dataType = mp.rdData.name ? mp.rdData.type : mp.wrData.type;
        group.addrWidth = cast<IntegerType>(mp.addr.type).getWidth();
        group.depth = 1u << group.addrWidth;
      }
      group.ports.push_back(mp);
    }
    if (!group.ports.empty())
      groups.push_back(group);
  }
  return groups;
}

/// Check if a port belongs to any memory group.
static bool isMemoryPort(StringRef name,
                         const SmallVectorImpl<MemPortGroup> &memGroups) {
  for (auto &g : memGroups) {
    for (auto &mp : g.ports) {
      const std::string &p = mp.physPrefix;
      if (g.isBram) {
        if (name == p + "_addr" || name == p + "_en" || name == p + "_we" ||
            name == p + "_din" || name == p + "_dout")
          return true;
      } else if (name == p + "_rd_data" || name == p + "_addr" ||
                 name == p + "_wr_data" || name == p + "_wr_en" ||
                 name == p + "_rd_en") {
        return true;
      }
    }
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
  /// Position in the `amc.axi_bundles` array, which is the order the
  /// s_axilite register map is laid out in.
  unsigned metaIndex = 0;
  /// Element count of each ARGUMENT on this bundle, in base-register order,
  /// and the element offset this testbench places each of them at inside the
  /// bundle's one behavioral slave (see `sharedBundleOffsets`).
  llvm::SmallVector<uint64_t, 1> argElems;
  llvm::SmallVector<uint64_t, 1> argOffsets;
  /// DUT ports keyed by bare signal name ("araddr", "rdata", ...).
  llvm::StringMap<hw::PortInfo> ports;
};
} // namespace

/// Gap, in elements, this testbench leaves between the arrays of a SHARED
/// m_axi bundle. Deliberately small and odd: it is there to make the layout
/// non-contiguous and non-power-of-two-aligned, not to be convenient.
static constexpr uint64_t kSharedArgGapElems = 3;

/// Where this testbench PLACES each argument of an m_axi bundle inside the
/// bundle's one behavioral slave, and how big that slave must be.
///
/// Every argument has its own runtime base register, so the placement is the
/// HARNESS's choice, not the compiler's — and the choice here is deliberately
/// hostile to any surviving compile-time packed-layout assumption: REVERSE
/// declaration order, with a gap between the arrays, so declaration order is
/// not address order and nothing is contiguous.
///
/// A single-argument (dedicated) bundle is offset 0, depth = its element
/// count: byte-identical to the layout before per-argument bases existed.
///
/// KEEP IN SYNC with `shared_bundle_offsets` in allo/allo/backend/amc.py,
/// which writes the hex image this placement reads and splits the dumped
/// memory back apart at the same offsets.
static std::pair<llvm::SmallVector<uint64_t, 1>, uint64_t>
sharedBundleOffsets(ArrayRef<uint64_t> argElems) {
  llvm::SmallVector<uint64_t, 1> offsets(argElems.size(), 0);
  if (argElems.size() <= 1)
    return {offsets, argElems.empty() ? 0 : argElems[0]};
  uint64_t off = 0;
  for (unsigned k = argElems.size(); k-- > 0;) {
    offsets[k] = off;
    off += argElems[k] + kSharedArgGapElems;
  }
  return {offsets, offsets[0] + argElems[0]};
}

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

/// Ports of the axil_handshake wrapper's s_axilite control face (plus its
/// interrupt line). Driven by the axi_lite_ctrl_bfm instance, never by hex
/// files, and never dumped as scalar outputs.
static constexpr StringLiteral kCtrlSlavePrefix = "s_axi_control_";
static bool isCtrlSlavePort(StringRef name) {
  return name.starts_with(kCtrlSlavePrefix) || name == "interrupt";
}

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
      for (auto [idx, attr] : llvm::enumerate(bundlesAttr)) {
        auto dict = dyn_cast<DictionaryAttr>(attr);
        if (!dict)
          continue;
        auto nameAttr = dict.getAs<StringAttr>("name");
        auto depthAttr = dict.getAs<IntegerAttr>("depth");
        if (nameAttr && depthAttr && nameAttr.getValue() == kv.first) {
          info.metaIndex = (unsigned)idx;
          // Per-ARGUMENT element counts (one per runtime base register).
          // Absent metadata means a single argument spanning the whole ram.
          if (auto argElems = dict.getAs<DenseI64ArrayAttr>("arg_elems"))
            for (int64_t n : argElems.asArrayRef())
              info.argElems.push_back((uint64_t)n);
          if (info.argElems.empty())
            info.argElems.push_back((uint64_t)depthAttr.getInt());
          std::tie(info.argOffsets, info.depth) =
              sharedBundleOffsets(info.argElems);
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
  // External Block-RAM adapter memories (amc.memory_ref args lowered via
  // buildBramInterface) present a different boundary protocol; fold them into
  // the same group list so the drive/dump logic below treats them uniformly.
  for (auto &g : classifyBramMemoryPorts(dutPorts))
    memGroups.push_back(g);

  // axil_handshake DUTs (wrapped by amc-insert-axi-lite-control) have no raw
  // start/ready/done handshake; they are driven through their s_axilite
  // control slave by a behavioral AXI-Lite master (axi_lite_ctrl_bfm.sv).
  auto ctrlAttr =
      dutMod->getAttrOfType<StringAttr>("amc.control_interface");
  const bool ctrlHs = ctrlAttr && ctrlAttr.getValue() == "axil_handshake";

  // AXI bundles get a behavioral slave instance each (axi_slave_mem.sv).
  SmallVector<AxiBundleInfo> axiBundles;
  if (failed(classifyAxiBundles(dutMod, dutPorts, axiBundles)))
    return signalPassFailure();

  // Classify scalar ports (excluding control, memory, and AXI ports).
  SmallVector<hw::PortInfo> scalarInputs;
  SmallVector<hw::PortInfo> scalarOutputs;
  for (auto &port : dutPorts) {
    if (isControlPort(port.getName()) ||
        isMemoryPort(port.getName(), memGroups) || isAxiPort(port.getName()) ||
        (ctrlHs && isCtrlSlavePort(port.getName())))
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
  // needing per-transaction tags from the DUT — write attribution uses
  // `tb_done_count`, the number of transactions that have already
  // finished, which IS the id of the transaction currently executing
  // (transactions are serialized on `done`; see the txn-slot register).
  //
  // It used to use a per-address write counter instead, on the theory
  // that "transaction T's write to address A is the (T+1)th write to A
  // overall". That holds only when every transaction writes each address
  // at most ONCE. A kernel that writes one address twice in a single
  // transaction — a scatter with a repeated destination, a
  // data-dependent write cursor rewriting its slot, `B[0] = src[i]` in a
  // loop — had its second and later writes filed under transactions
  // 1, 2, ... which nothing ever reads back, so the FIRST write to each
  // address won and the compiler looked like it had reordered stores.
  // It had not: the writes were issued in the right order, into the
  // wrong slots. Keying on the transaction id cannot misfile a repeated
  // write, and it also fixes the mirror case the counter got wrong (a
  // transaction that skips address A shifted every later transaction's
  // write to A down a slot).
  //
  // MAX_N_TXNS is a fixed compile-time bound on batch size. Tests need
  // num_transactions <= MAX_N_TXNS. Sim-only memory; not synthesized.
  const unsigned MAX_N_TXNS = 1024;

  struct MemInfo {
    sv::RegOp reg;          // partitioned: MAX_N_TXNS * group.depth
    Value readVal;          // combinational read wire (set later)
    unsigned perTxnDepth;   // = group.depth
  };
  DenseMap<StringRef, MemInfo> memInfoMap;

  for (auto &group : memGroups) {
    unsigned partitionedDepth = MAX_N_TXNS * group.depth;
    auto arrayType =
        hw::UnpackedArrayType::get(group.dataType, partitionedDepth);
    auto memReg = sv::RegOp::create(builder, loc, arrayType,
                                    builder.getStringAttr(group.name));

    sv::InitialOp::create(builder, loc, [&] {
      sv::ReadMemOp::create(builder, loc, memReg,
                            dataDir + "/" + group.name + ".hex",
                            MemBaseTypeAttr::MemBaseHex);
    });

    memInfoMap[group.name] = {memReg, Value(), group.depth};
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

  // --- axil_handshake scalar ARGUMENTS live in the control register map ---
  // Under axil_handshake a scalar argument is not a DUT port at all: it is a
  // register in the s_axilite map, and the host writes it before ap_start
  // (amc-insert-axi-lite-control moved it there and recorded the byte offset
  // it assigned in `amc.axil_scalars`). The value still comes from
  // `<arg>.hex` at simulation time — the Allo backend writes one hex file per
  // call against RTL it built once — so load it exactly as above, then split
  // it into the 32-bit words the BFM writes over AXI4-Lite.
  //
  // `axilScalarWordAddrs[i]` / `axilScalarWordVals[i]` are one 32-bit
  // register word each, in the order the BFM writes them.
  SmallVector<unsigned> axilScalarWordAddrs;
  SmallVector<Value> axilScalarWordVals;
  if (ctrlHs) {
    auto scalarsAttr = dutMod->getAttrOfType<ArrayAttr>("amc.axil_scalars");
    for (auto attr : scalarsAttr ? scalarsAttr.getValue()
                                 : ArrayRef<Attribute>()) {
      auto dict = dyn_cast<DictionaryAttr>(attr);
      auto nameAttr = dict ? dict.getAs<StringAttr>("name") : StringAttr();
      auto widthAttr = dict ? dict.getAs<IntegerAttr>("width") : IntegerAttr();
      auto offAttr = dict ? dict.getAs<IntegerAttr>("offset") : IntegerAttr();
      if (!nameAttr || !widthAttr || !offAttr) {
        dutMod.emitError("malformed amc.axil_scalars entry (need name, width, "
                         "offset)");
        return signalPassFailure();
      }
      unsigned width = (unsigned)widthAttr.getInt();
      auto scalarTy = builder.getIntegerType(width);
      auto arrayType = hw::UnpackedArrayType::get(scalarTy, 1);
      auto scalarReg = sv::RegOp::create(
          builder, loc, arrayType,
          builder.getStringAttr(nameAttr.getValue().str() + "_data"));
      sv::InitialOp::create(builder, loc, [&] {
        sv::ReadMemOp::create(builder, loc, scalarReg,
                              dataDir + "/" + nameAttr.getValue().str() +
                                  ".hex",
                              MemBaseTypeAttr::MemBaseHex);
      });
      Value idx = hw::ConstantOp::create(builder, loc,
                                         builder.getIntegerType(1), 0);
      Value elemRef =
          sv::ArrayIndexInOutOp::create(builder, loc, scalarReg, idx);
      Value scalarVal = sv::ReadInOutOp::create(builder, loc, elemRef);
      // Word 0 is the register's low half, matching the slave's layout.
      for (unsigned lo = 0; lo < width; lo += 32) {
        unsigned w = std::min(32u, width - lo);
        Value word = comb::ExtractOp::create(
            builder, loc, builder.getIntegerType(w), scalarVal, lo);
        if (w < 32) {
          Value pad = hw::ConstantOp::create(
              builder, loc, builder.getIntegerType(32 - w), 0);
          word = comb::ConcatOp::create(builder, loc,
                                        ArrayRef<Value>{pad, word});
        }
        axilScalarWordAddrs.push_back((unsigned)offAttr.getInt() +
                                      (lo / 32) * 4);
        axilScalarWordVals.push_back(word);
      }
    }
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

  // The memory dump is DEFERRED one posedge past the final done. With the
  // early-done cut-through, `done` can land on the very edge the kernel's
  // final store commits; that store is a non-blocking assign in this same
  // always_ff block, so a dump on the done edge reads the pre-store value
  // of the last address. One settle cycle restores the read-after-write
  // order (the AXI slave dump already works this way — it fires on the
  // NEXT posedge via tb_axi_dump). The cycle counter is latched AT done so
  // @CYCLES still reports the done edge, not the dump edge.
  auto dumpPendingReg = sv::RegOp::create(
      builder, loc, builder.getI1Type(),
      builder.getStringAttr("tb_dump_pending"));
  Value dumpPendingVal = sv::ReadInOutOp::create(builder, loc, dumpPendingReg);
  auto doneCyclesReg = sv::RegOp::create(
      builder, loc, i8Type, builder.getStringAttr("tb_done_cycles"));
  Value doneCyclesVal = sv::ReadInOutOp::create(builder, loc, doneCyclesReg);

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

  // Which transaction's memory image serves reads AND absorbs writes:
  // `tb_done_count`, the number of transactions that have already
  // finished, which is the id of the one now running. One register
  // suffices because transactions never overlap — sequential functions
  // serialize them on `done`, and func-level pipelining rejects memory
  // arguments outright (a memory argument has no per-transaction
  // identity, so overlapped transactions would each need their own image
  // — unsupported).
  //
  // Reads used to key off a separate `tb_txn_slot` that bumped on each
  // accepted `start` under the raw handshake (and on `done` only under
  // axil_handshake). Bumping on start makes the slot the id of the NEXT
  // transaction for the whole of the current one, so a raw-handshake
  // `run_batch` served every transaction its successor's inputs — masked
  // in-tree because the only multi-transaction test uses axil_handshake,
  // and by the clamp at the last slot, which made the final transaction
  // (the one a same-data `run_streaming` checks) come out right.
  Value txnSlotVal = doneCntVal;

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
  // Per physical read port: the sv.wire feeding its rd_data/dout DUT input,
  // keyed by the port's physical prefix (`mem0`, or `mem0_bram1` for the 2nd
  // port of a dual-port BRAM). Assigned from the registered array read below.
  DenseMap<StringRef, Value> physRdWire;

  // AXI: wires feeding the DUT's slave->master inputs (assigned from the
  // slave instances below) and the DUT's master->slave output values.
  llvm::StringMap<Value> axiInWires;
  llvm::StringMap<Value> axiOutVals;

  // axil_handshake: wires feeding the DUT's s_axi_control_* inputs (assigned
  // from the axi_lite_ctrl_bfm instance below) and the DUT's control-slave
  // output values (the BFM's inputs).
  llvm::StringMap<Value> ctrlInWires;
  llvm::StringMap<Value> ctrlOutVals;

  for (auto &port : dutPorts) {
    if (!port.isInput())
      continue;
    if (port.getName() == "clk") {
      dutInputs.push_back(clk);
    } else if (port.getName() == "rst") {
      dutInputs.push_back(rst);
    } else if (port.getName() == "start") {
      dutInputs.push_back(startVal);
    } else if (ctrlHs && isCtrlSlavePort(port.getName())) {
      auto ctrlWire = sv::WireOp::create(
          builder, loc, port.type,
          builder.getStringAttr(port.getName().str() + "_wire"));
      Value ctrlVal = sv::ReadInOutOp::create(builder, loc, ctrlWire);
      dutInputs.push_back(ctrlVal);
      ctrlInWires[port.getName()] = ctrlWire;
    } else if (port.getName().ends_with("_rd_data") ||
               port.getName().ends_with("_dout")) {
      // Memory read data — placeholder; will be wired after instance.
      // hw.instance inputs can't be back-patched, so feed the DUT an sv.wire
      // now and assign it after the instance from the registered array read.
      // Key the wire by the PHYSICAL port prefix: `<mem>_rd_data` (AMC) or
      // `<mem>_bram<k>_dout` (BRAM), so each port of a dual-port memory gets
      // its own read path.
      StringRef pn = port.getName();
      StringRef prefix = pn.ends_with("_dout")
                             ? pn.drop_back(5)  // "_dout"
                             : pn.drop_back(8); // "_rd_data"
      auto rdWire = sv::WireOp::create(builder, loc, port.type,
                                       builder.getStringAttr(
                                           prefix.str() + "_rd_wire"));
      Value rdVal = sv::ReadInOutOp::create(builder, loc, rdWire);
      dutInputs.push_back(rdVal);
      memRdDataInputIdx[prefix] = dutInputs.size() - 1;
      physRdWire[prefix] = rdWire;
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
  DenseMap<StringRef, Value> memBramEnValues; // BRAM chip-enable (en)

  // Memory value maps are keyed by PHYSICAL port prefix (`mem0`, or
  // `mem0_bram1` for the 2nd port of a dual-port BRAM) so each physical port
  // of a memory drives independently.
  unsigned outIdx = 0;
  for (auto &port : dutPorts) {
    if (!port.isOutput())
      continue;
    Value result = dutInst.getResult(outIdx++);
    StringRef pn = port.getName();
    // Order matters: the AMC `_wr_en`/`_rd_en` suffixes also end in `_en`, so
    // match them before the BRAM chip-enable `_en`.
    if (pn == "done") {
      doneVal = result;
    } else if (pn == "ready") {
      readyVal = result;
    } else if (ctrlHs && isCtrlSlavePort(pn)) {
      ctrlOutVals[pn] = result;
    } else if (isAxiPort(pn)) {
      axiOutVals[pn] = result;
    } else if (pn.ends_with("_wr_data")) {
      memWrDataValues[pn.drop_back(8)] = result; // AMC write data
    } else if (pn.ends_with("_wr_en")) {
      memWrEnValues[pn.drop_back(6)] = result; // AMC write enable
    } else if (pn.ends_with("_rd_en")) {
      memRdEnValues[pn.drop_back(6)] = result; // AMC read enable
    } else if (pn.ends_with("_din")) {
      memWrDataValues[pn.drop_back(4)] = result; // BRAM write data
    } else if (pn.ends_with("_we")) {
      memWrEnValues[pn.drop_back(3)] = result; // BRAM write enable
    } else if (pn.ends_with("_en")) {
      memBramEnValues[pn.drop_back(3)] = result; // BRAM chip enable
    } else if (pn.ends_with("_addr")) {
      memAddrValues[pn.drop_back(5)] = result; // address (AMC + BRAM)
    } else {
      scalarOutputValues.push_back(result);
    }
  }

  // For BRAM ports, recover the effective read-enable as `en & ~we` (the
  // adapter drives `en = rd_en | wr_en`, `we = wr_en`). This feeds the read
  // register's hold gating exactly like an AMC `rd_en` port.
  for (auto &group : memGroups) {
    if (!group.isBram)
      continue;
    for (auto &mp : group.ports) {
      Value en = memBramEnValues.lookup(mp.physPrefix);
      if (!en)
        continue;
      Value we = memWrEnValues.lookup(mp.physPrefix);
      Value rdEn = en;
      if (we) {
        Value notWe = comb::XorOp::create(builder, loc, we, trueVal);
        rdEn = comb::AndOp::create(builder, loc, en, notWe);
      }
      memRdEnValues[mp.physPrefix] = rdEn;
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
    // One read path per PHYSICAL read port; all share this memory's array and
    // transaction read-shift. Write-only ports have no rd wire and are skipped.
    for (auto [pi, mp] : llvm::enumerate(group.ports)) {
      Value rdWire = physRdWire.lookup(mp.physPrefix);
      if (!rdWire)
        continue;
      Value addr = memAddrValues[mp.physPrefix];
      // Width-extend the DUT addr (group.addrWidth) and the issue count
      // (i32) to a common width that fits MAX_N_TXNS * group.depth.
      Value addrExt = comb::ConcatOp::create(
          builder, loc,
          ValueRange{hw::ConstantOp::create(
                         builder, loc,
                         builder.getIntegerType(partAddrWidth -
                                                group.addrWidth),
                         0),
                     addr});
      Value txnExt = comb::ExtractOp::create(builder, loc, partAddrType,
                                              txnSlotVal, 0);
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
          builder.getStringAttr(mp.physPrefix + "_rd_reg"));
      auto rdEnIt = memRdEnValues.find(mp.physPrefix);
      Value rdEnVal =
          rdEnIt != memRdEnValues.end() ? rdEnIt->second : Value();
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
      sv::AssignOp::create(builder, loc, rdWire, rdRegVal);
    }
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

  // --- axil_handshake: behavioral AXI-Lite control master ---
  // The BFM (hdl/systemverilog/axi_lite_ctrl_bfm.sv) plays the host: it
  // programs the base-address registers and the scalar-argument registers,
  // then per transaction writes ap_start and polls ap_done, for `numTxns`
  // transactions. Its start/done pulses stand in for the raw start/done
  // handshake in the transaction accounting below (transactions are strictly
  // serialized under axil_handshake, so the few-cycle skew against the
  // kernel's internal start is harmless). The argument registers are written
  // ONCE, after reset: they persist across transactions, and the wrapper
  // re-captures the scalars on every launch.
  Value ctrlStartPulse;
  if (ctrlHs) {
    static const char *kBfmIns[] = {"awready", "wready", "bresp", "bvalid",
                                    "arready", "rdata",  "rresp", "rvalid"};
    static const char *kBfmOuts[] = {"awaddr", "awvalid", "wdata",
                                     "wstrb",  "wvalid",  "bready",
                                     "araddr", "arvalid", "rready"};
    llvm::StringMap<Type> ctrlPortTypes;
    for (auto &port : dutPorts)
      if (port.getName().starts_with(kCtrlSlavePrefix))
        ctrlPortTypes[port.getName().drop_front(kCtrlSlavePrefix.size())] =
            port.type;
    for (const char *sig : kBfmIns)
      if (!ctrlPortTypes.count(sig)) {
        dutMod.emitError("axil_handshake DUT is missing control-slave port '")
            << kCtrlSlavePrefix << sig << "'";
        return signalPassFailure();
      }
    for (const char *sig : kBfmOuts)
      if (!ctrlPortTypes.count(sig)) {
        dutMod.emitError("axil_handshake DUT is missing control-slave port '")
            << kCtrlSlavePrefix << sig << "'";
        return signalPassFailure();
      }
    unsigned ctrlAddrW =
        cast<IntegerType>(ctrlPortTypes["awaddr"]).getWidth();

    SmallVector<hw::PortInfo> extPorts;
    auto addPort = [&](StringRef pname, Type ty,
                       hw::ModulePort::Direction dir) {
      extPorts.push_back({{builder.getStringAttr(pname), ty, dir}});
    };
    // Packed scalar-register words, 32 bits each: {word[N-1], .., word[0]}.
    // The declared width follows the same never-truncate rule as BASE_VALUES.
    unsigned numScalarWords = axilScalarWordVals.size();
    auto scalarDataTy =
        builder.getIntegerType(32 * std::max(1u, numScalarWords));
    addPort("clk", builder.getI1Type(), hw::ModulePort::Direction::Input);
    addPort("rst", builder.getI1Type(), hw::ModulePort::Direction::Input);
    addPort("num_txns", i32Type, hw::ModulePort::Direction::Input);
    addPort("scalar_data", scalarDataTy, hw::ModulePort::Direction::Input);
    for (const char *sig : kBfmIns)
      addPort(("s_axi_" + StringRef(sig)).str(), ctrlPortTypes[sig],
              hw::ModulePort::Direction::Input);
    for (const char *sig : kBfmOuts)
      addPort(("s_axi_" + StringRef(sig)).str(), ctrlPortTypes[sig],
              hw::ModulePort::Direction::Output);
    addPort("start_pulse", builder.getI1Type(),
            hw::ModulePort::Direction::Output);
    addPort("done_pulse", builder.getI1Type(),
            hw::ModulePort::Direction::Output);

    // One 64-bit base register PER ARGUMENT, in bundle-then-argument order —
    // the order amc-insert-axi-lite-control lays the register map out in — and
    // each one holds the BYTE address this testbench placed that argument at
    // inside its bundle's slave (see `sharedBundleOffsets`). A dedicated
    // bundle's single argument sits at 0, so its register is written 0 exactly
    // as before per-argument bases existed.
    SmallVector<const AxiBundleInfo *> byRegOrder;
    for (auto &b : axiBundles)
      byRegOrder.push_back(&b);
    llvm::sort(byRegOrder, [](const AxiBundleInfo *a, const AxiBundleInfo *b) {
      return a->metaIndex < b->metaIndex;
    });
    SmallVector<uint64_t> baseValues;
    for (const AxiBundleInfo *b : byRegOrder)
      for (uint64_t off : b->argOffsets)
        baseValues.push_back(off << b->addrShift);
    // Packed {base[N-1], .., base[0]}, 64 bits each; the BFM slices it.
    APInt packed(std::max<unsigned>(64, 64 * baseValues.size()), 0);
    for (auto [k, v] : llvm::enumerate(baseValues))
      packed.insertBits(APInt(64, v), 64 * k);

    // Scalar-register WORD addresses, packed 32 bits each — the byte offsets
    // amc-insert-axi-lite-control recorded, not offsets recomputed here.
    APInt scalarAddrs(32 * std::max(1u, numScalarWords), 0);
    for (auto [k, a] : llvm::enumerate(axilScalarWordAddrs))
      scalarAddrs.insertBits(APInt(32, a), 32 * k);

    auto i32ParamType = builder.getIntegerType(32);
    SmallVector<Attribute> paramDecls = {
        hw::ParamDeclAttr::get("ADDR_W", i32ParamType),
        hw::ParamDeclAttr::get("NUM_BASE_ADDRS", i32ParamType),
        hw::ParamDeclAttr::get(
            "BASE_VALUES", builder.getIntegerType(packed.getBitWidth())),
        hw::ParamDeclAttr::get("NUM_SCALAR_WORDS", i32ParamType),
        hw::ParamDeclAttr::get(
            "SCALAR_ADDRS",
            builder.getIntegerType(scalarAddrs.getBitWidth()))};
    OpBuilder externBuilder = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto bfmExt = hw::HWModuleExternOp::create(
        externBuilder, loc, builder.getStringAttr("axi_lite_ctrl_bfm"),
        ArrayRef<hw::PortInfo>(extPorts), "axi_lite_ctrl_bfm",
        builder.getArrayAttr(paramDecls));

    SmallVector<Attribute> paramVals = {
        hw::ParamDeclAttr::get("ADDR_W", builder.getI32IntegerAttr(ctrlAddrW)),
        hw::ParamDeclAttr::get(
            "NUM_BASE_ADDRS",
            builder.getI32IntegerAttr((int32_t)baseValues.size())),
        hw::ParamDeclAttr::get("BASE_VALUES",
                               builder.getIntegerAttr(
                                   builder.getIntegerType(packed.getBitWidth()),
                                   packed)),
        hw::ParamDeclAttr::get(
            "NUM_SCALAR_WORDS",
            builder.getI32IntegerAttr((int32_t)numScalarWords)),
        hw::ParamDeclAttr::get(
            "SCALAR_ADDRS",
            builder.getIntegerAttr(
                builder.getIntegerType(scalarAddrs.getBitWidth()),
                scalarAddrs))};
    // Packed scalar data, word 0 in the low bits (concat takes MSB first).
    Value scalarData;
    if (numScalarWords) {
      SmallVector<Value> msbFirst(axilScalarWordVals.rbegin(),
                                  axilScalarWordVals.rend());
      scalarData = numScalarWords == 1
                       ? axilScalarWordVals[0]
                       : comb::ConcatOp::create(builder, loc, msbFirst)
                             .getResult();
    } else {
      scalarData = hw::ConstantOp::create(builder, loc, scalarDataTy, 0);
    }
    SmallVector<Value> operands = {clk, rst, numTxns, scalarData};
    for (const char *sig : kBfmIns)
      operands.push_back(ctrlOutVals[(kCtrlSlavePrefix + sig).str()]);
    auto bfmInst = hw::InstanceOp::create(
        builder, loc, bfmExt, builder.getStringAttr("ctrl_bfm"), operands,
        builder.getArrayAttr(paramVals));
    unsigned resIdx = 0;
    for (const char *sig : kBfmOuts)
      sv::AssignOp::create(builder, loc,
                           ctrlInWires[(kCtrlSlavePrefix + sig).str()],
                           bfmInst.getResult(resIdx++));
    ctrlStartPulse = bfmInst.getResult(resIdx++);
    // The BFM's done pulse is the DUT-completion signal for everything
    // downstream (dump + $finish).
    doneVal = bfmInst.getResult(resIdx++);
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
              if (!axiBundles.empty()) {
                sv::PAssignOp::create(builder, loc, axiDumpReg, falseVal);
                sv::PAssignOp::create(
                    builder, loc, axiFinishWaitReg,
                    hw::ConstantOp::create(builder, loc, i2Type, 0));
              }
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
              // Memory writes: when wr_en high, place the value in the
              // CURRENT transaction's slot, `tb_done_count*K + addr`.
              // `tb_done_count` is the number of transactions that have
              // already raised `done`, i.e. the id of the one now
              // running — it is read here pre-edge, so a write landing in
              // the same cycle as its own `done` still files under that
              // transaction. Repeated writes to one address within a
              // transaction all target the same slot, so the last writer
              // wins exactly as the source says (see the write-attribution
              // note on the memory arrays above). Each physical port of a
              // dual-port memory writes the shared array independently
              // (the scheduler guarantees the two ports never target the
              // same address in the same cycle).
              for (auto &group : memGroups) {
                auto &info = memInfoMap[group.name];
                for (auto &mp : group.ports) {
                  Value wrEn = memWrEnValues.lookup(mp.physPrefix);
                  Value wrData = memWrDataValues.lookup(mp.physPrefix);
                  if (!wrEn || !wrData)
                    continue; // read-only port
                  Value addr = memAddrValues[mp.physPrefix];

                  sv::IfOp::create(builder, loc, wrEn, [&] {
                    // Compute effective address = txn * K + addr.
                    Value txnTrunc = comb::ExtractOp::create(
                        builder, loc, partAddrType, doneCntVal, 0);
                    Value kConst = hw::ConstantOp::create(
                        builder, loc, partAddrType, group.depth);
                    Value shift =
                        comb::MulOp::create(builder, loc, txnTrunc, kConst);
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
                  });
                }
              }

              // Done handling: count pulses; arm the DEFERRED dump on the
              // *final* done. Each done pulse increments tb_done_count;
              // the dump itself runs one posedge later (tb_dump_pending)
              // so the final store's non-blocking assign — which may land
              // on the done edge under the early-done cut-through — is
              // visible to the dump reads.
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
                  sv::PAssignOp::create(builder, loc, dumpPendingReg, trueVal);
                  // @CYCLES reports the DONE edge, not the dump edge.
                  sv::PAssignOp::create(builder, loc, doneCyclesReg, ctrVal);
                });
              });

              // Deferred dump: one posedge after the final done.
              sv::IfOp::create(builder, loc, dumpPendingVal, [&] {
                Value falseC =
                    hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                           0);
                sv::PAssignOp::create(builder, loc, dumpPendingReg, falseC);
                {
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

                  // Print cycle count so downstream tooling can report it
                  // (latched at the done edge — the dump runs a cycle
                  // later).
                  sv::FWriteOp::create(builder, loc, fd, "@CYCLES %0d\n",
                                       ValueRange{doneCyclesVal});
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
                }
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
  //
  // axil_handshake DUTs have no start input; the BFM sequences the starts
  // itself, and tb_start just mirrors its start pulse so the issue-count /
  // read-shift accounting in the always_ff above keeps working.
  if (ctrlHs) {
    sv::AssignOp::create(builder, loc, startWire, ctrlStartPulse);
  } else {
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
