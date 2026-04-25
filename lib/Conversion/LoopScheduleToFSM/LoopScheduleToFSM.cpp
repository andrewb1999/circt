//===- LoopScheduleToFSM.cpp - LoopSchedule to FSM conversion pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LoopSchedule to FSM + HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Conversion/LoopScheduleToFSM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LoopSchedule/HWMemoryLoweringState.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Dialect/OpLib/OpLibOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
#define GEN_PASS_DEF_LOOPSCHEDULETOFSM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

namespace {

/// Per-dim address widths for a memref. Each dim contributes
/// `max(1, ceil_log2(dim_size))` so that even size-1 dims have a valid
/// 1-bit signal on the loop hw.module ports. The hlmem itself uses
/// `seq::HLMemType::getAddressTypes()` (without the floor of 1), so a
/// width-fixup may be needed at the hlmem boundary.
static SmallVector<unsigned> getDimAddrWidths(MemRefType memType) {
  SmallVector<unsigned> widths;
  for (int64_t dim : memType.getShape()) {
    unsigned w = (dim <= 1) ? 1u : llvm::Log2_64_Ceil(dim);
    widths.push_back(std::max(1u, w));
  }
  return widths;
}

/// Number of address signals for a memref's loop-module port set.
/// 0-rank memrefs collapse to 0 address ports (loads/stores still work
/// because there are no indices to drive).
static unsigned getNumAddrPorts(MemRefType memType) {
  return (unsigned)memType.getShape().size();
}

/// Resize an integer Value to a target bit-width via truncation or
/// zero-extension. Width-equal Values pass through.
static Value resizeIntTo(OpBuilder &builder, Location loc, Value v,
                         unsigned targetWidth) {
  auto srcWidth = cast<IntegerType>(v.getType()).getWidth();
  if (srcWidth == targetWidth)
    return v;
  auto targetType = IntegerType::get(builder.getContext(), targetWidth);
  if (srcWidth > targetWidth)
    return comb::ExtractOp::create(builder, loc, targetType, v, 0);
  // Zero-extend.
  auto padType = IntegerType::get(builder.getContext(), targetWidth - srcWidth);
  Value zero = hw::ConstantOp::create(builder, loc, padType, 0);
  return comb::ConcatOp::create(builder, loc, ValueRange{zero, v});
}

/// Check whether a type is a `!loopschedule.handle`. Handle-typed values are
/// scheduling metadata with no HW representation: they are not registered,
/// captured, or forwarded as module outputs.
static bool isHandleType(Type type) {
  return isa<loopschedule::HandleType>(type);
}

/// Create a zero constant of the given type.
static Value createZeroConstant(OpBuilder &builder, Location loc, Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return hw::ConstantOp::create(builder, loc, intType, 0);
  if (isa<IndexType>(type))
    return arith::ConstantIndexOp::create(builder, loc, 0);
  llvm_unreachable("unsupported type for zero constant");
}

/// Returns the per-op cycle latency stamped by SCFToLoopSchedule
/// (`loopschedule.cycle_latency`), defaulting to 1 if absent.
static unsigned getOpCycleLatency(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("loopschedule.cycle_latency"))
    return (unsigned)attr.getInt();
  return 1;
}

/// Compute the latency contribution of an `at` body: the max cycle_latency
/// of any op in the body (nested loops and launches are not expected here).
/// Returns at least 1.
static unsigned computeAtBodyLatency(Block &block) {
  unsigned maxLat = 0;
  for (Operation &op : block) {
    if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp>(op))
      continue;
    if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(op))
      continue;
    maxLat = std::max(maxLat, getOpCycleLatency(&op));
  }
  return std::max(maxLat, 1u);
}

/// Top-level latency of a frame: max(at.offset + atBodyLatency) over the
/// frame body's `at` children. A frame with only a single `at 0` whose body
/// is all 1-cycle ops has latency 1.
static unsigned computeFrameLatency(LoopScheduleFrameOp frame) {
  unsigned maxLat = 0;
  for (auto atOp : frame.getBodyBlock().getOps<LoopScheduleAtOp>()) {
    unsigned bodyLat = computeAtBodyLatency(atOp.getBodyBlock());
    maxLat = std::max(maxLat, (unsigned)atOp.getOffset() + bodyLat);
  }
  return std::max(maxLat, 1u);
}

/// Tracks the hw.module ports for a memref function argument.
/// `addrs` carries one address Value per memref dim (empty for 0-rank).
/// One hardware port's worth of address/data/enable drives. Used both as
/// the direct fields of `MemPortMapping` (for the default port 0) and as
/// the element type of `MemPortMapping::extraPorts` (for multi-port
/// memories, ports 1..N-1).
struct PortDrives {
  Value rdData;             // input: read data from memory
  SmallVector<Value> addrs; // per-dim address outputs
  Value wrData;             // write data output
  Value wrEn;               // write enable output
  // Optional read-enable output. Set by loads whose
  // HWLoadLoweringInterface::requiresReadEnable() is true (e.g. AMC
  // ports). Null means tie-high (memref case).
  Value rdEn;
};

struct MemPortMapping {
  // Port 0's drives are held directly as fields so the bulk of the
  // existing single-port codepaths don't have to change. Additional
  // ports live in `extraPorts` (port K>0 is `extraPorts[K-1]`). This is
  // the per-port data model the multi-port emission refactor consumes;
  // today `extraPorts` is always empty because `computeNumPortsFromUsers`
  // is stubbed to 1.
  Value rdData;
  SmallVector<Value> addrs;
  Value wrData;
  Value wrEn;
  Value rdEn;
  SmallVector<PortDrives> extraPorts;
  // Memory-driven "request completed this cycle" signal. Non-null for
  // amc ports (supplied by the hw.instance's `done` output). Null for
  // memref-backed local memories — the FSM treats missing `done` as
  // tied-high (no pipeline stall contribution).
  Value done;
};

/// Extend `mp.extraPorts` so port `k` is indexable; no-op for `k == 0`
/// (port 0 aliases the direct fields on `MemPortMapping`).
[[maybe_unused]] static void ensurePort(MemPortMapping &mp, unsigned k) {
  if (k == 0)
    return;
  while (mp.extraPorts.size() < k)
    mp.extraPorts.emplace_back();
}

/// Information about a memory-backed value (memref argument, local
/// memref.alloc, or amc.instance port) that gets threaded through loop
/// hw.modules. `isAmcPort` distinguishes memref-backed entries (whose
/// loop ports include implicit wr_data / wr_en outputs) from amc-port
/// entries (whose loop ports only include the directional signals the
/// port actually needs).
struct PortArgInfo {
  Value originalArg;
  // Unified metadata (populated for both memref and amc-port cases).
  SmallVector<int64_t> shape;
  Type elementType;
  SmallVector<unsigned> addrWidths;
  bool isRead = true;
  bool isWrite = true;
  bool requiresRdEn = false;
  unsigned latency = 1;
  // Number of distinct hardware ports the memory exposes. For memrefs,
  // this is derived from the oplib operator's `limit` (e.g. `mem_f_0
  // latency<1>, limit<2>` → numPorts == 2). Default 1. Amc-port entries
  // always have numPorts == 1 (the amc.instance port is its own unit).
  unsigned numPorts = 1;
  // Memref-specific.
  MemRefType memType;        // null for amc ports
  bool isLocalMem = false;   // true for memref.alloc → seq.hlmem
  // Amc-port-specific.
  bool isAmcPort = false;
};

/// Probe \p memref's memory-op users for `loopschedule.operator` and look
/// up the referenced `oplib.operator`'s declared `limit` attribute, which
/// `OperatorAllocation` emits as the memory's hardware port count.
/// Returns 1 if no binding metadata is present (single-port default).
static unsigned computeNumPortsFromUsers(Value memref) {
  // TODO(multi-port): full lookup is below but currently commented out.
  // With the lookup active, single-port AMC/CIRCT lit tests pass and the
  // top-level hw.module port list correctly doubles per-memref ports, but
  // a separate BackedgeBuilder type assertion fires somewhere in the
  // Allo end-to-end flow — an off-by-one in port-group result indexing
  // slips through that the single-port regressions don't catch. Keeping
  // stubbed until that's tracked down.
  (void)memref;
  return 1;
#if 0
  auto moduleOp = memref.getParentRegion()
                      ->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return 1;
  for (Operation *user : memref.getUsers()) {
    if (!isa<loopschedule::LoopScheduleLoadOp,
              loopschedule::LoopScheduleStoreOp>(user))
      continue;
    auto attr =
        user->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    if (!attr)
      continue;
    StringRef leaf = attr.getLeafReference().getValue();
    for (auto lib : moduleOp.getOps<oplib::LibraryOp>()) {
      for (auto op : lib.getBodyBlock()->getOps<oplib::OperatorOp>()) {
        if (op.getSymName() == leaf) {
          if (auto limit = op.getLimit())
            return static_cast<unsigned>(*limit);
          return 1;
        }
      }
    }
    break;
  }
  return 1;
#endif
}

/// Read an op's `loopschedule.binding` attr, defaulting to 0 (port 0).
static unsigned getBindingPort(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("loopschedule.binding"))
    return static_cast<unsigned>(attr.getInt());
  return 0;
}

/// Per-port read accessor for a `MemPortMapping`. Returns null pointers
/// when the requested port has not been allocated (used by merge logic
/// to default unset drives to zero).
struct PortDrivesView {
  const SmallVector<Value> *addrs = nullptr;
  Value wrData;
  Value wrEn;
  Value rdEn;
};

inline PortDrivesView portView(const MemPortMapping &mp, unsigned k) {
  PortDrivesView v;
  if (k == 0) {
    v.addrs = &mp.addrs;
    v.wrData = mp.wrData;
    v.wrEn = mp.wrEn;
    v.rdEn = mp.rdEn;
    return v;
  }
  if (mp.extraPorts.size() <= k - 1)
    return v;
  auto &p = mp.extraPorts[k - 1];
  v.addrs = &p.addrs;
  v.wrData = p.wrData;
  v.wrEn = p.wrEn;
  v.rdEn = p.rdEn;
  return v;
}

/// Pointer-bundle into a `MemPortMapping`'s drives for a specific port.
/// Port 0 aliases the direct fields; higher ports live in `extraPorts`.
/// Pointer members keep the bundle copy-/assignable.
struct PortDrivesRef {
  Value *rdData = nullptr;
  SmallVector<Value> *addrs = nullptr;
  Value *wrData = nullptr;
  Value *wrEn = nullptr;
  Value *rdEn = nullptr;
};

inline PortDrivesRef portRef(MemPortMapping &mp, unsigned k) {
  PortDrivesRef ref;
  if (k == 0) {
    ref.rdData = &mp.rdData;
    ref.addrs = &mp.addrs;
    ref.wrData = &mp.wrData;
    ref.wrEn = &mp.wrEn;
    ref.rdEn = &mp.rdEn;
    return ref;
  }
  while (mp.extraPorts.size() < k)
    mp.extraPorts.emplace_back();
  auto &p = mp.extraPorts[k - 1];
  ref.rdData = &p.rdData;
  ref.addrs = &p.addrs;
  ref.wrData = &p.wrData;
  ref.wrEn = &p.wrEn;
  ref.rdEn = &p.rdEn;
  return ref;
}

/// Build a PortArgInfo for a memref-backed value (function argument or
/// local alloc). Amc-port entries are built inline by the instance walk.
static PortArgInfo makePortArgInfoFromMemref(Value arg, MemRefType memType,
                                              bool isLocalMem) {
  PortArgInfo info;
  info.originalArg = arg;
  info.memType = memType;
  info.isLocalMem = isLocalMem;
  info.shape.assign(memType.getShape().begin(), memType.getShape().end());
  info.elementType = memType.getElementType();
  info.addrWidths = getDimAddrWidths(memType);
  info.numPorts = computeNumPortsFromUsers(arg);
  return info;
}

/// Append hw.module output port declarations for a port-arg (memref or amc
/// port). Memref entries emit per-dim addresses + wr_data + wr_en
/// unconditionally (matches the legacy always-maybe-writable shape). Amc
/// entries emit only the directional signals the port actually carries:
/// addresses always; rd_en output if `requiresRdEn`; wr_data/wr_en
/// outputs only if writable.
static void appendPortOutputPorts(OpBuilder &builder, StringRef baseName,
                                   const PortArgInfo &info,
                                   SmallVectorImpl<hw::PortInfo> &ports) {
  auto *ctx = builder.getContext();
  bool isOneDim = info.addrWidths.size() == 1;
  bool multi = info.numPorts > 1;
  for (unsigned port = 0; port < info.numPorts; ++port) {
    std::string portPrefix = multi ? (baseName.str() + "_p" +
                                       std::to_string(port))
                                   : baseName.str();
    for (auto [d, w] : llvm::enumerate(info.addrWidths)) {
      std::string addrName =
          isOneDim ? portPrefix + "_addr"
                   : portPrefix + "_addr_" + std::to_string(d);
      ports.push_back({{builder.getStringAttr(addrName),
                         IntegerType::get(ctx, w),
                         hw::ModulePort::Direction::Output}});
    }
    if (info.requiresRdEn) {
      ports.push_back({{builder.getStringAttr(portPrefix + "_rd_en"),
                         builder.getI1Type(),
                         hw::ModulePort::Direction::Output}});
    }
    bool emitWrite = info.isAmcPort ? info.isWrite : true;
    if (emitWrite) {
      ports.push_back({{builder.getStringAttr(portPrefix + "_wr_data"),
                         info.elementType,
                         hw::ModulePort::Direction::Output}});
      ports.push_back({{builder.getStringAttr(portPrefix + "_wr_en"),
                         builder.getI1Type(),
                         hw::ModulePort::Direction::Output}});
    }
  }
}

/// Append hw.output values for a port-arg's output ports (addr, optional
/// rd_en, wr_data, wr_en), falling back to safe defaults when the mapping
/// is absent or incomplete. For multi-port memories, emits one full
/// signal group per port in the same order as `appendPortOutputPorts`.
static void appendPortOutputValues(OpBuilder &builder, Location loc,
                                    const PortArgInfo &info, Value memKey,
                                    DenseMap<Value, MemPortMapping> &memPortMap,
                                    SmallVectorImpl<Value> &outputs) {
  auto *ctx = builder.getContext();
  Type i1 = builder.getI1Type();
  auto it = memPortMap.find(memKey);
  bool has = it != memPortMap.end();
  bool emitWrite = info.isAmcPort ? info.isWrite : true;
  for (unsigned port = 0; port < info.numPorts; ++port) {
    PortDrivesView pv = has ? portView(it->second, port) : PortDrivesView{};
    for (auto [d, w] : llvm::enumerate(info.addrWidths)) {
      Type addrType = IntegerType::get(ctx, w);
      if (pv.addrs && d < pv.addrs->size() && (*pv.addrs)[d])
        outputs.push_back((*pv.addrs)[d]);
      else
        outputs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
    }
    if (info.requiresRdEn) {
      if (pv.rdEn)
        outputs.push_back(pv.rdEn);
      else
        outputs.push_back(hw::ConstantOp::create(builder, loc, i1, 0));
    }
    if (emitWrite) {
      if (pv.wrData)
        outputs.push_back(pv.wrData);
      else
        outputs.push_back(
            hw::ConstantOp::create(builder, loc, info.elementType, 0));
      if (pv.wrEn)
        outputs.push_back(pv.wrEn);
      else
        outputs.push_back(hw::ConstantOp::create(builder, loc, i1, 0));
    }
  }
}

/// Represents one sequential loop in the nesting tree.
///
/// A frame can contain multiple launches (e.g. doitgen's q-frame runs an
/// accumulator pipeline at at-0 and a copyback pipeline at at-1). Each
/// launch gets its own wait slot that the FSM chains: the frame cannot
/// advance until every slot's child has reported done. Slots run in
/// at-offset order within the frame.
struct LoopNode {
  LoopScheduleSequentialOp seqOp;
  std::string prefix;              // e.g., "loop0", "loop0_loop1"
  std::vector<LoopNode> children;
  SmallVector<LoopSchedulePipelineOp> pipelineChildren;
  /// Per-frame list of launch slots, in at-offset order. Each slot
  /// carries either a sequential-child index (childIdx >= 0, pipIdx ==
  /// -1) or a pipeline-child index (pipIdx >= 0, childIdx == -1), plus
  /// the at-offset (cycle within the frame) at which its child_start
  /// pulse fires.
  struct LaunchSlot {
    int childIdx = -1;
    int pipIdx = -1;
    unsigned atOffset = 0;
  };
  SmallVector<SmallVector<LaunchSlot>> frameLaunches;
  bool isLeaf() const {
    return children.empty() && pipelineChildren.empty();
  }
  bool hasChild(unsigned frameIdx) const {
    return !frameLaunches[frameIdx].empty();
  }
  unsigned numLaunches(unsigned frameIdx) const {
    return frameLaunches[frameIdx].size();
  }
};

/// Compute the cycle latency of a defining op for pipelined hw lowering.
/// Prefers an explicit `loopschedule.cycle_latency` attribute, falls back
/// to the operator library's declared latency for ops tagged with
/// `loopschedule.operator = @...` (covers pipelined arith ops like
/// `i32_muli_l4` whose latency lives on the `oplib.operator` entry).
/// Returns 0 when no latency information is available.
///
/// Memory ops are excluded: their real latency is modeled by the FSM's
/// port state machine, not by the cross-stage delay-chain logic that
/// consumes this value. `OperatorAllocation` tags memory ops with
/// `loopschedule.operator` purely for the binding pass, and that operator's
/// declared latency is a hardware property, not a pipeline-stage count —
/// returning it here would insert a spurious extra register.
static unsigned
computeOpCycleLatency(Operation *def,
                      analysis::OperatorLibraryAnalysis *operatorLibrary) {
  if (!def)
    return 0;
  if (auto attr =
          def->getAttrOfType<IntegerAttr>("loopschedule.cycle_latency"))
    return (unsigned)attr.getInt();
  if (isa<LoopScheduleLoadOp, LoopScheduleStoreOp, LoadInterface,
          StoreInterface>(def))
    return 0;
  if (!operatorLibrary)
    return 0;
  auto operatorAttr =
      def->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
  if (!operatorAttr)
    return 0;
  StringRef opName = operatorLibrary->getOperatorBySymbol(operatorAttr);
  if (opName.empty())
    return 0;
  return operatorLibrary->getOperatorLatency(opName);
}

/// Materializes the cross-stage delay registers needed when a value
/// produced at stage J is consumed at stage K with K > J + L + 1
/// (where L is the producing op's cycle latency). Owns the per-value
/// `delayChain` and lazily extends each chain on first use.
///
/// Both the loop-pipeline child lowering and the function-pipeline
/// driver instantiate one of these and call `resolveForStage` whenever
/// they encounter an operand that crosses stage boundaries; the only
/// per-site differences are the register-name prefix and which
/// `OpBuilder` to wire ops onto.
class CrossStageValueResolver {
public:
  CrossStageValueResolver(OpBuilder &builder, Location loc, Block *hwBody,
                          Value clk, Value rst,
                          ArrayRef<LoopScheduleAtOp> stages,
                          ArrayRef<Value> stageCE, IRMapping &mapping,
                          analysis::OperatorLibraryAnalysis *operatorLibrary,
                          StringRef regNamePrefix)
      : builder(builder), loc(loc), hwBody(hwBody), clk(clk), rst(rst),
        stages(stages), stageCE(stageCE), mapping(mapping),
        operatorLibrary(operatorLibrary),
        regNamePrefix(regNamePrefix.str()) {}

  /// Returns the delayed copy of `origVal` to feed into `consumerStage`,
  /// or null if the consumer can read the existing stage register / op
  /// wrapper output directly. Lazily grows the per-value chain.
  Value resolveForStage(Value origVal, unsigned consumerStage) {
    auto result = dyn_cast<OpResult>(origVal);
    if (!result)
      return Value();
    auto defStage = dyn_cast<LoopScheduleAtOp>(result.getOwner());
    if (!defStage)
      return Value();
    unsigned J = stages.size();
    for (unsigned s = 0; s < stages.size(); ++s)
      if (stages[s] == defStage) {
        J = s;
        break;
      }
    // Effective "ready stage" for this value accounts for any internal
    // operator latency: the result is only logically valid at `J+L`.
    unsigned L = getYieldLatency(origVal);
    unsigned readyStage = J + L;
    // Only delay values whose consumer is strictly later than the
    // stage immediately after the ready point. Consumers at or one
    // past readyStage read the wrapper / stage register directly.
    if (J == stages.size() || consumerStage <= readyStage + 1)
      return Value();
    auto &chain = delayChain[origVal];
    if (chain.empty())
      chain.push_back(mapping.lookup(origVal));
    while (chain.size() < consumerStage - readyStage) {
      unsigned ceStage = readyStage + chain.size();
      if (ceStage >= stages.size())
        break;
      Value prev = chain.back();
      Value resetVal = createZeroConstant(builder, loc, prev.getType());
      auto regName = builder.getStringAttr(
          regNamePrefix + "_s" + std::to_string(ceStage) + "_dly_r" +
          std::to_string(result.getResultNumber()) + "_from_s" +
          std::to_string(J));
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(hwBody);
      Value reg = seq::CompRegClockEnabledOp::create(
          builder, loc, prev, clk, stageCE[ceStage], rst, resetVal, regName);
      chain.push_back(reg);
    }
    unsigned idx = consumerStage - readyStage - 1;
    if (idx >= chain.size())
      idx = chain.size() - 1;
    return chain[idx];
  }

private:
  /// Latency of the op that produces the at-op's yielded value `origVal`.
  unsigned getYieldLatency(Value origVal) const {
    auto result = dyn_cast<OpResult>(origVal);
    if (!result)
      return 0;
    auto atOp = dyn_cast<LoopScheduleAtOp>(result.getOwner());
    if (!atOp)
      return 0;
    auto yieldOp = atOp.getYieldOp();
    unsigned idx = result.getResultNumber();
    if (idx >= yieldOp.getNumOperands())
      return 0;
    return computeOpCycleLatency(yieldOp.getOperand(idx).getDefiningOp(),
                                  operatorLibrary);
  }

  OpBuilder &builder;
  Location loc;
  Block *hwBody;
  Value clk;
  Value rst;
  ArrayRef<LoopScheduleAtOp> stages;
  ArrayRef<Value> stageCE;
  IRMapping &mapping;
  analysis::OperatorLibraryAnalysis *operatorLibrary;
  std::string regNamePrefix;
  DenseMap<Value, SmallVector<Value>> delayChain;
};

/// Main pass converting LoopSchedule ops to FSM + HW.
class LoopScheduleToFSMPass
    : public circt::impl::LoopScheduleToFSMBase<LoopScheduleToFSMPass> {
public:
  void runOnOperation() override;

  /// Backedge info for a local hlmem (memref.alloc → seq.hlmem). Lifted
  /// to the top of the class so member-function declarations below can
  /// reference it by name.
  struct HLMemBackedges {
    Value allocResult;          // original memref.alloc result
    SmallVector<Backedge> addrBEs; // one per memref dim (in hlmem's widths)
    Backedge wrDataBE;
    Backedge wrEnBE;
  };

private:
  LogicalResult lowerFunction(loopschedule::LoopScheduleFuncSequentialOp funcOp);
  LogicalResult lowerFunction(loopschedule::LoopScheduleFuncPipelineOp funcOp);

  /// Shared between the function-sequential and function-pipeline drivers:
  /// scan the function body, clone non-skip ops via `mapping`, lower
  /// `memref.alloc` to `seq.hlmem` with backedges, lower each
  /// `HWMemoryInstanceLoweringInterface` op, and assemble the unified
  /// `memrefArgs` (function memref args + local allocs + amc ports).
  /// `isStageOrFrameOp` decides which top-level body ops are the
  /// caller's responsibility (LoopScheduleFrameOp/Sequential/Pipeline
  /// for sequential funcs, LoopScheduleAtOp for pipelined funcs) and
  /// should not be cloned by the prelude.
  LogicalResult setupFunctionPrelude(
      Block &body, ValueRange funcArgs, OpBuilder &builder, Location loc,
      Value clk, Value rst, ModuleOp enclosingModule,
      llvm::function_ref<bool(Operation *)> isStageOrFrameOp,
      IRMapping &mapping, BackedgeBuilder &funcBB,
      DenseMap<Value, MemPortMapping> &memPortMap,
      SmallVectorImpl<HLMemBackedges> &hlmemBEs,
      loopschedule::HWMemoryLoweringState &memInstState,
      SmallVectorImpl<PortArgInfo> &memrefArgs);

  /// Shared end-of-function wiring: resolve the address / write-data /
  /// write-enable backedges left dangling by `setupFunctionPrelude` for
  /// each local hlmem and amc.instance port. Unset ports tie to safe
  /// defaults (read-enable defaults high; addr / write-data /
  /// write-enable default to zero). Mutates the backedges in `hlmemBEs`
  /// and `memInstState`, so neither is taken by const reference.
  void resolveFunctionMemoryBackedges(
      OpBuilder &builder, Location loc, Block *hwBody,
      SmallVectorImpl<HLMemBackedges> &hlmemBEs,
      loopschedule::HWMemoryLoweringState &memInstState,
      DenseMap<Value, MemPortMapping> &memPortMap);

  /// Build the nesting tree from a sequential op.
  void buildLoopTree(LoopScheduleSequentialOp seqOp, LoopNode &node,
                     const std::string &prefix, unsigned &loopCounter);

  /// Create FSM machine for a sequential loop.
  ///
  /// `waitFrameIndices` lists the frame indices whose body launches an
  /// external computation that must be waited on (a child sequential loop, a
  /// pipeline child, or any future variable-latency op). It must be sorted
  /// ascending. Each such frame gets dedicated child_start/child_done/
  /// post_active signals and inserts WAIT_<i>/POST_<i> states around
  /// FRAME_<i>. Frames not in the list are "regular" frames that drive
  /// frame_active_<i> in their FRAME_<i> state.
  ///
  /// Inputs (fixed order):
  ///   start, cond, child_done_0..C-1            (C = waitFrameIndices.size())
  ///
  /// Outputs (fixed order):
  ///   done, first_iter, iter_advance,
  ///   frame_active_0..N-1,                      (N = numFrames)
  ///   child_start_0..C-1,
  ///   post_active_0..C-1
  fsm::MachineOp createSequentialFSM(OpBuilder &builder, Location loc,
                                     StringRef fsmName, unsigned numFrames,
                                     ArrayRef<unsigned> waitFrameIndices,
                                     ArrayRef<unsigned> launchAtOffsets,
                                     ArrayRef<unsigned> frameLatencies);

  /// Create linear run-once FSM for sequencing top-level function frames.
  /// frameChildKind[i]: -1 = leaf, 0 = sequential child, 1 = pipeline child.
  fsm::MachineOp createFunctionFSM(OpBuilder &builder, Location loc,
                                    StringRef fsmName,
                                    ArrayRef<int> frameChildKind);

  /// Recursively lower a loop node as its own hw.module.
  /// Creates the module and populates outModule. capturedVals are the
  /// external values the caller must wire as inputs when instantiating.
  LogicalResult lowerLoopNodeAsModule(
      const LoopNode &node, OpBuilder &builder, Location loc,
      loopschedule::LoopScheduleFuncSequentialOp funcOp, ArrayRef<PortArgInfo> memrefArgs,
      IRMapping &parentMapping,
      hw::HWModuleOp &outModule,
      SmallVectorImpl<Value> &capturedVals);

  /// Lower a frame body, gating store wrEn with the per-issue-cycle active
  /// signal. `cycleGates[c]` is the i1 high in cycle `c` of the enclosing
  /// frame. For single-cycle frames the array has one entry equal to the
  /// frame's overall active signal.
  LogicalResult lowerFrameBody(Block *frameBody, OpBuilder &builder,
                               IRMapping &mapping,
                               ArrayRef<Value> cycleGates,
                               DenseMap<Value, MemPortMapping> &memPorts,
                               ModuleOp moduleOp, Value clk, Value rst);

  /// Materialize a single compute op via the operator-library dispatch.
  /// Used by every internal cloning site that previously called
  /// `builder.clone(op, mapping)` for arith ops. `op` is the original op
  /// (not yet cloned). On success, populates `mapping` for the original
  /// op's results.
  LogicalResult emitComputeOp(Operation *op, OpBuilder &builder,
                              IRMapping &mapping, ModuleOp moduleOp,
                              Value clk, Value rst);

  /// `lowerAtBody` is a method (not a free function) so it can access
  /// `operatorLibrary` and `instanceUniquer` directly.
  LogicalResult lowerAtBody(Block *body, OpBuilder &builder,
                            IRMapping &mapping, ArrayRef<Value> cycleGates,
                            unsigned baseCycle,
                            DenseMap<Value, MemPortMapping> &memPorts,
                            ModuleOp moduleOp, Value clk, Value rst);

  /// Lower a pipeline as a child of a sequential loop (no FSM needed).
  LogicalResult lowerPipelineChild(LoopSchedulePipelineOp pipOp,
                                   OpBuilder &builder, Location loc,
                                   Block *hwBody, IRMapping &mapping,
                                   Value clk, Value rst, Value startSignal,
                                   StringRef namePrefix, Value &doneSignal,
                                   DenseMap<Value, MemPortMapping> &memPorts,
                                   ArrayRef<PortArgInfo> memrefArgs);

  /// Map from original func memref args to their hw.module port values.
  DenseMap<Value, MemPortMapping> memPortMap;

  /// Per-function operator library analysis. Bound for the duration of
  /// `lowerFunction`. Compute ops in leaf bodies are required to carry a
  /// `loopschedule.operator` attribute and are materialized by consulting
  /// this analysis. A raw pointer is used so the pass class stays
  /// copyable, as MLIR's pass manager expects.
  analysis::OperatorLibraryAnalysis *operatorLibrary = nullptr;

  /// Counter map keyed by operator name; used to uniquify `hw.instance`
  /// sym_names emitted by the operator-library dispatch helpers. Reset per
  /// `hw.module` we generate.
  llvm::StringMap<unsigned> instanceUniquer;
};

//===----------------------------------------------------------------------===//
// Load/store helpers
//===----------------------------------------------------------------------===//

/// Lower any `HWLoadLoweringInterface` op: drive the read addresses on
/// the memory port mapping (port selected by `loopschedule.binding`,
/// default 0) and map the load result to the port's read data.
static LogicalResult
handleHWLoad(loopschedule::HWLoadLoweringInterface loadOp, OpBuilder &builder,
             IRMapping &mapping,
             DenseMap<Value, MemPortMapping> &memPorts,
             Value rdEnGate = nullptr) {
  SmallVector<unsigned> widths = loadOp.getAddrWidths();
  unsigned port = getBindingPort(loadOp);
  auto it = memPorts.find(loadOp.getMemoryValue());
  if (it == memPorts.end())
    return loadOp->emitError("unmapped memory");
  PortDrivesRef pref = portRef(it->second, port);
  if (loadOp.getIndices().size() != widths.size())
    return loadOp->emitError("memory access index count (")
           << loadOp.getIndices().size() << ") does not match memory rank ("
           << widths.size() << ")";
  pref.addrs->resize(widths.size());
  for (auto [d, idx] : llvm::enumerate(loadOp.getIndices())) {
    Value addr = mapping.lookup(idx);
    (*pref.addrs)[d] =
        resizeIntTo(builder, loadOp->getLoc(), addr, widths[d]);
  }
  if (loadOp.requiresReadEnable()) {
    assert(rdEnGate && "HW load requires an explicit read-enable gate");
    *pref.rdEn = rdEnGate;
  }
  mapping.map(loadOp.getResult(), *pref.rdData);
  return success();
}

/// Lower any `HWStoreLoweringInterface` op: drive the addresses, write
/// data, and write enable (gated by `wrEnGate`) on the bound port.
static LogicalResult
handleHWStore(loopschedule::HWStoreLoweringInterface storeOp,
               OpBuilder &builder, IRMapping &mapping, Value wrEnGate,
               DenseMap<Value, MemPortMapping> &memPorts) {
  assert(wrEnGate && "handleHWStore requires a wrEnGate");
  SmallVector<unsigned> widths = storeOp.getAddrWidths();
  unsigned port = getBindingPort(storeOp);
  auto it = memPorts.find(storeOp.getMemoryValue());
  if (it == memPorts.end())
    return storeOp->emitError("unmapped memory");
  PortDrivesRef pref = portRef(it->second, port);
  if (storeOp.getIndices().size() != widths.size())
    return storeOp->emitError("memory access index count (")
           << storeOp.getIndices().size() << ") does not match memory rank ("
           << widths.size() << ")";
  pref.addrs->resize(widths.size());
  for (auto [d, idx] : llvm::enumerate(storeOp.getIndices())) {
    Value addr = mapping.lookup(idx);
    (*pref.addrs)[d] =
        resizeIntTo(builder, storeOp->getLoc(), addr, widths[d]);
  }
  *pref.wrData = mapping.lookup(storeOp.getValueToStore());
  *pref.wrEn = wrEnGate;
  return success();
}

//===----------------------------------------------------------------------===//
// Operator-library dispatch helpers
//===----------------------------------------------------------------------===//

/// Ops that bypass the operator-library requirement: constants,
/// extension/truncation/index_cast, plus any arith op operating purely on
/// MLIR `index` type. Loop-control increments and bounds checks are
/// `index`-typed and represent loop bookkeeping rather than the kernel
/// computation; they're left to subsequent legalization (e.g. arith
/// expansion / map-arith-to-comb) instead of being routed through the
/// operator library.
static bool isFreeArithOp(Operation *op) {
  if (isa<arith::ConstantOp, arith::IndexCastOp, arith::ExtSIOp,
          arith::ExtUIOp, arith::TruncIOp>(op))
    return true;
  // Arith ops whose result and all operands are `index` are loop-control
  // ops — treat as free.
  if (op->getDialect() &&
      op->getDialect()->getNamespace() == "arith") {
    auto isIndex = [](Type t) { return isa<IndexType>(t); };
    bool allIndex = llvm::all_of(op->getResultTypes(), isIndex) &&
                    llvm::all_of(op->getOperandTypes(), isIndex);
    // cmpi has an i1 result; allow it if the operand types are index.
    if (auto cmp = dyn_cast<arith::CmpIOp>(op))
      allIndex = isIndex(cmp.getLhs().getType()) &&
                 isIndex(cmp.getRhs().getType());
    if (allIndex)
      return true;
  }
  return false;
}

/// Clone the real comb op living inside `origOp`'s hw_match body.
/// Operands are remapped: the body's block args correspond positionally
/// to the target function's inputs, i.e. `origOp->getOperand(i)`.
static LogicalResult
emitCombOpFromOperator(Operation *origOp, OpBuilder &builder,
                       IRMapping &mapping, oplib::HwMatchOp hwMatch) {
  Block *body = hwMatch.getBodyBlock();
  Operation *templateOp = nullptr;
  for (auto &op : *body) {
    if (isa<oplib::HwReturnOp>(op))
      continue;
    templateOp = &op;
    break;
  }
  if (!templateOp)
    return origOp->emitOpError(
        "comb-op hw_match body is missing its template op");

  // Map body block args -> origOp operands.
  IRMapping bodyMap;
  for (auto [arg, operand] :
       llvm::zip(body->getArguments(), origOp->getOperands())) {
    bodyMap.map(arg, mapping.lookup(operand));
  }
  Operation *newOp = builder.clone(*templateOp, bodyMap);
  for (auto [oldRes, newRes] :
       llvm::zip(origOp->getResults(), newOp->getResults()))
    mapping.map(oldRes, newRes);
  return success();
}

/// Materialize an `hw.instance` of the operator's extern for `origOp` by
/// cloning the `oplib.hw_instance` op inside `templateInst` and
/// substituting its operands. Clock / reset are wired from the enclosing
/// `hw.module`'s `clk` / `rst`. The clock-enable port (if present) is
/// tied to a constant `1 : i1`. Per-operand / per-result routing uses
/// the extern's `oplib.clock` / `oplib.reset` / `oplib.enable` /
/// `oplib.operand = N` / `oplib.result = N` per-port attrs stamped by
/// `OperatorLibraryLoader::makePortAttrDict`.
static LogicalResult
emitHwInstanceFromOperator(Operation *origOp, OpBuilder &builder,
                           IRMapping &mapping, ModuleOp moduleOp,
                           oplib::HwInstanceOp templateInst, Value clk,
                           Value rst, llvm::StringMap<unsigned> &uniquer,
                           StringRef opName) {
  auto externOp = moduleOp.lookupSymbol<hw::HWModuleExternOp>(
      templateInst.getModuleNameAttr().getValue());
  if (!externOp)
    return origOp->emitOpError("operator '")
           << opName << "' references unknown extern module @"
           << templateInst.getModuleName();

  auto inputAttrs = externOp.getAllInputAttrs();
  auto outputAttrs = externOp.getAllOutputAttrs();

  Value oneI1 = hw::ConstantOp::create(builder, origOp->getLoc(),
                                       builder.getI1Type(), (int64_t)1);
  Value clkValue = clk;

  SmallVector<Value> instanceOperands;
  unsigned inIdx = 0;
  for (auto port : externOp.getPortList()) {
    if (port.dir != hw::ModulePort::Direction::Input)
      continue;
    DictionaryAttr portAttrs =
        inIdx < inputAttrs.size()
            ? dyn_cast_or_null<DictionaryAttr>(inputAttrs[inIdx])
            : DictionaryAttr();
    Value driver;
    if (portAttrs && portAttrs.get("oplib.clock")) {
      if (clkValue.getType() != port.type &&
          isa<seq::ClockType>(clkValue.getType()))
        clkValue = seq::FromClockOp::create(builder, origOp->getLoc(), clk);
      driver = clkValue;
    } else if (portAttrs && portAttrs.get("oplib.reset")) {
      driver = rst;
    } else if (portAttrs && portAttrs.get("oplib.enable")) {
      driver = oneI1;
    } else if (portAttrs) {
      if (auto opIdxAttr =
              dyn_cast_or_null<IntegerAttr>(portAttrs.get("oplib.operand"))) {
        unsigned j = opIdxAttr.getInt();
        if (j >= origOp->getNumOperands())
          return origOp->emitOpError("operator '")
                 << opName << "' extern port operand index " << j
                 << " exceeds op operand count";
        driver = mapping.lookup(origOp->getOperand(j));
      }
    }
    if (!driver)
      return origOp->emitOpError("operator '")
             << opName << "' extern input port #" << inIdx
             << " is missing an oplib role attr";
    instanceOperands.push_back(driver);
    ++inIdx;
  }

  // Uniquify the instance name across the enclosing hw.module.
  unsigned tag = uniquer[opName]++;
  std::string instanceName =
      (templateInst.getInstanceName() + "_" + std::to_string(tag)).str();

  auto instOp = hw::InstanceOp::create(
      builder, origOp->getLoc(), externOp,
      builder.getStringAttr(instanceName), instanceOperands);

  // Map each op-result to the corresponding instance output via the
  // extern's `oplib.result = N` per-port attr.
  unsigned outIdx = 0;
  for (auto port : externOp.getPortList()) {
    if (port.dir != hw::ModulePort::Direction::Output)
      continue;
    DictionaryAttr portAttrs =
        outIdx < outputAttrs.size()
            ? dyn_cast_or_null<DictionaryAttr>(outputAttrs[outIdx])
            : DictionaryAttr();
    if (portAttrs) {
      if (auto resIdxAttr =
              dyn_cast_or_null<IntegerAttr>(portAttrs.get("oplib.result"))) {
        unsigned j = resIdxAttr.getInt();
        if (j < origOp->getNumResults())
          mapping.map(origOp->getResult(j), instOp.getResult(outIdx));
      }
    }
    ++outIdx;
  }
  return success();
}

/// Dispatch a single compute op to either a comb-op materialization or an
/// hw.instance materialization based on the operator's hw_match.
static LogicalResult
emitOpFromOperatorLibrary(Operation *origOp, OpBuilder &builder,
                          IRMapping &mapping, ModuleOp moduleOp, Value clk,
                          Value rst, llvm::StringMap<unsigned> &uniquer,
                          analysis::OperatorLibraryAnalysis &ola) {
  auto operatorAttr =
      origOp->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
  if (!operatorAttr)
    return origOp->emitOpError(
        "LoopScheduleToFSM: missing loopschedule.operator attribute; "
        "operator-allocation must run before this pass");

  StringRef opName = ola.getOperatorBySymbol(operatorAttr);
  oplib::HwMatchOp hwMatch = ola.getHwMatchOp(opName);
  if (!hwMatch)
    return origOp->emitOpError("operator '")
           << opName << "' has no hw_match in the operator library";

  // Dispatch by inspecting the hw_match body: an `oplib.hw_instance`
  // names the referenced extern; anything else (comb op, etc.) is
  // cloned as-is via `emitCombOpFromOperator`.
  for (auto &op : *hwMatch.getBodyBlock()) {
    if (auto inst = dyn_cast<oplib::HwInstanceOp>(op))
      return emitHwInstanceFromOperator(origOp, builder, mapping, moduleOp,
                                        inst, clk, rst, uniquer, opName);
  }
  return emitCombOpFromOperator(origOp, builder, mapping, hwMatch);
}

//===----------------------------------------------------------------------===//
// Frame body lowering
//===----------------------------------------------------------------------===//

/// Lower ops inside an `at` body at the given `baseCycle` (the at's offset
/// within the enclosing frame). Stores are gated on `cycleGates[baseCycle]`;
/// loads drive their addresses combinationally.
LogicalResult LoopScheduleToFSMPass::lowerAtBody(
    Block *body, OpBuilder &builder, IRMapping &mapping,
    ArrayRef<Value> cycleGates, unsigned baseCycle,
    DenseMap<Value, MemPortMapping> &memPorts, ModuleOp moduleOp, Value clk,
    Value rst) {
  auto pickGate = [&](unsigned c) -> Value {
    assert(c < cycleGates.size() &&
           "issue cycle exceeds enclosing frame latency");
    return cycleGates[c];
  };
  // Recursive body-processor: handles ops including nested
  // `loopschedule.if` by AND'ing the condition into the store/load
  // gates for body ops. See the pipeline-stage counterpart in
  // lowerPipelineChild for the same pattern.
  std::function<LogicalResult(Operation *, Value)> processOp =
      [&](Operation *inner, Value gate) -> LogicalResult {
    if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp>(inner))
      return success();
    if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp,
            LoopScheduleLaunchOp>(inner))
      return success();
    if (auto ifOp = dyn_cast<LoopScheduleIfOp>(inner)) {
      Value cond = mapping.lookup(ifOp.getCond());
      Value innerGate = comb::AndOp::create(builder, inner->getLoc(),
                                              gate, cond);
      for (auto &nested : ifOp.getBody().front()) {
        if (auto yieldOp = dyn_cast<LoopScheduleYieldOp>(&nested)) {
          for (auto [res, val] :
               llvm::zip(ifOp.getResults(), yieldOp.getOperands()))
            mapping.map(res, mapping.lookup(val));
          continue;
        }
        if (failed(processOp(&nested, innerGate)))
          return failure();
      }
      return success();
    }
    if (auto storeOp = dyn_cast<HWStoreLoweringInterface>(inner))
      return handleHWStore(storeOp, builder, mapping, gate, memPorts);
    if (auto loadOp = dyn_cast<HWLoadLoweringInterface>(inner))
      return handleHWLoad(loadOp, builder, mapping, memPorts, gate);
    return emitComputeOp(inner, builder, mapping, moduleOp, clk, rst);
  };
  for (auto &op : *body) {
    if (failed(processOp(&op, pickGate(baseCycle))))
      return failure();
  }
  return success();
}

LogicalResult LoopScheduleToFSMPass::emitComputeOp(
    Operation *op, OpBuilder &builder, IRMapping &mapping, ModuleOp moduleOp,
    Value clk, Value rst) {
  // Constants and free-pass casts (extsi/extui/trunci/index_cast) are not
  // first-class operator-library entries. They get cloned through the
  // mapping so downstream consumers see them.
  if (isFreeArithOp(op)) {
    builder.clone(*op, mapping);
    return success();
  }

  // Without a library or without a `loopschedule.operator` attribute,
  // fall back to cloning so passes that synthesize fresh arith ops after
  // operator-allocation (e.g. address-calc muli emitted by
  // memref-to-loopschedule on narrow widths) still flow through. The
  // cloned op stays as arith.* in the output IR; downstream lowerings
  // (map-arith-to-comb etc.) handle the legalization.
  auto operatorAttr =
      op->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
  if (!operatorLibrary || !operatorAttr) {
    builder.clone(*op, mapping);
    return success();
  }
  return emitOpFromOperatorLibrary(op, builder, mapping, moduleOp, clk, rst,
                                    instanceUniquer, *operatorLibrary);
}

LogicalResult LoopScheduleToFSMPass::lowerFrameBody(
    Block *frameBody, OpBuilder &builder, IRMapping &mapping,
    ArrayRef<Value> cycleGates,
    DenseMap<Value, MemPortMapping> &memPorts, ModuleOp moduleOp,
    Value clk, Value rst) {
  for (auto &op : *frameBody) {
    if (isa<LoopScheduleYieldOp>(&op))
      continue;
    if (auto atOp = dyn_cast<LoopScheduleAtOp>(&op)) {
      unsigned offset = (unsigned)atOp.getOffset();
      if (failed(lowerAtBody(&atOp.getBodyBlock(), builder, mapping, cycleGates,
                             offset, memPorts, moduleOp, clk, rst)))
        return failure();
      // Forward at results to the caller's mapping via the at's yield.
      auto yieldOp = atOp.getYieldOp();
      for (auto [res, val] :
           llvm::zip(atOp.getResults(), yieldOp.getOperands()))
        mapping.map(res, mapping.lookup(val));
      continue;
    }
    // Frames should only contain `at` children and the terminator yield.
    return op.emitOpError(
        "LoopScheduleToFSM: unexpected op in frame body; expected only "
        "loopschedule.at and loopschedule.yield");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Loop tree construction
//===----------------------------------------------------------------------===//

void LoopScheduleToFSMPass::buildLoopTree(
    LoopScheduleSequentialOp seqOp, LoopNode &node,
    const std::string &prefix, unsigned &loopCounter) {
  node.seqOp = seqOp;
  node.prefix = prefix;

  SmallVector<LoopScheduleFrameOp> frames;
  for (auto &op : seqOp.getScheduleBlock().getOperations())
    if (auto frameOp = dyn_cast<LoopScheduleFrameOp>(&op))
      frames.push_back(frameOp);

  node.frameLaunches.resize(frames.size());

  // Nested LoopInterface ops (seq/pipeline) are wrapped in
  // `loopschedule.launch` siblings of `at` ops inside the frame body.
  // Each launch contains exactly one child LoopInterface plus a yield.
  // Launches at different at-offsets run concurrently: each launch's
  // child_start pulse fires at FRAME_i_<atOffset>, and the frame's
  // single WAIT state waits for the AND of all latched child_dones.
  // Multiple launches at the same at-offset also run concurrently,
  // firing their start pulses in the same cycle.
  for (auto [frameIdx, frameOp] : llvm::enumerate(frames)) {
    auto launches = loopschedule::getLaunchOpsInOrder(frameOp);
    for (auto launch : launches) {
      Operation *child = nullptr;
      for (auto &op : launch.getBodyBlock().getOperations()) {
        if (isa<LoopScheduleYieldOp>(op))
          continue;
        child = &op;
        break;
      }
      if (!child)
        continue;
      LoopNode::LaunchSlot slot;
      if (auto childSeq = dyn_cast<LoopScheduleSequentialOp>(child)) {
        slot.childIdx = (int)node.children.size();
        node.children.emplace_back();
        std::string childPrefix =
            prefix + "_loop" + std::to_string(loopCounter++);
        buildLoopTree(childSeq, node.children.back(), childPrefix,
                       loopCounter);
      } else if (auto childPip = dyn_cast<LoopSchedulePipelineOp>(child)) {
        slot.pipIdx = (int)node.pipelineChildren.size();
        node.pipelineChildren.push_back(childPip);
      } else {
        continue;
      }
      if (auto launchAt = launch->getParentOfType<LoopScheduleAtOp>())
        slot.atOffset = (unsigned)launchAt.getOffset();
      node.frameLaunches[frameIdx].push_back(slot);
    }
  }
}

//===----------------------------------------------------------------------===//
// Captured value analysis
//===----------------------------------------------------------------------===//

/// Collect non-memref, non-constant values used inside seqOp but defined
/// outside it. These must become input ports on the child module.
static SmallVector<Value>
collectCapturedValues(LoopScheduleSequentialOp seqOp) {
  SetVector<Value> captured;
  auto processOperand = [&](Value operand) {
    if (isa<MemRefType>(operand.getType()))
      return;
    // Amc port-like values (results of an `amc.instance` or downstream port
    // composition ops) are threaded via `memrefArgs`, just like memrefs —
    // skip them so they don't also try to come in as plain scalar captures
    // (which would fail to resolve in the caller's `mapping`).
    if (isa<loopschedule::HandleType>(operand.getType()))
      return;
    if (auto *defOp = operand.getDefiningOp())
      if (defOp->getDialect() &&
          defOp->getDialect()->getNamespace() == "amc")
        return;
    // Skip values defined inside the seqOp.
    if (operand.getParentRegion() &&
        seqOp->isAncestor(operand.getParentRegion()->getParentOp()))
      return;
    // Skip constants — they will be cloned into the child module.
    if (auto *defOp = operand.getDefiningOp())
      if (defOp->hasTrait<OpTrait::ConstantLike>())
        return;
    captured.insert(operand);
  };
  // Process seqOp's own operands (init values).
  for (Value operand : seqOp->getOperands())
    processOperand(operand);
  // Walk ops inside the seqOp's regions.
  for (auto &region : seqOp->getRegions()) {
    region.walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        processOperand(operand);
    });
  }
  return captured.takeVector();
}

/// Collect all constant-like ops referenced inside seqOp but defined outside.
static SmallVector<Operation *>
collectReferencedConstants(LoopScheduleSequentialOp seqOp) {
  SetVector<Operation *> constants;
  auto processOperand = [&](Value operand) {
    if (auto *defOp = operand.getDefiningOp()) {
      if (defOp->hasTrait<OpTrait::ConstantLike>() &&
          !seqOp->isAncestor(defOp))
        constants.insert(defOp);
    }
  };
  // Process seqOp's own operands (init values).
  for (Value operand : seqOp->getOperands())
    processOperand(operand);
  // Walk ops inside the seqOp's regions.
  for (auto &region : seqOp->getRegions()) {
    region.walk([&](Operation *op) {
      for (Value operand : op->getOperands())
        processOperand(operand);
    });
  }
  return constants.takeVector();
}

//===----------------------------------------------------------------------===//
// FSM machine creation
//===----------------------------------------------------------------------===//

fsm::MachineOp LoopScheduleToFSMPass::createSequentialFSM(
    OpBuilder &builder, Location loc, StringRef fsmName, unsigned numFrames,
    ArrayRef<unsigned> waitFrameIndices,
    ArrayRef<unsigned> launchAtOffsets,
    ArrayRef<unsigned> frameLatencies) {
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();

  // Per-frame list of global wait-slot indices. A frame can hold multiple
  // launches; `waitFrameIndices` has one entry per launch and multiple
  // entries for the same frame index represent launches that run
  // concurrently at their respective `launchAtOffsets[j]` cycle.
  SmallVector<SmallVector<int>> frameWaitIdx(numFrames);
  for (auto [j, i] : llvm::enumerate(waitFrameIndices)) {
    assert(i < numFrames && "wait frame index out of range");
    frameWaitIdx[i].push_back((int)j);
  }
  // Verify non-descending frame order; within a frame, launches may
  // appear at arbitrary at-offsets (ordering is by at-offset inside the
  // per-frame state chain).
  for (unsigned k = 1; k < waitFrameIndices.size(); ++k)
    assert(waitFrameIndices[k - 1] <= waitFrameIndices[k] &&
           "waitFrameIndices must be sorted non-descending");
  assert(launchAtOffsets.size() == waitFrameIndices.size() &&
         "launchAtOffsets must have one entry per launch");

  unsigned numWaits = waitFrameIndices.size();

  // Normalize frameLatencies — default 1 per frame. Extend each
  // wait-frame's latency so it covers the latest launch's at-offset
  // (child_start pulses at FRAME_i_<atOffset>, which must exist).
  SmallVector<unsigned> frameLats(numFrames, 1);
  for (unsigned i = 0; i < numFrames && i < frameLatencies.size(); ++i)
    frameLats[i] = std::max(frameLatencies[i], 1u);
  for (unsigned i = 0; i < numFrames; ++i) {
    for (int j : frameWaitIdx[i])
      frameLats[i] = std::max(frameLats[i],
                              launchAtOffsets[(unsigned)j] + 1);
  }

  // Compute the per-frame base index in the "frame_cycle" output region.
  // Frames with latency > 1 contribute L_i outputs; single-cycle frames
  // contribute 0 (their per-cycle gate is the frame_active_i output).
  SmallVector<int> cycleOutBase(numFrames, -1);
  unsigned totalCycleOuts = 0;
  for (unsigned i = 0; i < numFrames; ++i) {
    if (frameLats[i] > 1) {
      cycleOutBase[i] = (int)totalCycleOuts;
      totalCycleOuts += frameLats[i];
    }
  }

  // Inputs: start, cond, child_done_0..C-1.
  SmallVector<Type> inputTypes;
  inputTypes.push_back(i1); // start
  inputTypes.push_back(i1); // cond
  for (unsigned j = 0; j < numWaits; ++j)
    inputTypes.push_back(i1); // child_done_j

  // Outputs: done, first_iter, iter_advance,
  //          frame_active_0..N-1,
  //          child_start_0..C-1, child_active_0..C-1, post_active_0..C-1,
  //          frame_cycle_<i>_<c>... (one per multi-cycle frame's sub-cycle).
  // child_start_j  : 1-cycle pulse in FRAME_<wait_frame_j>; drives child start.
  // child_active_j : high while child j is producing on the wires
  //                  (in FRAME_<wait_frame_j> launch AND throughout WAIT_<...>).
  //                  Used to mux child j's mem ports out of the parent module.
  // post_active_j  : high in POST_<wait_frame_j> (parent's post-child phase).
  // frame_cycle_<i>_<c> : high in the c-th sub-state of FRAME_<i>; only
  //                       emitted for frames with latency > 1.
  unsigned numOutputs = 3 + numFrames + 3 * numWaits + totalCycleOuts;
  SmallVector<Type> outTypes(numOutputs, i1);

  auto funcType = FunctionType::get(ctx, inputTypes, outTypes);
  auto machine =
      fsm::MachineOp::create(builder, loc, fsmName, "IDLE", funcType);

  SmallVector<Attribute> argNames;
  argNames.push_back(builder.getStringAttr("start"));
  argNames.push_back(builder.getStringAttr("cond"));
  for (unsigned j = 0; j < numWaits; ++j)
    argNames.push_back(
        builder.getStringAttr("child_done_" + std::to_string(j)));
  machine.setArgNamesAttr(builder.getArrayAttr(argNames));

  SmallVector<Attribute> resNames;
  resNames.push_back(builder.getStringAttr("done"));
  resNames.push_back(builder.getStringAttr("first_iter"));
  resNames.push_back(builder.getStringAttr("iter_advance"));
  for (unsigned i = 0; i < numFrames; ++i)
    resNames.push_back(
        builder.getStringAttr("frame_active_" + std::to_string(i)));
  for (unsigned j = 0; j < numWaits; ++j)
    resNames.push_back(
        builder.getStringAttr("child_start_" + std::to_string(j)));
  for (unsigned j = 0; j < numWaits; ++j)
    resNames.push_back(
        builder.getStringAttr("child_active_" + std::to_string(j)));
  for (unsigned j = 0; j < numWaits; ++j)
    resNames.push_back(
        builder.getStringAttr("post_active_" + std::to_string(j)));
  for (unsigned i = 0; i < numFrames; ++i) {
    if (cycleOutBase[i] < 0)
      continue;
    for (unsigned c = 0; c < frameLats[i]; ++c)
      resNames.push_back(builder.getStringAttr(
          "frame_cycle_" + std::to_string(i) + "_" + std::to_string(c)));
  }
  machine.setResNamesAttr(builder.getArrayAttr(resNames));

  OpBuilder fb(ctx);
  fb.setInsertionPointToEnd(&machine.getBody().front());

  Value trueVal = hw::ConstantOp::create(fb, loc, i1, 1);
  Value falseVal = hw::ConstantOp::create(fb, loc, i1, 0);
  auto fiVar = fsm::VariableOp::create(
      fb, loc, i1, fb.getBoolAttr(true), "first_iter");

  // Helper: build output vector. Each of the launch-indexed argument lists
  // (child_start, child_active, post_active) selects which global launch
  // indices should be driven high in this state.
  //   activeFrame         : index of frame_active to drive high (-1 = none)
  //   activeChildStarts   : launch indices whose child_start pulses
  //   activeLive          : launch indices whose child_active is high
  //   activePosts         : launch indices whose post_active is high
  //   cycleFrame, cycleIdx: if cycleFrame >= 0 drives
  //                         frame_cycle_<cycleFrame>_<cycleIdx>
  auto buildOutSets = [&](Value done, bool iterAdv, int activeFrame,
                          ArrayRef<unsigned> activeChildStarts,
                          ArrayRef<unsigned> activeLive,
                          ArrayRef<unsigned> activePosts,
                          int cycleFrame = -1,
                          int cycleIdx = -1) -> SmallVector<Value> {
    SmallVector<Value> v;
    v.push_back(done);
    v.push_back(fiVar);
    v.push_back(iterAdv ? trueVal : falseVal);
    for (unsigned i = 0; i < numFrames; ++i)
      v.push_back((int)i == activeFrame ? trueVal : falseVal);
    for (unsigned j = 0; j < numWaits; ++j)
      v.push_back(llvm::is_contained(activeChildStarts, j) ? trueVal
                                                           : falseVal);
    for (unsigned j = 0; j < numWaits; ++j)
      v.push_back(llvm::is_contained(activeLive, j) ? trueVal : falseVal);
    for (unsigned j = 0; j < numWaits; ++j)
      v.push_back(llvm::is_contained(activePosts, j) ? trueVal : falseVal);
    for (unsigned i = 0; i < numFrames; ++i) {
      if (cycleOutBase[i] < 0)
        continue;
      for (unsigned c = 0; c < frameLats[i]; ++c) {
        bool on = ((int)i == cycleFrame) && ((int)c == cycleIdx);
        v.push_back(on ? trueVal : falseVal);
      }
    }
    return v;
  };
  // Convenience single-index wrapper (negative = empty).
  auto buildOut = [&](Value done, bool iterAdv, int activeFrame,
                      int activeChild, int activeLive,
                      int activePost, int cycleFrame = -1,
                      int cycleIdx = -1) -> SmallVector<Value> {
    SmallVector<unsigned> starts, lives, posts;
    if (activeChild >= 0) starts.push_back((unsigned)activeChild);
    if (activeLive >= 0)  lives.push_back((unsigned)activeLive);
    if (activePost >= 0)  posts.push_back((unsigned)activePost);
    return buildOutSets(done, iterAdv, activeFrame, starts, lives, posts,
                        cycleFrame, cycleIdx);
  };

  // State-name helpers. Every frame has L_i cycle states plus (if it
  // has launches) a single shared WAIT_i and POST_i. Single-cycle
  // frames name their sole cycle state FRAME_i (no suffix); multi-cycle
  // frames use FRAME_i_<c>.
  auto frameStateName = [&](unsigned i, unsigned c) -> std::string {
    if (frameLats[i] <= 1)
      return "FRAME_" + std::to_string(i);
    return "FRAME_" + std::to_string(i) + "_" + std::to_string(c);
  };
  auto waitStateName = [&](unsigned i) -> std::string {
    return "WAIT_" + std::to_string(i);
  };
  auto postStateName = [&](unsigned i) -> std::string {
    return "POST_" + std::to_string(i);
  };

  // --- IDLE ---
  {
    auto st = fsm::StateOp::create(fb, loc, "IDLE");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    fsm::OutputOp::create(fb, loc, buildOut(falseVal, false, -1, -1, -1, -1));
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(
        fb, loc, StringRef("COND"),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
        [&]() { fsm::UpdateOp::create(fb, loc, fiVar, trueVal); });
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // --- COND ---
  {
    auto st = fsm::StateOp::create(fb, loc, "COND");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    fsm::OutputOp::create(fb, loc, buildOut(falseVal, false, -1, -1, -1, -1));
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(
        fb, loc, StringRef(frameStateName(0, 0)),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(1)); },
        [&]() { fsm::UpdateOp::create(fb, loc, fiVar, falseVal); });
    fsm::TransitionOp::create(fb, loc, StringRef("DONE"));
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // Helper: emit the last frame's exit transition. We go back to COND so the
  // condition is evaluated with the UPDATED iter_args (the iter_arg register
  // latches on the clock edge leaving the last frame, which asserts
  // iter_advance). Checking `cond` directly at the last frame instead would
  // race: the register hasn't updated yet, so cond still reflects the OLD
  // iter_args — producing one extra spurious iteration.
  auto emitLastFrameTransition = [&](Block *tb) {
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(fb, loc, StringRef("COND"));
  };

  // --- FRAME_i cycle states (+ WAIT_i/POST_i for frames with launches) ---
  //
  // Every frame emits L_i sequential states FRAME_i_0 → FRAME_i_1 →
  // ... → FRAME_i_{L-1}. At cycle c:
  //   - any launch j with atOffset == c pulses child_start_j
  //   - any launch j with atOffset <= c has child_active_j high
  //   - static at-K ops use frame_cycle_i_K as their gate
  //
  // If the frame has launches, the last cycle state transitions to a
  // single WAIT_i (guarded on the AND of all latched child_dones in
  // this frame — latching is done in the hardware wrapper), then
  // POST_i (1-cycle settle). Otherwise the last cycle state transitions
  // directly to the next frame (or COND for the last frame).
  //
  // iter_advance fires in POST_i of the last frame (if it has
  // launches) or the last cycle state of the last frame (if it
  // doesn't).
  //
  // Multiple launches in the same frame run concurrently: each fires
  // its start pulse at its own atOffset, and the single WAIT waits for
  // all of them to finish. Launches at the same atOffset fire in the
  // same cycle.
  for (unsigned i = 0; i < numFrames; ++i) {
    bool isLast = (i + 1 == numFrames);
    unsigned L = frameLats[i];
    bool frameHasLaunch = !frameWaitIdx[i].empty();

    // Precompute per-cycle start/live lists for this frame.
    // frameWaitIdx[i] holds global launch indices for launches in frame i;
    // launchAtOffsets[j] is the at-offset for launch j.
    SmallVector<SmallVector<unsigned>> startsPerCycle(L);
    SmallVector<SmallVector<unsigned>> livesPerCycle(L);
    for (int j : frameWaitIdx[i]) {
      unsigned o = launchAtOffsets[(unsigned)j];
      if (o < L)
        startsPerCycle[o].push_back((unsigned)j);
      for (unsigned c = o; c < L; ++c)
        livesPerCycle[c].push_back((unsigned)j);
    }
    SmallVector<unsigned> allLaunches;
    for (int j : frameWaitIdx[i])
      allLaunches.push_back((unsigned)j);

    std::string leaveTargetName =
        isLast ? std::string("COND") : frameStateName(i + 1, 0);

    // Emit L cycle states.
    for (unsigned c = 0; c < L; ++c) {
      bool isLastCycle = (c + 1 == L);
      std::string stateName = frameStateName(i, c);
      // iter_advance fires in the LAST cycle of the LAST frame only
      // when the frame has no launches. If the frame has launches,
      // iter_advance fires in POST_i instead (so the iter_arg register
      // latches the final frame result produced via the await region).
      bool iterAdv = isLast && isLastCycle && !frameHasLaunch;

      auto st = fsm::StateOp::create(fb, loc, stateName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      // frame_active_<i> is high during every cycle state of frame i,
      // whether or not the frame has launches. This lets the per-frame
      // merge mux keep driving the frame's contributed memory address
      // while the launch's cycle hasn't arrived yet (e.g. a load at
      // at-0 inside a frame whose launch fires at at-1 needs its
      // address held through cycles 0..1 so the memory's 1-cycle-
      // latency read-register produces the right value when the
      // pipeline consumes it).
      fsm::OutputOp::create(
          fb, loc,
          buildOutSets(falseVal, iterAdv, /*activeFrame=*/(int)i,
                       /*activeChildStarts=*/startsPerCycle[c],
                       /*activeLive=*/livesPerCycle[c],
                       /*activePosts=*/{},
                       /*cycleFrame=*/cycleOutBase[i] >= 0 ? (int)i : -1,
                       /*cycleIdx=*/cycleOutBase[i] >= 0 ? (int)c : -1));
      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      if (!isLastCycle) {
        fsm::TransitionOp::create(fb, loc,
                                  StringRef(frameStateName(i, c + 1)));
      } else if (frameHasLaunch) {
        fsm::TransitionOp::create(fb, loc, StringRef(waitStateName(i)));
      } else if (isLast) {
        emitLastFrameTransition(tb);
      } else {
        fsm::TransitionOp::create(fb, loc, StringRef(leaveTargetName));
      }
      fb.setInsertionPointToEnd(&machine.getBody().front());
    }

    if (!frameHasLaunch)
      continue;

    // WAIT_i: hold while any launch is still running. child_active_j
    // stays high for every launch in this frame. Transition guard: AND
    // of all (hardware-latched) child_dones for launches in this
    // frame. The wrapper feeds a latched signal into child_done_<j> so
    // that a launch which finished early still reports done while WAIT
    // is sampling.
    {
      auto wSt = fsm::StateOp::create(fb, loc, waitStateName(i));
      Block *ob = wSt.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      fsm::OutputOp::create(
          fb, loc,
          buildOutSets(falseVal, /*iterAdv=*/false, /*activeFrame=*/-1,
                       /*activeChildStarts=*/{},
                       /*activeLive=*/allLaunches,
                       /*activePosts=*/{}));
      Block *tb = &wSt.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      fsm::TransitionOp::create(
          fb, loc, StringRef(postStateName(i)),
          [&]() {
            // AND of child_done_<j> for every launch in this frame.
            Value guard;
            for (int j : frameWaitIdx[i]) {
              Value done = machine.getArgument(2 + (unsigned)j);
              guard = guard ? comb::AndOp::create(fb, loc, guard, done)
                            : done;
            }
            fsm::ReturnOp::create(fb, loc, guard);
          },
          []() {});
      fb.setInsertionPointToEnd(&machine.getBody().front());
    }

    // POST_i: 1-cycle settle where post_active goes high for every
    // launch in this frame. iter_advance pulses in POST_i of the last
    // frame (the cycle where the iter_arg register actually latches
    // the next iteration's value).
    {
      bool pIterAdv = isLast;
      auto pSt = fsm::StateOp::create(fb, loc, postStateName(i));
      Block *ob = pSt.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      fsm::OutputOp::create(
          fb, loc,
          buildOutSets(falseVal, pIterAdv, /*activeFrame=*/-1,
                       /*activeChildStarts=*/{},
                       /*activeLive=*/{},
                       /*activePosts=*/allLaunches));
      Block *tb = &pSt.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      if (isLast) {
        emitLastFrameTransition(tb);
      } else {
        fsm::TransitionOp::create(fb, loc, StringRef(leaveTargetName));
      }
      fb.setInsertionPointToEnd(&machine.getBody().front());
    }
  }

  // --- DONE ---
  {
    auto st = fsm::StateOp::create(fb, loc, "DONE");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    fsm::OutputOp::create(fb, loc, buildOut(trueVal, false, -1, -1, -1, -1));
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(fb, loc, StringRef("IDLE"));
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  return machine;
}

//===----------------------------------------------------------------------===//
// Function-level FSM (linear run-once, no loop condition)
//===----------------------------------------------------------------------===//

fsm::MachineOp LoopScheduleToFSMPass::createFunctionFSM(
    OpBuilder &builder, Location loc, StringRef fsmName,
    ArrayRef<int> frameChildKind) {
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();
  unsigned numFrames = frameChildKind.size();

  // Count non-leaf frames to determine child_done inputs and child_start
  // outputs.
  unsigned numChildren = 0;
  SmallVector<int> childIndexForFrame(numFrames, -1);
  for (unsigned i = 0; i < numFrames; ++i) {
    if (frameChildKind[i] >= 0) {
      childIndexForFrame[i] = numChildren;
      numChildren++;
    }
  }

  // Inputs: start, child_done_0, ..., child_done_{numChildren-1}
  SmallVector<Type> inputTypes(1 + numChildren, i1);
  // Outputs: done, child_start_0..N, frame_running_0..M
  SmallVector<Type> outputTypes(1 + numChildren + numFrames, i1);

  auto funcType = FunctionType::get(ctx, inputTypes, outputTypes);
  auto machine =
      fsm::MachineOp::create(builder, loc, fsmName, "IDLE", funcType);

  // Arg names.
  SmallVector<Attribute> argNameAttrs;
  argNameAttrs.push_back(builder.getStringAttr("start"));
  for (unsigned i = 0; i < numChildren; ++i)
    argNameAttrs.push_back(
        builder.getStringAttr("child_done_" + std::to_string(i)));
  machine.setArgNamesAttr(builder.getArrayAttr(argNameAttrs));

  // Result names.
  SmallVector<Attribute> resNameAttrs;
  resNameAttrs.push_back(builder.getStringAttr("done"));
  for (unsigned i = 0; i < numChildren; ++i)
    resNameAttrs.push_back(
        builder.getStringAttr("child_start_" + std::to_string(i)));
  for (unsigned i = 0; i < numFrames; ++i)
    resNameAttrs.push_back(
        builder.getStringAttr("frame_running_" + std::to_string(i)));
  machine.setResNamesAttr(builder.getArrayAttr(resNameAttrs));

  OpBuilder fb(ctx);
  fb.setInsertionPointToEnd(&machine.getBody().front());

  Value trueVal = hw::ConstantOp::create(fb, loc, i1, 1);
  Value falseVal = hw::ConstantOp::create(fb, loc, i1, 0);

  // Helper to build output vector.
  // out[0] = done, out[1..numChildren] = child_start,
  // out[1+numChildren..] = frame_running
  auto makeOutput = [&](bool done, int activeChildStart,
                        int activeFrameRunning) -> SmallVector<Value> {
    SmallVector<Value> vals;
    vals.push_back(done ? trueVal : falseVal);
    for (unsigned i = 0; i < numChildren; ++i)
      vals.push_back((int)i == activeChildStart ? trueVal : falseVal);
    for (unsigned i = 0; i < numFrames; ++i)
      vals.push_back((int)i == activeFrameRunning ? trueVal : falseVal);
    return vals;
  };

  // --- IDLE ---
  {
    auto st = fsm::StateOp::create(fb, loc, "IDLE");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    fsm::OutputOp::create(fb, loc, makeOutput(false, -1, -1));
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(
        fb, loc, StringRef("FRAME_0"),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
        []() {});
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // --- FRAME_i and WAIT_i states ---
  for (unsigned i = 0; i < numFrames; ++i) {
    std::string frameName = "FRAME_" + std::to_string(i);
    std::string nextState =
        (i + 1 < numFrames) ? "FRAME_" + std::to_string(i + 1) : "DONE";
    bool isLeaf = (frameChildKind[i] < 0);

    // FRAME_i
    {
      auto st = fsm::StateOp::create(fb, loc, frameName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);

      if (isLeaf) {
        // Leaf: frame_running_i = 1, no child_start
        fsm::OutputOp::create(fb, loc, makeOutput(false, -1, i));
      } else {
        // Non-leaf: child_start_j = 1, frame_running_i = 1
        fsm::OutputOp::create(fb, loc,
                               makeOutput(false, childIndexForFrame[i], i));
      }

      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);

      if (isLeaf) {
        // Leaf: unconditional to next
        fsm::TransitionOp::create(fb, loc, StringRef(nextState));
      } else {
        // Non-leaf: go to WAIT_i
        std::string waitName = "WAIT_" + std::to_string(i);
        fsm::TransitionOp::create(fb, loc, StringRef(waitName));
      }
    }
    fb.setInsertionPointToEnd(&machine.getBody().front());

    // WAIT_i (non-leaf only)
    if (!isLeaf) {
      std::string waitName = "WAIT_" + std::to_string(i);
      auto st = fsm::StateOp::create(fb, loc, waitName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      // frame_running_i stays high during WAIT
      fsm::OutputOp::create(fb, loc, makeOutput(false, -1, i));
      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      // Transition to next on child_done
      unsigned childDoneArgIdx = 1 + childIndexForFrame[i];
      fsm::TransitionOp::create(
          fb, loc, StringRef(nextState),
          [&]() {
            fsm::ReturnOp::create(fb, loc,
                                   machine.getArgument(childDoneArgIdx));
          },
          []() {});
      fb.setInsertionPointToEnd(&machine.getBody().front());
    }
  }

  // --- DONE ---
  {
    auto st = fsm::StateOp::create(fb, loc, "DONE");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    fsm::OutputOp::create(fb, loc, makeOutput(true, -1, -1));
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(fb, loc, StringRef("IDLE"));
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  return machine;
}

//===----------------------------------------------------------------------===//
// Loop module creation helpers
//===----------------------------------------------------------------------===//

/// Create an hw.module for a single sequential loop.
static hw::HWModuleOp createLoopModule(
    OpBuilder &builder, Location loc, StringRef moduleName,
    ArrayRef<Value> capturedValues,
    ArrayRef<PortArgInfo> memrefArgs,
    ArrayRef<Type> resultTypes,
    IRMapping &capturedMapping,
    DenseMap<Value, MemPortMapping> &localMemPortMap,
    unsigned &clkIdx, unsigned &rstIdx, unsigned &startIdx) {

  auto *ctx = builder.getContext();
  SmallVector<hw::PortInfo> ports;
  unsigned inputIdx = 0;

  auto clockType = seq::ClockType::get(ctx);
  ports.push_back({{builder.getStringAttr("clk"), clockType,
                     hw::ModulePort::Direction::Input}});
  clkIdx = inputIdx++;

  ports.push_back({{builder.getStringAttr("rst"), builder.getI1Type(),
                     hw::ModulePort::Direction::Input}});
  rstIdx = inputIdx++;

  ports.push_back({{builder.getStringAttr("start"), builder.getI1Type(),
                     hw::ModulePort::Direction::Input}});
  startIdx = inputIdx++;

  for (auto [i, val] : llvm::enumerate(capturedValues)) {
    ports.push_back({{builder.getStringAttr("cap_" + std::to_string(i)),
                       val.getType(), hw::ModulePort::Direction::Input}});
    inputIdx++;
  }

  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    bool hasRdInput = memInfo.isAmcPort ? memInfo.isRead : true;
    if (!hasRdInput)
      continue;
    std::string baseName = "mem" + std::to_string(i);
    bool multi = memInfo.numPorts > 1;
    for (unsigned port = 0; port < memInfo.numPorts; ++port) {
      std::string portPrefix = multi ? (baseName + "_p" +
                                         std::to_string(port))
                                     : baseName;
      ports.push_back({{builder.getStringAttr(portPrefix + "_rd_data"),
                         memInfo.elementType,
                         hw::ModulePort::Direction::Input}});
      inputIdx++;
    }
  }

  ports.push_back({{builder.getStringAttr("done"), builder.getI1Type(),
                     hw::ModulePort::Direction::Output}});

  for (auto [i, ty] : llvm::enumerate(resultTypes)) {
    ports.push_back({{builder.getStringAttr("result_" + std::to_string(i)), ty,
                       hw::ModulePort::Direction::Output}});
  }

  for (auto [i, memInfo] : llvm::enumerate(memrefArgs))
    appendPortOutputPorts(builder, "mem" + std::to_string(i), memInfo, ports);

  hw::ModulePortInfo portInfo(ports);
  auto hwMod = hw::HWModuleOp::create(builder, loc,
                                        builder.getStringAttr(moduleName),
                                        portInfo, ArrayAttr{}, {},
                                        StringAttr{}, false);

  Block *hwBody = hwMod.getBodyBlock();
  unsigned argIdx = 3; // skip clk, rst, start

  for (auto [i, val] : llvm::enumerate(capturedValues)) {
    capturedMapping.map(val, hwBody->getArgument(argIdx++));
  }

  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    MemPortMapping mp;
    bool hasRdInput = memInfo.isAmcPort ? memInfo.isRead : true;
    if (hasRdInput) {
      mp.rdData = hwBody->getArgument(argIdx++);
      for (unsigned k = 1; k < memInfo.numPorts; ++k) {
        PortDrives p;
        p.rdData = hwBody->getArgument(argIdx++);
        p.addrs.assign(memInfo.addrWidths.size(), Value());
        mp.extraPorts.push_back(p);
      }
    }
    mp.addrs.assign(memInfo.addrWidths.size(), Value());
    localMemPortMap[memInfo.originalArg] = mp;
  }

  return hwMod;
}

/// Build hw.output for a loop module.
static void buildLoopModuleOutput(
    OpBuilder &builder, Location loc,
    DenseMap<Value, MemPortMapping> &localMemPortMap,
    ArrayRef<PortArgInfo> memrefArgs,
    Value doneSignal,
    ArrayRef<Value> resultValues) {

  SmallVector<Value> outputs;

  outputs.push_back(doneSignal);

  for (Value v : resultValues)
    outputs.push_back(v);

  for (auto &memInfo : memrefArgs)
    appendPortOutputValues(builder, loc, memInfo, memInfo.originalArg,
                           localMemPortMap, outputs);

  hw::OutputOp::create(builder, loc, outputs);
}

/// Merge per-step memory port mappings into a single mapping using
/// step_running signals. Since steps are mutually exclusive, a priority mux
/// chain selects the active step's ports. Loops over each declared port
/// of the memref independently — accesses bound to port K only contend
/// with other port-K accesses.
static void mergeStepMemPorts(
    OpBuilder &builder, Location loc,
    ArrayRef<DenseMap<Value, MemPortMapping>> perStepPorts,
    ArrayRef<Value> stepRunningSignals,
    ArrayRef<PortArgInfo> memrefArgs,
    DenseMap<Value, MemPortMapping> &mergedPorts) {

  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();

  for (auto &memInfo : memrefArgs) {
    Type dataType = memInfo.elementType;
    auto &merged = mergedPorts[memInfo.originalArg];

    for (unsigned port = 0; port < memInfo.numPorts; ++port) {
      SmallVector<Value> addrs;
      for (unsigned w : memInfo.addrWidths) {
        Type addrType = IntegerType::get(ctx, w);
        addrs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
      }
      Value wrData = hw::ConstantOp::create(builder, loc, dataType, 0);
      Value wrEn = hw::ConstantOp::create(builder, loc, i1, 0);
      Value rdEn = memInfo.requiresRdEn
                       ? hw::ConstantOp::create(builder, loc, i1, 0)
                       : Value();

      for (int i = perStepPorts.size() - 1; i >= 0; --i) {
        auto it = perStepPorts[i].find(memInfo.originalArg);
        if (it == perStepPorts[i].end())
          continue;
        PortDrivesView pv = portView(it->second, port);
        for (auto [d, w] : llvm::enumerate(memInfo.addrWidths)) {
          Type addrType = IntegerType::get(ctx, w);
          Value stepAddr = (pv.addrs && d < pv.addrs->size() && (*pv.addrs)[d])
              ? (*pv.addrs)[d]
              : hw::ConstantOp::create(builder, loc, addrType, 0);
          addrs[d] = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                          stepAddr, addrs[d]);
        }
        Value stepWrData = pv.wrData ? pv.wrData
            : hw::ConstantOp::create(builder, loc, dataType, 0);
        Value stepWrEn = pv.wrEn ? pv.wrEn
            : hw::ConstantOp::create(builder, loc, i1, 0);
        wrData = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                      stepWrData, wrData);
        wrEn = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                    stepWrEn, wrEn);
        if (memInfo.requiresRdEn) {
          Value stepRdEn = pv.rdEn ? pv.rdEn
              : hw::ConstantOp::create(builder, loc, i1, 0);
          rdEn = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                      stepRdEn, rdEn);
        }
      }

      PortDrivesRef mref = portRef(merged, port);
      *mref.addrs = std::move(addrs);
      *mref.wrData = wrData;
      *mref.wrEn = wrEn;
      if (memInfo.requiresRdEn)
        *mref.rdEn = rdEn;
    }
  }
}

/// Merge per-pipeline-stage memory port mappings into a single mapping using
/// per-stage clock-enable signals. When the pipeline's II is at least the
/// stage count, each `stageCE[i]` pulses in a distinct cycle within the II
/// window, so the selectors are mutually exclusive and a priority mux suffices.
/// `outPorts` keeps the caller-provided `rdData` unchanged and only overwrites
/// the addrs/wrData/wrEn fields with the muxed values.
static void muxStageMemPorts(
    OpBuilder &builder, Location loc,
    ArrayRef<DenseMap<Value, MemPortMapping>> perStagePorts,
    ArrayRef<Value> stageCE,
    DenseMap<Value, MemPortMapping> &outPorts) {

  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();

  // Collect every memref that is accessed by any stage.
  SmallVector<Value, 4> memrefs;
  llvm::SmallDenseSet<Value> seen;
  for (auto &stagePorts : perStagePorts) {
    for (auto &entry : stagePorts) {
      if (seen.insert(entry.first).second)
        memrefs.push_back(entry.first);
    }
  }

  for (Value memref : memrefs) {
    // Derive per-dim addr widths, element type, and read-enable need from
    // the stage ports themselves rather than the memref type — the latter
    // does not exist for amc port-typed values.
    SmallVector<unsigned> widths;
    Type dataType;
    bool needsRdEn = false;
    for (auto &stagePorts : perStagePorts) {
      auto it = stagePorts.find(memref);
      if (it == stagePorts.end())
        continue;
      const MemPortMapping &ports = it->second;
      if (widths.empty() && !ports.addrs.empty()) {
        for (Value a : ports.addrs)
          if (a)
            widths.push_back(cast<IntegerType>(a.getType()).getWidth());
      }
      if (!dataType) {
        if (ports.wrData)
          dataType = ports.wrData.getType();
        else if (ports.rdData)
          dataType = ports.rdData.getType();
      }
      if (ports.rdEn)
        needsRdEn = true;
    }
    if (widths.empty() || !dataType)
      continue; // nothing to mux for this memref (probably read-only with
                // data-type already known to caller — not our concern).

    // Use the caller's pre-existing contributions (if any) as the
    // fallback values — so static at-body ops that set addr/wrEn at
    // non-pipeline cycles continue to drive those signals when no
    // stage is active. Without this, a load at at-K inside a wait
    // frame has its address wiped the moment the pipeline runs
    // muxStageMemPorts on the same memref.
    auto &existing = outPorts[memref];
    SmallVector<Value> addrs;
    for (auto [d, w] : llvm::enumerate(widths)) {
      Type addrType = IntegerType::get(ctx, w);
      Value fallback = (d < existing.addrs.size() && existing.addrs[d])
                            ? existing.addrs[d]
                            : hw::ConstantOp::create(builder, loc, addrType, 0);
      addrs.push_back(fallback);
    }
    Value wrData = existing.wrData
                        ? existing.wrData
                        : hw::ConstantOp::create(builder, loc, dataType, 0);
    Value wrEn = existing.wrEn
                      ? existing.wrEn
                      : hw::ConstantOp::create(builder, loc, i1, 0);
    Value rdEn;
    if (needsRdEn)
      rdEn = existing.rdEn ? existing.rdEn
                           : hw::ConstantOp::create(builder, loc, i1, 0);

    // Priority mux chain — last stage has lowest priority. Stages that did
    // not actually touch this memref (no address, no write) contribute
    // nothing: folding in their zero-address would clobber a concurrent
    // stage's real address, since in the pipelined steady state every
    // stageCE is high simultaneously.
    for (int s = (int)perStagePorts.size() - 1; s >= 0; --s) {
      auto it = perStagePorts[s].find(memref);
      if (it == perStagePorts[s].end())
        continue;
      const MemPortMapping &ports = it->second;
      bool hasAddr = llvm::any_of(ports.addrs,
                                  [](Value v) { return (bool)v; });
      if (!hasAddr && !ports.wrData && !ports.wrEn && !ports.rdEn)
        continue;
      Value gate = stageCE[s];
      for (auto [d, w] : llvm::enumerate(widths)) {
        Type addrType = IntegerType::get(ctx, w);
        Value stageAddr = (d < ports.addrs.size() && ports.addrs[d])
            ? ports.addrs[d]
            : hw::ConstantOp::create(builder, loc, addrType, 0);
        addrs[d] =
            comb::MuxOp::create(builder, loc, gate, stageAddr, addrs[d]);
      }
      if (ports.wrData)
        wrData = comb::MuxOp::create(builder, loc, gate, ports.wrData, wrData);
      if (ports.wrEn)
        wrEn = comb::MuxOp::create(builder, loc, gate, ports.wrEn, wrEn);
      if (needsRdEn && ports.rdEn)
        rdEn = comb::MuxOp::create(builder, loc, gate, ports.rdEn, rdEn);
    }

    auto &out = outPorts[memref];
    out.addrs = std::move(addrs);
    out.wrData = wrData;
    out.wrEn = wrEn;
    if (needsRdEn)
      out.rdEn = rdEn;
    // rdData is preserved (it is set by the caller from the module's input
    // port and shared across all stages).
  }
}

//===----------------------------------------------------------------------===//
// Modular loop lowering (one hw.module per sequential loop)
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleToFSMPass::lowerLoopNodeAsModule(
    const LoopNode &node, OpBuilder &builder, Location loc,
    loopschedule::LoopScheduleFuncSequentialOp funcOp, ArrayRef<PortArgInfo> memrefArgs,
    IRMapping &parentMapping,
    hw::HWModuleOp &outModule,
    SmallVectorImpl<Value> &capturedVals) {

  auto *ctx = builder.getContext();
  auto seqOp = node.seqOp;
  auto i1 = builder.getI1Type();

  // --- Collect captured values and constants ---
  SmallVector<Value> captured = collectCapturedValues(seqOp);
  capturedVals.assign(captured.begin(), captured.end());

  SmallVector<Operation *> referencedConsts = collectReferencedConstants(seqOp);

  // Collect result types.
  SmallVector<Type> resultTypes;
  for (auto result : seqOp.getResults())
    resultTypes.push_back(result.getType());

  // --- Create the hw.module for this loop ---
  IRMapping localMapping;
  DenseMap<Value, MemPortMapping> localMemPorts;
  unsigned clkIdx, rstIdx, startIdx;

  auto moduleOp = dyn_cast<ModuleOp>(builder.getBlock()->getParentOp());
  if (!moduleOp)
    moduleOp = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());

  auto hwMod = createLoopModule(builder, loc, node.prefix, captured, memrefArgs,
                                 resultTypes, localMapping, localMemPorts,
                                 clkIdx, rstIdx, startIdx);
  outModule = hwMod;

  Block *hwBody = hwMod.getBodyBlock();
  Value clk = hwBody->getArgument(clkIdx);
  Value rst = hwBody->getArgument(rstIdx);
  Value startSignal = hwBody->getArgument(startIdx);

  // Reset the per-hw.module instance-name uniquer so generated `hw.instance`
  // sym_names don't collide across nested loop modules.
  instanceUniquer.clear();

  // Clone referenced constants into the module body.
  OpBuilder hw(ctx);
  hw.setInsertionPointToEnd(hwBody);
  for (auto *constOp : referencedConsts) {
    hw.clone(*constOp, localMapping);
  }

  // --- Collect frames ---
  SmallVector<LoopScheduleFrameOp> frames;
  for (auto &op : seqOp.getScheduleBlock().getOperations())
    if (auto frameOp = dyn_cast<LoopScheduleFrameOp>(&op))
      frames.push_back(frameOp);

  if (frames.empty())
    return seqOp.emitError("sequential loop has no frames");

  auto terminatorOp =
      cast<LoopScheduleTerminatorOp>(seqOp.getScheduleBlock().getTerminator());

  unsigned numFrames = frames.size();

  // Collect the global wait slots — one per launch across all frames,
  // emitted in (frame, at-offset) order. `frameWaitIdx[i]` lists the
  // global wait indices for frame i's launches; `launchAtOffsets[j]`
  // carries launch j's at-offset (= cycle within its frame where
  // child_start pulses).
  SmallVector<unsigned> waitFrameIndices;
  SmallVector<unsigned> launchAtOffsets;
  SmallVector<SmallVector<int>> frameWaitIdx(numFrames);
  for (unsigned i = 0; i < numFrames; ++i) {
    for (auto &slot : node.frameLaunches[i]) {
      frameWaitIdx[i].push_back((int)waitFrameIndices.size());
      waitFrameIndices.push_back(i);
      launchAtOffsets.push_back(slot.atOffset);
    }
  }
  unsigned numWaits = waitFrameIndices.size();

  // --- Compute per-frame latencies ---
  // A frame's latency is max(computed latency from its body, max launch
  // at-offset + 1). Frames with no launches use the body-derived
  // latency; frames with launches extend to cover their latest launch.
  SmallVector<unsigned> frameLatencies(numFrames, 1);
  for (unsigned i = 0; i < numFrames; ++i) {
    frameLatencies[i] = computeFrameLatency(frames[i]);
    for (int j : frameWaitIdx[i])
      frameLatencies[i] = std::max(frameLatencies[i],
                                    launchAtOffsets[(unsigned)j] + 1);
  }
  // Per-frame base index in the FSM's appended cycle-output region.
  SmallVector<int> frameCycleOutBase(numFrames, -1);
  unsigned totalCycleOuts = 0;
  for (unsigned i = 0; i < numFrames; ++i) {
    if (frameLatencies[i] > 1) {
      frameCycleOutBase[i] = (int)totalCycleOuts;
      totalCycleOuts += frameLatencies[i];
    }
  }

  // --- Create FSM machine ---
  std::string fsmName = node.prefix + "_fsm";
  builder.setInsertionPointToEnd(moduleOp.getBody());
  (void)createSequentialFSM(builder, loc, fsmName, numFrames, waitFrameIndices,
                            launchAtOffsets, frameLatencies);

  // --- Create FSM instance with backedges ---
  hw.setInsertionPointToEnd(hwBody);
  BackedgeBuilder bb(hw, loc);

  Backedge condBE = bb.get(i1);

  // One backedge per wait step for its child_done input.
  SmallVector<Backedge> childDoneBEs;
  childDoneBEs.reserve(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    childDoneBEs.push_back(bb.get(i1));

  // Build instance inputs: start, cond, child_done_0..C-1.
  SmallVector<Value> instInputs;
  instInputs.push_back(startSignal);
  instInputs.push_back(Value(condBE));
  for (auto &be : childDoneBEs)
    instInputs.push_back(Value(be));

  // Result types: done, first_iter, iter_advance,
  //               frame_active_0..N-1, child_start_0..C-1,
  //               child_active_0..C-1, post_active_0..C-1,
  //               frame_cycle_<i>_<c>...
  unsigned numFsmResults = 3 + numFrames + 3 * numWaits + totalCycleOuts;
  SmallVector<Type> resTys(numFsmResults, i1);
  auto inst = fsm::HWInstanceOp::create(
      hw, loc, resTys, hw.getStringAttr(fsmName + "_inst"),
      hw.getAttr<FlatSymbolRefAttr>(fsmName), instInputs, clk, rst);

  Value fsmDone = inst.getResult(0);
  Value fsmFirstIter = inst.getResult(1);
  Value fsmIterAdvance = inst.getResult(2);
  SmallVector<Value> fsmFrameActives(numFrames);
  for (unsigned i = 0; i < numFrames; ++i)
    fsmFrameActives[i] = inst.getResult(3 + i);
  SmallVector<Value> fsmChildStarts(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    fsmChildStarts[j] = inst.getResult(3 + numFrames + j);
  SmallVector<Value> fsmChildActives(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    fsmChildActives[j] = inst.getResult(3 + numFrames + numWaits + j);
  SmallVector<Value> fsmPostActives(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    fsmPostActives[j] = inst.getResult(3 + numFrames + 2 * numWaits + j);

  // SR-latch every raw child_done so concurrent launches that finish at
  // different cycles all contribute to the WAIT_i AND guard. The latch
  // sets on the raw done (covers both level-held pip_done and
  // 1-cycle-pulse sequential-child done) and resets on the launch's
  // own child_start (next iteration). The wrapper returns a
  // combinational OR of rawDone with the held register, so WAIT_i
  // still sees "done" on the SAME cycle rawDone fires — matching
  // pre-latch timing for single-launch frames while still covering
  // the race case where a concurrent launch finishes early.
  //
  // Register creation is pinned to the module body (hwBody), not the
  // wandering hw.setInsertionPoint during per-frame lowering.
  auto latchDone = [&](Value rawDone, Value childStart,
                       StringRef name) -> Value {
    // Signals:
    //   fed       = (rawDone | reg) & ~childStart     -- combinational
    //   reg_next  = fed
    // The AND-with-~childStart is the crucial reset: when a new
    // iteration begins, childStart pulses and fed goes 0 in that
    // cycle even if rawDone is still held high from the previous
    // iteration (e.g. pip_done, which resets one cycle after
    // child_start fires). Without this reset the register would
    // never clear between iterations for pipelines whose done is a
    // held level, causing WAIT_i to transition immediately on the
    // next iteration.
    OpBuilder savedBuilder(hw.getContext());
    savedBuilder.setInsertionPointToEnd(hwBody);
    Value falseConstHere =
        hw::ConstantOp::create(savedBuilder, loc, i1, 0);
    Backedge nextBE = bb.get(i1);
    auto reg = seq::CompRegOp::create(savedBuilder, loc, Value(nextBE), clk,
                                       rst, falseConstHere,
                                       savedBuilder.getStringAttr(name));
    Value notStart =
        comb::createOrFoldNot(savedBuilder, loc, childStart);
    Value rawOrReg =
        comb::OrOp::create(savedBuilder, loc, rawDone, reg);
    Value fed =
        comb::AndOp::create(savedBuilder, loc, rawOrReg, notStart);
    nextBE.setValue(fed);
    return fed;
  };
  // Per-frame per-cycle gates. For single-cycle frames the gate vector
  // contains just the frame's overall frame_active signal; for multi-cycle
  // frames it contains the L_i dedicated frame_cycle_<i>_<c> outputs.
  unsigned cycleOutOffset = 3 + numFrames + 3 * numWaits;
  SmallVector<SmallVector<Value>> fsmFrameCycleGates(numFrames);
  for (unsigned i = 0; i < numFrames; ++i) {
    if (frameCycleOutBase[i] < 0) {
      fsmFrameCycleGates[i].push_back(fsmFrameActives[i]);
    } else {
      for (unsigned c = 0; c < frameLatencies[i]; ++c)
        fsmFrameCycleGates[i].push_back(inst.getResult(
            cycleOutOffset + (unsigned)frameCycleOutBase[i] + c));
    }
  }

  // Clock enable for iter_arg registers: advance exactly once per loop trip,
  // on the FSM's last cycle before COND. first_iter forces the init load on
  // entry to a new loop invocation.
  Value ce = comb::OrOp::create(hw, loc, fsmIterAdvance, fsmFirstIter);

  // --- Create iter arg registers ---
  SmallVector<Value> iterArgRegs;
  for (auto [init, iterArg] :
       llvm::zip(seqOp.getInits(), seqOp.getScheduleBlock().getArguments())) {
    Value mappedInit = localMapping.lookup(init);
    auto regName = hw.getStringAttr(
        node.prefix + "_iter_arg_" + std::to_string(iterArgRegs.size()));
    auto reg = seq::CompRegClockEnabledOp::create(hw, loc, mappedInit, clk, ce,
                                                   rst, mappedInit, regName);
    iterArgRegs.push_back(reg);
    Value muxed = comb::MuxOp::create(hw, loc, fsmFirstIter, mappedInit, reg);
    localMapping.map(iterArg, muxed);
  }

  // --- Per-frame capture gates ---
  // Regular (non-wait) frames latch their yield-op operands into hardware
  // registers at the end of the frame so that later frames read the
  // correctly-held value rather than the combinational expression driven off
  // now-deallocated addresses. For single-cycle regular frames, capture on
  // frame_active_<i>. For multi-cycle regular frames, capture on the last
  // sub-cycle gate. Wait frames (child sequential / pipeline) capture
  // nothing: their results come from child modules whose outputs are already
  // registered.
  SmallVector<Value> frameCaptureGate(numFrames);
  for (unsigned i = 0; i < numFrames; ++i) {
    if (!frameWaitIdx[i].empty())
      continue;
    frameCaptureGate[i] = fsmFrameCycleGates[i].back();
  }

  // Combinational aliases for frame-result values, captured BEFORE each
  // frame's capture registers rewrite the mapping. Used for wiring the FSM
  // cond input and sequential iter_arg register feedback — both of which
  // must see the current-iteration combinational value, not a one-cycle-
  // lagged register.
  DenseMap<Value, Value> frameResultComb;

  // Per-frame memory port mappings. Each starts with the shared `rdData`
  // from the module's input port so that loads always read the correct port.
  // Per-frame addrs/wrData/wrEn are merged at the end via a priority mux
  // gated on each frame's "alive" signal — the wait-frame memory ops (child
  // launch, child active, post-child stores) must all see their frame as
  // alive, so wait frames use `child_active_j | post_active_j` as their
  // merge selector instead of `frame_active_i` (which is -1 for wait
  // frames).
  SmallVector<DenseMap<Value, MemPortMapping>> perFramePorts(numFrames);
  for (unsigned i = 0; i < numFrames; ++i) {
    for (auto &memInfo : memrefArgs) {
      perFramePorts[i][memInfo.originalArg].rdData =
          localMemPorts[memInfo.originalArg].rdData;
    }
  }

  // Determine which frame produces the loop condition so we can resolve the
  // condition backedge after lowering that frame.
  Value condTermVal = terminatorOp.getCondition();
  unsigned condFrameIdx = 0;
  for (auto [idx, f] : llvm::enumerate(frames)) {
    if (condTermVal.getDefiningOp() == f.getOperation()) {
      condFrameIdx = idx;
      break;
    }
  }

  // --- Lower frame bodies ---
  // Helper: lower the non-launch ops of a wait frame (frame containing a
  // launch). `at` ops in the same frame execute concurrently with the
  // launched child, gated on `preGate` (the launch-cycle signal). The
  // launch op itself is handled separately by the caller. `postGate` is
  // retained for API symmetry but is not currently applied: this emitter
  // does not place post-launch ops inside a wait frame's body (post-wait
  // work lives in a subsequent frame).
  auto lowerFrameWithChild =
      [&](LoopScheduleFrameOp frame, unsigned frameIdx, Value preGate,
          Value /*postGate*/, DenseMap<Value, MemPortMapping> &framePorts)
      -> LogicalResult {
    bool hasCycleGates = frameLatencies[frameIdx] > 1;
    Block &frameBody = frame.getBodyBlock();

    // Is this at-op's `res` consumed at a strictly-later at-offset in
    // the same frame, or inside a launch-holder at-op in the same
    // frame? Such consumers run ≥1 cycle after `atOp`, so the
    // combinational mapping (e.g. a load's rd_data) goes stale as the
    // memory port moves on and we need to latch the value.
    auto resultNeedsCapture = [&](Value atResult, unsigned myOffset) {
      for (auto *user : atResult.getUsers()) {
        Operation *anc = frameBody.findAncestorOpInBlock(*user);
        if (!anc)
          continue;
        auto otherAt = dyn_cast<LoopScheduleAtOp>(anc);
        if (!otherAt)
          continue;
        if ((unsigned)otherAt.getOffset() > myOffset)
          return true;
        for (auto &op : otherAt.getBodyBlock())
          if (isa<LoopScheduleLaunchOp>(&op))
            return true;
      }
      return false;
    };

    for (auto atOp : frame.getBodyBlock().getOps<LoopScheduleAtOp>()) {
      Block &atBody = atOp.getBodyBlock();
      // Skip launch-holder ats; their child loop is lowered as a module.
      bool isLaunchHolder = false;
      for (auto &op : atBody)
        if (isa<LoopScheduleLaunchOp>(&op)) {
          isLaunchHolder = true;
          break;
        }
      if (isLaunchHolder)
        continue;
      unsigned atOffset = (unsigned)atOp.getOffset();
      // For multi-cycle wait frames, gate each at-K op by its own
      // FRAME_<i>_<K> cycle output; for single-cycle wait frames
      // fsmFrameCycleGates[i][0] is just frame_active_<i>.
      Value atGate =
          hasCycleGates && atOffset < fsmFrameCycleGates[frameIdx].size()
              ? fsmFrameCycleGates[frameIdx][atOffset]
              : preGate;
      for (auto &op : atBody) {
        if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp,
                LoopScheduleLaunchOp>(&op))
          continue;
        if (auto loadOp =
                dyn_cast<loopschedule::HWLoadLoweringInterface>(&op)) {
          if (failed(handleHWLoad(loadOp, hw, localMapping, framePorts,
                                     atGate)))
            return failure();
          continue;
        }
        if (auto storeOp =
                dyn_cast<loopschedule::HWStoreLoweringInterface>(&op)) {
          if (failed(handleHWStore(storeOp, hw, localMapping, atGate,
                                      framePorts)))
            return failure();
          continue;
        }
        hw.clone(op, localMapping);
      }
      // Map this at-op's external results from its yield operands *now*,
      // so subsequent at-ops in the same frame (which are processed in
      // order) can use those results. Without this, a later at-op that
      // reads `%prev_at#k` would clone into hw with a dangling operand
      // because the at-result hadn't been bound yet.
      auto atYield = atOp.getYieldOp();
      for (auto [res, val] :
           llvm::zip(atOp.getResults(), atYield.getOperands())) {
        if (localMapping.lookupOrNull(res))
          continue;
        if (auto mapped = localMapping.lookupOrNull(val))
          localMapping.map(res, mapped);
      }
      // Latch any at-op result that a later at-op or a launched
      // child reads. The combinational mapping for a load is the
      // memory port's `rd_data`, which is only valid for the single
      // cycle after the load's address is applied; without this
      // capture the consumer reads whatever address the port has
      // moved to next.
      if (hasCycleGates && atOffset + 1 < fsmFrameCycleGates[frameIdx].size()) {
        Value captureGate = fsmFrameCycleGates[frameIdx][atOffset + 1];
        hw.setInsertionPointToEnd(hwBody);
        for (auto res : atOp.getResults()) {
          if (!resultNeedsCapture(res, atOffset))
            continue;
          Value mapped = localMapping.lookupOrNull(res);
          if (!mapped)
            continue;
          if (!isa<IntegerType>(mapped.getType()))
            continue;
          Value resetVal = createZeroConstant(hw, loc, mapped.getType());
          auto regName = hw.getStringAttr(
              node.prefix + "_f" + std::to_string(frameIdx) + "_at" +
              std::to_string(atOffset) + "_r" +
              std::to_string(res.getResultNumber()) + "_latched");
          Value latched = seq::CompRegClockEnabledOp::create(
              hw, loc, mapped, clk, captureGate, rst, resetVal, regName);
          localMapping.map(res, latched);
        }
      }
    }
    return success();
  };

  // Map from handle SSA values to the SmallVector of SSA Values produced by
  // the corresponding launched child (after the child instance has been
  // created). Used to resolve `loopschedule.await` operands in later frames.
  DenseMap<Value, SmallVector<Value>> handleValueMap;

  // Helper: process a frame's await region. For each `loopschedule.await`
  // op, map its results to the stashed child outputs from `handleValueMap`.
  // Then forward the await-region's yield operands to the body-region
  // entry-block arguments.
  auto processAwaitRegion = [&](LoopScheduleFrameOp frame) {
    for (auto &op : frame.getAwaitBlock().getOperations()) {
      if (auto awaitOp = dyn_cast<LoopScheduleAwaitOp>(&op)) {
        // Gather stashed child values across all handle operands.
        SmallVector<Value> childVals;
        for (Value h : awaitOp.getHandles()) {
          auto it = handleValueMap.find(h);
          if (it != handleValueMap.end())
            for (Value v : it->second)
              childVals.push_back(v);
        }
        // Child values may include iter-arg passthrough results that the
        // downstream consumer doesn't want (e.g. a pipeline with
        // `results(counter_next, accumulator)` is followed by
        // `await … -> i32` that only picks up the accumulator). Right-align
        // the await's result list to the tail of the child-value list so
        // those leading iter-arg values fall off.
        unsigned offset = childVals.size() > awaitOp.getNumResults()
                               ? childVals.size() - awaitOp.getNumResults()
                               : 0;
        for (auto [idx, res] : llvm::enumerate(awaitOp.getResults())) {
          if (offset + idx < childVals.size())
            localMapping.map(res, childVals[offset + idx]);
        }
      }
    }
    // Forward yield operands to body-block entry args.
    auto awaitYield = frame.getAwaitYield();
    Block &body = frame.getBodyBlock();
    for (auto [arg, val] :
         llvm::zip(body.getArguments(), awaitYield.getOperands())) {
      if (auto mapped = localMapping.lookupOrNull(val))
        localMapping.map(arg, mapped);
      else
        localMapping.map(arg, val);
    }
  };

  for (auto [frameIdx, frameOp] : llvm::enumerate(frames)) {
    hw.setInsertionPointToEnd(hwBody);

    // Resolve any awaits in the await region before lowering the body.
    processAwaitRegion(frameOp);

    auto &slots = node.frameLaunches[frameIdx];
    if (!slots.empty()) {
      // Run non-launch-holder at-body ops once for the whole frame (they
      // don't depend on which slot is currently running).
      Value firstSlotStart = fsmChildStarts[frameWaitIdx[frameIdx].front()];
      Value framePostActive =
          fsmPostActives[frameWaitIdx[frameIdx].back()];
      if (failed(lowerFrameWithChild(frameOp, frameIdx,
                                      firstSlotStart, framePostActive,
                                      perFramePorts[frameIdx])))
        return failure();

      auto launches = loopschedule::getLaunchOpsInOrder(frameOp);
      assert(launches.size() == slots.size() &&
             "LoopNode launch slots must match frame's launch count");
      for (auto [slotIdx, slot] : llvm::enumerate(slots)) {
        int waitIdx = frameWaitIdx[frameIdx][slotIdx];
        Value slotChildStart = fsmChildStarts[waitIdx];
        Value slotChildActive = fsmChildActives[waitIdx];
        auto launchOp = launches[slotIdx];

        if (slot.childIdx >= 0) {
          auto &childNode = node.children[slot.childIdx];
          auto childSeqOp = childNode.seqOp;
          // Recursively create child module.
          hw::HWModuleOp childModule;
          SmallVector<Value> childCaptured;
          if (failed(lowerLoopNodeAsModule(childNode, builder, loc, funcOp,
                                            memrefArgs, localMapping,
                                            childModule, childCaptured)))
            return failure();

          // Instantiate child module.
          hw.setInsertionPointToEnd(hwBody);
          SmallVector<Value> childInputs;
          childInputs.push_back(clk);
          childInputs.push_back(rst);
          childInputs.push_back(slotChildStart);
          for (Value cap : childCaptured)
            childInputs.push_back(localMapping.lookup(cap));
          for (auto &memInfo : memrefArgs) {
            bool hasRdInput =
                memInfo.isAmcPort ? memInfo.isRead : true;
            if (!hasRdInput)
              continue;
            auto &callerMp =
                perFramePorts[frameIdx][memInfo.originalArg];
            childInputs.push_back(callerMp.rdData);
            for (unsigned k = 1; k < memInfo.numPorts; ++k) {
              if (k - 1 < callerMp.extraPorts.size())
                childInputs.push_back(callerMp.extraPorts[k - 1].rdData);
              else
                childInputs.push_back(hw::ConstantOp::create(
                    hw, loc, memInfo.elementType, 0));
            }
          }

          auto childInst = hw::InstanceOp::create(
              hw, loc, childModule,
              hw.getStringAttr(childNode.prefix + "_inst"),
              childInputs, nullptr);

          unsigned outIdx = 0;
          Value childDone = childInst.getResult(outIdx++);
          // The latch is only needed when multiple launches in this
          // frame may finish at different cycles — a single-launch
          // frame's WAIT_i samples done on the same cycle the raw
          // signal fires and doesn't risk missing it.
          Value feedDone = (slots.size() > 1)
              ? latchDone(childDone, slotChildStart,
                          childNode.prefix + "_done_latch")
              : childDone;
          childDoneBEs[waitIdx].setValue(feedDone);

          SmallVector<Value> childResultVals;
          for (auto result : childSeqOp.getResults()) {
            Value v = childInst.getResult(outIdx++);
            localMapping.map(result, v);
            childResultVals.push_back(v);
          }
          if (auto launchAt =
                  launchOp->getParentOfType<LoopScheduleAtOp>()) {
            auto atYield = launchAt.getYieldOp();
            for (auto [atRes, yOperand] :
                 llvm::zip(launchAt.getResults(), atYield.getOperands())) {
              if (yOperand == launchOp.getHandle())
                handleValueMap[atRes] = childResultVals;
            }
          }
          handleValueMap[launchOp.getHandle()] = std::move(childResultVals);

          // Mux this child's memory drives into perFramePorts[frameIdx]
          // under slotChildActive, per port. Multiple launches in the
          // same frame mux through cascading under each slot's active
          // signal.
          for (auto &memInfo : memrefArgs) {
            auto widths = ArrayRef<unsigned>(memInfo.addrWidths);
            bool emitWrite =
                memInfo.isAmcPort ? memInfo.isWrite : true;
            Type dataType = memInfo.elementType;
            for (unsigned port = 0; port < memInfo.numPorts; ++port) {
              SmallVector<Value> childAddrs;
              for (unsigned d = 0; d < widths.size(); ++d)
                childAddrs.push_back(childInst.getResult(outIdx++));
              Value childRdEn = memInfo.requiresRdEn
                                    ? childInst.getResult(outIdx++)
                                    : Value();
              Value childWrData, childWrEn;
              if (emitWrite) {
                childWrData = childInst.getResult(outIdx++);
                childWrEn = childInst.getResult(outIdx++);
              }
              PortDrivesRef portsR =
                  portRef(perFramePorts[frameIdx][memInfo.originalArg], port);
              if (portsR.addrs->size() != widths.size())
                portsR.addrs->assign(widths.size(), Value());
              for (auto [d, w] : llvm::enumerate(widths)) {
                Type addrType = IntegerType::get(ctx, w);
                Value myAddr = (*portsR.addrs)[d]
                    ? (*portsR.addrs)[d]
                    : hw::ConstantOp::create(hw, loc, addrType, 0);
                (*portsR.addrs)[d] = comb::MuxOp::create(
                    hw, loc, slotChildActive, childAddrs[d], myAddr);
              }
              if (memInfo.requiresRdEn) {
                Value myRdEn = *portsR.rdEn ? *portsR.rdEn
                                            : hw::ConstantOp::create(
                                                  hw, loc, i1, 0);
                *portsR.rdEn = comb::MuxOp::create(
                    hw, loc, slotChildActive, childRdEn, myRdEn);
              }
              if (emitWrite) {
                Value myWrData = *portsR.wrData
                    ? *portsR.wrData
                    : hw::ConstantOp::create(hw, loc, dataType, 0);
                Value myWrEn = *portsR.wrEn ? *portsR.wrEn
                                            : hw::ConstantOp::create(
                                                  hw, loc, i1, 0);
                *portsR.wrData = comb::MuxOp::create(
                    hw, loc, slotChildActive, childWrData, myWrData);
                *portsR.wrEn = comb::MuxOp::create(
                    hw, loc, slotChildActive, childWrEn, myWrEn);
              }
            }
          }
        } else if (slot.pipIdx >= 0) {
          int pipIdx = slot.pipIdx;
          auto pipOp = node.pipelineChildren[pipIdx];
          Value frameChildStart = slotChildStart;
          bool multiSlot = slots.size() > 1;

          // Lower pipeline child (inline in this module). For multi-launch
          // frames the pipeline needs a private mem-port mapping so its
          // muxStageMemPorts (which starts fresh from zero for every
          // memref it touches) doesn't clobber the drives a prior slot
          // has already composed into perFramePorts[frameIdx]; we then
          // mux its drives into perFramePorts under slotChildActive. For
          // single-launch frames the old direct-write path is
          // equivalent and avoids an extra mux layer.
          hw.setInsertionPointToEnd(hwBody);
          Value pipDone;
          std::string pipPrefix =
              node.prefix + "_pip" + std::to_string(pipIdx);
          if (!multiSlot) {
            if (failed(lowerPipelineChild(pipOp, hw, loc, hwBody, localMapping,
                                           clk, rst, frameChildStart,
                                           pipPrefix, pipDone,
                                           perFramePorts[frameIdx],
                                           memrefArgs)))
              return failure();
          } else {
            DenseMap<Value, MemPortMapping> slotPorts;
            for (auto &memInfo : memrefArgs)
              slotPorts[memInfo.originalArg].rdData =
                  perFramePorts[frameIdx][memInfo.originalArg].rdData;
            if (failed(lowerPipelineChild(pipOp, hw, loc, hwBody, localMapping,
                                           clk, rst, frameChildStart,
                                           pipPrefix, pipDone, slotPorts,
                                           memrefArgs)))
              return failure();
            for (auto &memInfo : memrefArgs) {
              auto widths = ArrayRef<unsigned>(memInfo.addrWidths);
              auto &slotP = slotPorts[memInfo.originalArg];
              auto &outP = perFramePorts[frameIdx][memInfo.originalArg];
              if (outP.addrs.size() != widths.size())
                outP.addrs.assign(widths.size(), Value());
              Type dataType = memInfo.elementType;
              for (auto [d, w] : llvm::enumerate(widths)) {
                Type addrType = IntegerType::get(ctx, w);
                Value slotAddr =
                    (d < slotP.addrs.size() && slotP.addrs[d])
                        ? slotP.addrs[d]
                        : hw::ConstantOp::create(hw, loc, addrType, 0);
                Value myAddr =
                    outP.addrs[d]
                        ? outP.addrs[d]
                        : hw::ConstantOp::create(hw, loc, addrType, 0);
                outP.addrs[d] = comb::MuxOp::create(hw, loc, slotChildActive,
                                                      slotAddr, myAddr);
              }
              if (memInfo.requiresRdEn) {
                Value slotRdEn =
                    slotP.rdEn ? slotP.rdEn
                               : hw::ConstantOp::create(hw, loc, i1, 0);
                Value myRdEn =
                    outP.rdEn ? outP.rdEn
                              : hw::ConstantOp::create(hw, loc, i1, 0);
                outP.rdEn = comb::MuxOp::create(hw, loc, slotChildActive,
                                                  slotRdEn, myRdEn);
              }
              bool emitWrite = memInfo.isAmcPort ? memInfo.isWrite : true;
              if (emitWrite) {
                Value slotWrData =
                    slotP.wrData
                        ? slotP.wrData
                        : hw::ConstantOp::create(hw, loc, dataType, 0);
                Value slotWrEn =
                    slotP.wrEn ? slotP.wrEn
                               : hw::ConstantOp::create(hw, loc, i1, 0);
                Value myWrData =
                    outP.wrData
                        ? outP.wrData
                        : hw::ConstantOp::create(hw, loc, dataType, 0);
                Value myWrEn =
                    outP.wrEn ? outP.wrEn
                              : hw::ConstantOp::create(hw, loc, i1, 0);
                outP.wrData = comb::MuxOp::create(hw, loc, slotChildActive,
                                                    slotWrData, myWrData);
                outP.wrEn = comb::MuxOp::create(hw, loc, slotChildActive,
                                                  slotWrEn, myWrEn);
              }
            }
          }
          // Latch only when this frame runs multiple launches
          // concurrently; for a single-launch frame WAIT_i catches
          // pip_done directly without the extra FF.
          Value feedPipDone = (slots.size() > 1)
              ? latchDone(pipDone, slotChildStart,
                          pipPrefix + "_done_latch")
              : pipDone;
          childDoneBEs[waitIdx].setValue(feedPipDone);

          // Stash pipeline's results under the launch's handle (for any
          // future await-with-value in a later frame). Also propagate through
          // the enclosing at's result(s).
          SmallVector<Value> pipResults;
          for (Value r : pipOp.getResults())
            if (auto m = localMapping.lookupOrNull(r))
              pipResults.push_back(m);
          if (auto launchAt =
                  launchOp->getParentOfType<LoopScheduleAtOp>()) {
            auto atYield = launchAt.getYieldOp();
            for (auto [atRes, yOperand] :
                 llvm::zip(launchAt.getResults(), atYield.getOperands())) {
              if (yOperand == launchOp.getHandle())
                handleValueMap[atRes] = pipResults;
            }
          }
          handleValueMap[launchOp.getHandle()] = std::move(pipResults);
        }
      }
    } else {
      // Regular frame (no child/pipeline launches). Gate stores per issue
      // cycle: ops in each `at K` body get cycleGates[K].
      if (failed(lowerFrameBody(&frameOp.getBodyBlock(), hw, localMapping,
                                 fsmFrameCycleGates[frameIdx],
                                 perFramePorts[frameIdx], moduleOp, clk,
                                 rst)))
        return failure();
    }

    // Detect whether this frame contains any loads from local hlmem
    // memories. Local hlmem read ports have latency=1: `rd_data` during
    // cycle N reflects `rd_addr` from cycle N-1. So during the cycle where
    // frame_active_<i> is high, rd_data is stale (reflecting the prior
    // state's addr). We must capture rd_data ONE CYCLE LATER, when it
    // correctly reflects the addr driven during the frame. For frames
    // without local loads, the normal gate works fine.
    bool frameHasLocalLoad = false;
    frameOp->walk([&](LoopScheduleLoadOp loadOp) {
      for (auto &memInfo : memrefArgs) {
        if (memInfo.originalArg == loadOp.getMemRef() && memInfo.isLocalMem) {
          frameHasLocalLoad = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // Forward each at's external results from its yield operands, so that
    // the frame body yield (whose operands reference at results) can be
    // resolved below. The leaf-frame path already does this inside
    // lowerFrameBody, but the wait-frame paths lowered their ops directly
    // into the module — we need to patch the at-result mapping here.
    for (auto atOp : frameOp.getBodyBlock().getOps<LoopScheduleAtOp>()) {
      auto atYield = atOp.getYieldOp();
      for (auto [res, val] :
           llvm::zip(atOp.getResults(), atYield.getOperands())) {
        if (localMapping.lookupOrNull(res))
          continue;
        if (auto mapped = localMapping.lookupOrNull(val))
          localMapping.map(res, mapped);
      }
    }

    // Map frame results. First save the combinational aliases (keyed on the
    // frameOp result value), then — if this frame has a capture gate —
    // create per-operand hardware registers and point the mapping at those
    // so subsequent frame bodies read the held value instead of a
    // combinational expression driven off now-deallocated memory addresses.
    // Wait frames (childIdx/pipelineIdx >= 0) skip the capture: their yield
    // operands come from the child instance and are already registered.
    // Handle-typed results are skipped entirely: they are scheduling
    // metadata with no HW representation.
    auto yieldOp = cast<LoopScheduleYieldOp>(
        frameOp.getBodyBlock().getTerminator());
    // Propagate handle-typed yields so later frames' awaits can resolve.
    for (auto [result, yVal] :
         llvm::zip(frameOp.getResults(), yieldOp.getOperands())) {
      if (!isHandleType(result.getType()))
        continue;
      auto it = handleValueMap.find(yVal);
      if (it != handleValueMap.end())
        handleValueMap[result] = it->second;
    }
    for (auto [result, yVal] :
         llvm::zip(frameOp.getResults(), yieldOp.getOperands())) {
      if (isHandleType(result.getType()))
        continue;
      Value combVal = localMapping.lookup(yVal);
      frameResultComb[result] = combVal;
      localMapping.map(result, combVal);
    }

    if (frameCaptureGate[frameIdx]) {
      hw.setInsertionPointToEnd(hwBody);
      // If this frame contains a local hlmem load, delay the capture gate
      // by one cycle so we latch rd_data when it's valid (the cycle after
      // the address is applied), not when it's still stale.
      Value captureGate = frameCaptureGate[frameIdx];
      if (frameHasLocalLoad) {
        Value falseConstCap =
            hw::ConstantOp::create(hw, loc, hw.getI1Type(), 0);
        captureGate = seq::CompRegOp::create(
            hw, loc, frameCaptureGate[frameIdx], clk, rst, falseConstCap,
            hw.getStringAttr(node.prefix + "_frame" +
                             std::to_string(frameIdx) +
                             "_capture_gate_delayed"));
      }
      for (auto [idx, it] : llvm::enumerate(llvm::zip(
               frameOp.getResults(), yieldOp.getOperands()))) {
        auto [result, yVal] = it;
        if (isHandleType(result.getType()))
          continue;
        Value combVal = frameResultComb[result];
        Value resetVal = createZeroConstant(hw, loc, combVal.getType());
        auto regName = hw.getStringAttr(node.prefix + "_frame" +
                                        std::to_string(frameIdx) + "_r" +
                                        std::to_string(idx));
        Value reg = seq::CompRegClockEnabledOp::create(
            hw, loc, combVal, clk, captureGate, rst, resetVal, regName);
        localMapping.map(result, reg);
      }
    }

    // After lowering the frame that produces the loop condition, wire it
    // into the FSM instance via the condition backedge. Prefer the
    // combinational alias (frameResultComb) over the mapping, which may now
    // point at the post-frame capture register.
    if (frameIdx == condFrameIdx) {
      auto it = frameResultComb.find(condTermVal);
      condBE.setValue(it != frameResultComb.end()
                          ? it->second
                          : localMapping.lookup(condTermVal));
    }
  }

  // --- Wire up iter_arg feedback ---
  // iter_arg registers clock on iter_advance (the last frame's last cycle)
  // which is the SAME posedge the frame-capture registers update on. The
  // captured register sees its own pre-edge value on that posedge, which is
  // stale by one iteration; always prefer the combinational alias for the
  // iter_arg D input.
  //
  // The feedback source is the sequential's terminator-result operand for
  // each iter-arg index — that's by definition the value the loop returns
  // at iteration-end and therefore the value the next iteration should
  // start from. Relying on `getIterArgPhaseResult(iterArgUpdates[i])`
  // instead is fragile: when tmp flows across loop levels via the
  // frame-await chain (e.g. test_conv's k-level tmp updated by the
  // launched l-pipeline), the scheduler emits a verifier-satisfying no-op
  // `iter_arg_update %argN = %argN` which makes the phase-result logic
  // point the iter-arg register back at itself and lose the real update.
  auto iterArgUpdates = loopschedule::getIterArgUpdatesInOrder(seqOp);
  auto termResults = terminatorOp.getResults();
  SmallVector<Value> iterArgPhaseResults(iterArgRegs.size());
  for (unsigned i = 0; i < iterArgRegs.size(); ++i) {
    // Prefer the terminator-result operand at this iter-arg index — that's
    // the value the loop returns at iteration-end, which is exactly what
    // the next iteration's iter-arg should hold. BUT not every iter-arg
    // is externally visible: a loop with empty `results()` still carries
    // iter-args for internal counters. For those (i >= termResults.size())
    // fall back to the iter_arg_update's phase result.
    if (i < termResults.size() && termResults[i])
      iterArgPhaseResults[i] = termResults[i];
    else if (auto u = iterArgUpdates[i])
      iterArgPhaseResults[i] = loopschedule::getIterArgPhaseResult(u);
  }
  hw.setInsertionPointToEnd(hwBody);
  for (unsigned i = 0; i < iterArgRegs.size(); ++i) {
    Value termArg = iterArgPhaseResults[i];
    auto combIt = frameResultComb.find(termArg);
    Value feedback = combIt != frameResultComb.end()
                         ? combIt->second
                         : localMapping.lookup(termArg);
    Value init = localMapping.lookup(seqOp.getInits()[i]);
    Value muxed = comb::MuxOp::create(hw, loc, fsmFirstIter, init, feedback);
    iterArgRegs[i].getDefiningOp()->setOperand(0, muxed);
  }

  // Map sequential op results to register values where possible.
  SmallVector<Value> resultValues;
  for (auto [idx, result] : llvm::enumerate(seqOp.getResults())) {
    Value termResult = terminatorOp.getResults()[idx];
    bool mapped = false;
    for (unsigned j = 0; j < iterArgPhaseResults.size(); ++j) {
      if (termResult == iterArgPhaseResults[j]) {
        resultValues.push_back(iterArgRegs[j]);
        mapped = true;
        break;
      }
    }
    if (!mapped)
      resultValues.push_back(localMapping.lookup(termResult));
  }

  // --- Merge per-frame memory ports ---
  // Build per-frame "alive" signals used by the merge priority mux. Regular
  // frames use their frame_active_<i> (high in every sub-state of
  // FRAME_<i>). Wait frames must cover FRAME+WAIT+POST, and since
  // `frame_active_<i>` is never driven high for wait frames, we OR together
  // child_active_<j> (which spans FRAME and WAIT for that child) and
  // post_active_<j> (POST state).
  hw.setInsertionPointToEnd(hwBody);
  SmallVector<Value> frameMergeSignals(numFrames);
  for (unsigned i = 0; i < numFrames; ++i) {
    // Frame is "alive" on the merge mux during any cycle state
    // (frame_active_<i>) plus — for wait frames — every launch's
    // child_active and post_active so the frame keeps driving its
    // contributed address through the launched child's run.
    Value aliveSoFar = fsmFrameActives[i];
    for (int j : frameWaitIdx[i]) {
      Value slotAlive = comb::OrOp::create(
          hw, loc, fsmChildActives[(unsigned)j], fsmPostActives[(unsigned)j]);
      aliveSoFar = comb::OrOp::create(hw, loc, aliveSoFar, slotAlive);
    }
    frameMergeSignals[i] = aliveSoFar;
  }

  DenseMap<Value, MemPortMapping> mergedMemPorts;
  for (auto &memInfo : memrefArgs)
    mergedMemPorts[memInfo.originalArg].rdData =
        localMemPorts[memInfo.originalArg].rdData;
  mergeStepMemPorts(hw, loc, perFramePorts, frameMergeSignals, memrefArgs,
                    mergedMemPorts);

  // Copy merged ports back into localMemPorts for output construction.
  for (auto &memInfo : memrefArgs) {
    localMemPorts[memInfo.originalArg].addrs =
        mergedMemPorts[memInfo.originalArg].addrs;
    localMemPorts[memInfo.originalArg].wrData =
        mergedMemPorts[memInfo.originalArg].wrData;
    localMemPorts[memInfo.originalArg].wrEn =
        mergedMemPorts[memInfo.originalArg].wrEn;
  }

  // --- Build module output ---
  hw.setInsertionPointToEnd(hwBody);
  buildLoopModuleOutput(hw, loc, localMemPorts, memrefArgs, fsmDone,
                         resultValues);

  return success();
}

//===----------------------------------------------------------------------===//
// Pipeline child lowering (for pipelines nested inside sequential loops)
//===----------------------------------------------------------------------===//

/// Inline each `loopschedule.launch` + `loopschedule.expect` pair in a
/// pipeline body so the FSM lowering sees the same shape it had pre-
/// Phase 2 (the scheduler wraps dynamic ops; the FSM unwraps them here).
/// The launch's single body op is moved to the launch's position, the
/// launch + its inner yield are erased, and the expect is replaced by
LogicalResult LoopScheduleToFSMPass::lowerPipelineChild(
    LoopSchedulePipelineOp pipOp, OpBuilder &builder, Location loc,
    Block *hwBody, IRMapping &mapping, Value clk, Value rst,
    Value startSignal, StringRef namePrefix, Value &doneSignal,
    DenseMap<Value, MemPortMapping> &memPorts,
    ArrayRef<PortArgInfo> memrefArgs) {
  // Phase 3B: before the scheduler's launch/expect pairs are inlined
  // away, snapshot the info the stall logic needs from each expect.
  // Every expect marks a point where a dynamic-latency op's result is
  // consumed — its `destStageOffset` is the cycle the FSM plans to see
  // the value, and its `portValue` identifies the memory port whose
  // `done` signal gates pipeline advance. After `inlineLaunchExpectPairs`
  // runs, the launch/expect ops are gone, so we must capture this first.
  struct ExpectInfo {
    unsigned destStageOffset;
    Value portValue;
  };
  SmallVector<ExpectInfo> expectInfos;
  pipOp.walk([&](LoopScheduleExpectOp expectOp) {
    auto dest = expectOp->getParentOfType<LoopScheduleAtOp>();
    if (!dest)
      return;
    auto launch = expectOp.getLaunchOp();
    if (!launch)
      return;
    Operation *payload = nullptr;
    for (Operation &op : launch.getBody().front()) {
      if (isa<LoopScheduleYieldOp>(op))
        continue;
      payload = &op;
      break;
    }
    if (!payload)
      return;
    auto loadIface = dyn_cast<HWLoadLoweringInterface>(payload);
    if (!loadIface)
      return;
    expectInfos.push_back({(unsigned)dest.getOffset(),
                            loadIface.getMemoryValue()});
  });

  inlineLaunchExpectPairs(pipOp);

  auto *ctx = builder.getContext();

  SmallVector<LoopScheduleAtOp> stages;
  for (auto &op : pipOp.getStagesBlock().getOperations())
    if (auto stageOp = dyn_cast<LoopScheduleAtOp>(&op))
      stages.push_back(stageOp);

  if (stages.empty())
    return pipOp.emitError("pipeline has no stages");

  auto terminatorOp =
      cast<LoopScheduleTerminatorOp>(pipOp.getStagesBlock().getTerminator());

  OpBuilder hwBuilder(ctx);
  hwBuilder.setInsertionPointToEnd(hwBody);

  Value falseConst =
      hw::ConstantOp::create(hwBuilder, loc, hwBuilder.getI1Type(), 0);
  Value trueConst =
      hw::ConstantOp::create(hwBuilder, loc, hwBuilder.getI1Type(), 1);

  BackedgeBuilder bb(hwBuilder, loc);

  unsigned numIterArgs = pipOp.getInits().size();
  SmallVector<Backedge> iterArgBackedges;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value init = mapping.lookup(pipOp.getInits()[i]);
    iterArgBackedges.push_back(bb.get(init.getType()));
  }

  for (auto [arg, be] :
       llvm::zip(pipOp.getStagesBlock().getArguments(), iterArgBackedges))
    mapping.map(arg, Value(be));

  // The loop condition is produced by some stage (guaranteed by the verifier
  // to be within the first II cycles). Stand it up as a backedge here so the
  // active/CE chain can be built first; resolve it after lowering the
  // producing stage below.
  Backedge condValueBE = bb.get(hwBuilder.getI1Type());
  Value condValue = Value(condValueBE);

  // Determine which stage produces the condition.
  Value condTermVal = terminatorOp.getCondition();
  unsigned condStageIdx = 0;
  for (auto [idx, s] : llvm::enumerate(stages)) {
    if (condTermVal.getDefiningOp() == s.getOperation()) {
      condStageIdx = idx;
      break;
    }
  }

  // Phase 3B: stall is a backedge resolved at the end once `memPorts`
  // are fully populated and per-expect `done` signals are looked up.
  // Gating below wires `notStall` into the pipeline's state registers
  // (active, II counter, stageCE shift chain) as well as the stage-CE
  // consumption for rd_en/wr_en gates, so a low `done` at destStage
  // freezes the whole pipeline without dropping in-flight requests.
  Backedge stallBE = bb.get(hwBuilder.getI1Type());
  Value stall = Value(stallBE);
  Value notStall = comb::createOrFoldNot(hwBuilder, loc, stall);

  Backedge activeNextBE = bb.get(hwBuilder.getI1Type());
  auto activeReg = seq::CompRegClockEnabledOp::create(
      hwBuilder, loc, Value(activeNextBE), clk, notStall, rst, falseConst,
      hwBuilder.getStringAttr(namePrefix + "_active"));
  Value active = activeReg;

  Value holdActive = comb::AndOp::create(hwBuilder, loc, active, condValue);
  Value activeNext =
      comb::OrOp::create(hwBuilder, loc, startSignal, holdActive);
  activeNextBE.setValue(activeNext);

  uint64_t II = pipOp.getII();
  Value ceGen;
  if (II == 1) {
    ceGen = active;
  } else {
    unsigned counterWidth = llvm::Log2_64_Ceil(II);
    Type counterType = IntegerType::get(ctx, counterWidth);
    Value cZero = hw::ConstantOp::create(hwBuilder, loc, counterType, 0);
    Value cOne = hw::ConstantOp::create(hwBuilder, loc, counterType, 1);
    Value cIIMinusOne =
        hw::ConstantOp::create(hwBuilder, loc, counterType, II - 1);

    Backedge counterBackedge = bb.get(counterType);
    auto counterReg = seq::CompRegClockEnabledOp::create(
        hwBuilder, loc, Value(counterBackedge), clk, notStall, rst, cZero,
        hwBuilder.getStringAttr(namePrefix + "_ii_counter"));

    Value counterPlusOne =
        comb::AddOp::create(hwBuilder, loc, counterReg, cOne);
    Value atMax = comb::ICmpOp::create(
        hwBuilder, loc, comb::ICmpPredicate::eq, counterReg, cIIMinusOne);
    Value wrapped =
        comb::MuxOp::create(hwBuilder, loc, atMax, cZero, counterPlusOne);
    Value counterNext =
        comb::MuxOp::create(hwBuilder, loc, active, wrapped, cZero);
    counterBackedge.setValue(counterNext);

    Value isZero = comb::ICmpOp::create(
        hwBuilder, loc, comb::ICmpPredicate::eq, counterReg, cZero);
    ceGen = comb::AndOp::create(hwBuilder, loc, isZero, active);
  }

  Value activeCE = comb::AndOp::create(hwBuilder, loc, ceGen, condValue);

  // Raw stage-CE shift chain (unstalled): each register holds during stall
  // (CE=notStall) so positional iteration info isn't lost.
  SmallVector<Value> stageCE(stages.size());
  stageCE[0] = activeCE;
  for (unsigned i = 1; i < stages.size(); ++i) {
    auto ceName = hwBuilder.getStringAttr(
        (namePrefix + "_ce_stage_" + std::to_string(i)).str());
    stageCE[i] = seq::CompRegClockEnabledOp::create(
        hwBuilder, loc, stageCE[i - 1], clk, notStall, rst, falseConst,
        ceName);
  }
  // Gated stage-CE used for memory-side enables (rd_en / wr_en) and any
  // other consumer that must go low during stall — without this the
  // memory would see stuck-high enables for the cycles the pipeline
  // idles, creating spurious repeats.
  SmallVector<Value> gatedStageCE(stages.size());
  for (unsigned i = 0; i < stages.size(); ++i)
    gatedStageCE[i] =
        comb::AndOp::create(hwBuilder, loc, stageCE[i], notStall);

  // Per-stage memory port mappings. Loads/stores in stage `i` accumulate in
  // `perStagePorts[i]` so that multiple accesses to the same memref across
  // stages don't clobber each other in a single shared MemPortMapping. After
  // all stages are lowered we mux them together into `memPorts` using the
  // stage clock-enables as selectors.
  SmallVector<DenseMap<Value, MemPortMapping>> perStagePorts(stages.size());
  for (unsigned s = 0; s < stages.size(); ++s)
    for (auto &entry : memPorts)
      perStagePorts[s][entry.first].rdData = entry.second.rdData;

  // Delay chains for cross-stage values. If a value V is produced at stage J
  // and consumed at stage K > J+1, we must insert (K - J - 1) delay registers
  // so the consumer sees the correct pipeline iteration rather than whatever
  // value currently occupies stage J's register (which is overwritten each
  // cycle in an II=1 pipeline). chain[0] is the direct stage J register;
  // chain[k] (k >= 1) is a delay register clocked on stageCE[J + k]. Consumer
  // stage K reads chain[K - J - 1].
  // Cross-stage delay registers. Owns a per-value delay chain so that
  // when stage J yields a value consumed at stage K > J + L + 1 (with L
  // = the producing op's cycle latency), the consumer reads from an
  // appropriately-clocked shift register rather than the constantly-
  // overwriting stage-J register.
  CrossStageValueResolver delayResolver(
      hwBuilder, loc, hwBody, clk, rst, stages, stageCE, mapping,
      operatorLibrary, namePrefix.str());

  // Helper: check whether a memref is backed by a local seq.hlmem.
  auto isLocalMemref = [&](Value memref) -> bool {
    for (auto &memInfo : memrefArgs)
      if (memInfo.originalArg == memref && memInfo.isLocalMem)
        return true;
    return false;
  };

  for (auto [stageIdx, stageOp] : llvm::enumerate(stages)) {
    Block &body = stageOp.getBodyBlock();
    hwBuilder.setInsertionPointToEnd(hwBody);

    // Track load results from local hlmem memories. These skip the stage
    // register because the hlmem's latency=1 read port already provides the
    // 1-cycle delay the scheduler expects.
    DenseSet<Value> localLoadResults;

    for (auto &op : body.getOperations()) {
      if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp>(&op))
        continue;

      // Temporarily override the mapping entries for any operands that come
      // from earlier stages so the clone sees the correctly-delayed version.
      // Restore the original mappings after lowering this op so later uses
      // aren't affected.
      SmallVector<std::pair<Value, Value>> savedMappings;
      for (Value operand : op.getOperands()) {
        Value delayed = delayResolver.resolveForStage(operand, stageIdx);
        if (!delayed)
          continue;
        Value current = mapping.lookup(operand);
        if (delayed != current) {
          savedMappings.emplace_back(operand, current);
          mapping.map(operand, delayed);
        }
      }

      LogicalResult opResult = success();
      // Recursive body-processor: handles ops inside the current pipeline
      // stage, including nested `loopschedule.if`. The `gate` argument is
      // what a store's `wr_en` (and a load's `rd_en`) gets gated by —
      // `stageCE[stageIdx]` at the top level, AND'd with the condition
      // when we descend into an if body. If-region results are forwarded
      // unconditionally via `mapping` (the hardware treats the body's
      // computation as always happening; the predication only affects
      // when effects reach memory).
      std::function<LogicalResult(Operation *, Value)> processOp =
          [&](Operation *inner, Value gate) -> LogicalResult {
        if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp>(inner))
          return success();
        if (auto ifOp = dyn_cast<LoopScheduleIfOp>(inner)) {
          Value cond = mapping.lookup(ifOp.getCond());
          Value innerGate =
              comb::AndOp::create(hwBuilder, loc, gate, cond);
          for (auto &nested : ifOp.getBody().front()) {
            if (auto yieldOp = dyn_cast<LoopScheduleYieldOp>(&nested)) {
              for (auto [res, val] :
                   llvm::zip(ifOp.getResults(), yieldOp.getOperands()))
                mapping.map(res, mapping.lookup(val));
              continue;
            }
            if (failed(processOp(&nested, innerGate)))
              return failure();
          }
          return success();
        }
        if (auto storeOp = dyn_cast<HWStoreLoweringInterface>(inner))
          return handleHWStore(storeOp, hwBuilder, mapping, gate,
                                  perStagePorts[stageIdx]);
        if (auto loadOp = dyn_cast<HWLoadLoweringInterface>(inner)) {
          if (auto ls = dyn_cast<LoopScheduleLoadOp>(inner))
            if (isLocalMemref(ls.getMemRef()))
              localLoadResults.insert(ls.getResult());
          if (loadOp.requiresReadEnable() && loadOp.getReadLatency() > 0)
            localLoadResults.insert(loadOp.getResult());
          return handleHWLoad(loadOp, hwBuilder, mapping,
                                 perStagePorts[stageIdx], gate);
        }
        return emitComputeOp(
            inner, hwBuilder, mapping,
            hwBody->getParentOp()->getParentOfType<ModuleOp>(), clk, rst);
      };
      // Use the stall-gated stage CE as the write-/read-enable gate so
      // memory requests don't re-fire when the pipeline idles on !done.
      opResult = processOp(&op, gatedStageCE[stageIdx]);

      for (auto &sm : savedMappings)
        mapping.map(sm.first, sm.second);

      if (failed(opResult))
        return failure();
    }

    auto regOp = cast<LoopScheduleYieldOp>(body.getTerminator());

    // Resolve the condition backedge from the producing stage's *unregistered*
    // condition value: the loop's combinational gate should not be delayed by
    // the pipeline registers. The condition is one of the producing stage's
    // results, which corresponds to the register operand at the same index.
    if (stageIdx == condStageIdx) {
      auto condResult = cast<OpResult>(condTermVal);
      condValueBE.setValue(
          mapping.lookup(regOp.getOperand(condResult.getResultNumber())));
      // Refresh the local condValue handle: BackedgeBuilder's RAUW updates
      // the underlying placeholder's uses, but the local Value snapshot still
      // points at the (now-orphaned) cast op.
      condValue = Value(condValueBE);
    }

    for (auto [regIdx, val] : llvm::enumerate(regOp.getOperands())) {
      // The register's D input should see the delayed view for this stage
      // as well, in case the register is a direct pass-through of an
      // earlier stage's value.
      Value delayed = delayResolver.resolveForStage(val, stageIdx);
      Value mappedVal = delayed ? delayed : mapping.lookup(val);

      if (localLoadResults.count(val)) {
        // Local hlmem read (latency=1) already provides the 1-cycle delay.
        // Map the stage result directly to rdData — no extra register.
        mapping.map(stageOp.getResult(regIdx), mappedVal);
        continue;
      }

      // A latent operator-library op (e.g. multi-cycle multiplier) has
      // its own internal pipeline: wrapper.OUTPUT at cycle T is the
      // result of inputs dispatched at cycle T-L. Capturing that output
      // at stage-J's cycle J would latch the wrapper output for a
      // cycle-(J-L) input (undefined for iter 0). Skip the stage
      // register here; `resolveForStage` uses `readyStage = J+L` so
      // stage-J+L consumers read the wrapper output directly and
      // further stages pick it up via delay registers.
      if (computeOpCycleLatency(val.getDefiningOp(), operatorLibrary) > 0) {
        mapping.map(stageOp.getResult(regIdx), mappedVal);
        continue;
      }

      Value resetVal = createZeroConstant(hwBuilder, loc, mappedVal.getType());
      auto regName = hwBuilder.getStringAttr(
          (namePrefix + "_s" + std::to_string(stageIdx) + "_r" +
           std::to_string(regIdx))
              .str());

      Value reg = seq::CompRegClockEnabledOp::create(
          hwBuilder, loc, mappedVal, clk, stageCE[stageIdx], rst, resetVal,
          regName);
      mapping.map(stageOp.getResult(regIdx), reg);
    }
  }

  hwBuilder.setInsertionPointToEnd(hwBody);

  // When II < #stages, stageCE signals form a shift-chain: stageCE[s+1] =
  // CompReg(stageCE[s]). In steady state, stageCE[s] pulses on cycles s,
  // s+II, s+2*II, ..., so two stages s1 < s2 fire on the SAME cycle only
  // when (s2 - s1) % II == 0. The muxStageMemPorts priority mux correctly
  // arbitrates stages that fire on different cycles — it only fails for
  // stages in the same mod-II class. Reject only genuine collisions.
  if (II < stages.size()) {
    for (auto &entry : memPorts) {
      SmallVector<SmallVector<unsigned>> classes(II);
      for (unsigned s = 0; s < stages.size(); ++s) {
        auto it = perStagePorts[s].find(entry.first);
        if (it == perStagePorts[s].end())
          continue;
        const MemPortMapping &ports = it->second;
        bool hasAddr =
            llvm::any_of(ports.addrs, [](Value v) { return (bool)v; });
        if (hasAddr || ports.wrData || ports.wrEn)
          classes[s % II].push_back(s);
      }
      for (auto &cls : classes) {
        if (cls.size() > 1)
          return pipOp.emitError()
                 << "pipeline with II=" << II << " and " << stages.size()
                 << " stages has stages " << cls[0] << " and " << cls[1]
                 << " accessing the same memory on the same cycle "
                 << "(s1 ≡ s2 mod II); increase II or partition the memory";
      }
    }
  }

  muxStageMemPorts(hwBuilder, loc, perStagePorts, stageCE, memPorts);

  // Per-stage first_iter signals. Each stage s's first_iter is 1 on reset,
  // re-arms to 1 on start, and clears the first time stageCE[s] fires. An
  // iter_arg whose feedback value is produced at stage s must read its init
  // value until that stage has fired at least once in this pipeline run — a
  // single global first_iter flipping on `active` is too early for iter_args
  // fed by late stages (e.g. a matmul accumulator at the last stage reads a
  // stale register for cycles 1..N before the stage first writes).
  SmallVector<Value> firstIterPerStage(stages.size());
  for (unsigned s = 0; s < stages.size(); ++s) {
    Backedge be = bb.get(hwBuilder.getI1Type());
    Value notCE = comb::createOrFoldNot(hwBuilder, loc, stageCE[s]);
    Value sticky = comb::AndOp::create(hwBuilder, loc, Value(be), notCE);
    Value nxt = comb::OrOp::create(hwBuilder, loc, startSignal, sticky);
    auto reg = seq::CompRegOp::create(
        hwBuilder, loc, nxt, clk, rst, trueConst,
        hwBuilder.getStringAttr(
            (namePrefix + "_first_iter_s" + std::to_string(s)).str()));
    be.setValue(reg);
    firstIterPerStage[s] = reg;
  }

  auto pipIterArgUpdates = loopschedule::getIterArgUpdatesInOrder(pipOp);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value init = mapping.lookup(pipOp.getInits()[i]);
    Value termVal = pipIterArgUpdates[i]
                        ? loopschedule::getIterArgPhaseResult(pipIterArgUpdates[i])
                        : Value{};
    Value feedback = mapping.lookup(termVal);
    // The feedback value is produced at some stage; first_iter must stay
    // high until that stage has fired at least once. If the feedback isn't
    // a stage result (shouldn't happen for well-formed pipelines), fall
    // back to stage 0.
    unsigned feedbackStage = 0;
    if (auto opResult = dyn_cast<OpResult>(termVal)) {
      if (auto stage = dyn_cast<LoopScheduleAtOp>(opResult.getOwner())) {
        for (unsigned s = 0; s < stages.size(); ++s)
          if (stages[s] == stage) {
            feedbackStage = s;
            break;
          }
      }
    }
    Value muxed = comb::MuxOp::create(
        hwBuilder, loc, firstIterPerStage[feedbackStage], init, feedback);
    iterArgBackedges[i].setValue(muxed);
  }

  for (auto [result, termResult] :
       llvm::zip(pipOp.getResults(), terminatorOp.getResults()))
    mapping.map(result, mapping.lookup(termResult));

  // LegUp-style epilogue done signal. Latch the loop-exit event into an
  // epilogue register, then shift it through a delay chain matching the
  // pipeline depth. Done fires when the delayed epilogue reaches the end —
  // exactly when the last valid bit has drained from the tail stage.
  // Combinational depth: O(1) (single AND of two register outputs).
  Value notCondValue = comb::createOrFoldNot(hwBuilder, loc, condValue);
  Value epilogueTrigger =
      comb::AndOp::create(hwBuilder, loc, active, notCondValue);

  Value notStart = comb::createOrFoldNot(hwBuilder, loc, startSignal);
  Backedge epilogueBE = bb.get(hwBuilder.getI1Type());
  Value epilogueHold =
      comb::AndOp::create(hwBuilder, loc, Value(epilogueBE), notStart);
  Value epilogueNext =
      comb::OrOp::create(hwBuilder, loc, epilogueTrigger, epilogueHold);
  auto epilogueReg = seq::CompRegOp::create(
      hwBuilder, loc, epilogueNext, clk, rst, falseConst,
      hwBuilder.getStringAttr(namePrefix + "_epilogue"));
  epilogueBE.setValue(epilogueReg);

  Value delayedEpilogue = Value(epilogueReg);
  for (unsigned i = 0; i + 2 < stages.size(); ++i) {
    Value delayInput =
        comb::MuxOp::create(hwBuilder, loc, startSignal, falseConst,
                            delayedEpilogue);
    delayedEpilogue = seq::CompRegOp::create(
        hwBuilder, loc, delayInput, clk, rst, falseConst,
        hwBuilder.getStringAttr(
            (namePrefix + "_epilogue_delay_" + std::to_string(i)).str()));
  }

  Value notTailCE = comb::createOrFoldNot(hwBuilder, loc, stageCE.back());
  Value doneComb =
      comb::AndOp::create(hwBuilder, loc, delayedEpilogue, notTailCE);
  Value doneInput =
      comb::MuxOp::create(hwBuilder, loc, startSignal, falseConst, doneComb);
  auto doneReg = seq::CompRegOp::create(
      hwBuilder, loc, doneInput, clk, rst, falseConst,
      hwBuilder.getStringAttr(namePrefix + "_done"));
  doneSignal = doneReg;

  // Phase 3B: resolve the stall backedge. For each expect collected
  // before inlining, stall whenever destStage's CE is firing but the
  // underlying memory port's `done` is low — i.e. the request we issued
  // `latency` cycles ago hasn't completed yet.
  SmallVector<Value> stallBits;
  for (auto &ei : expectInfos) {
    auto it = memPorts.find(ei.portValue);
    if (it == memPorts.end())
      continue;
    Value done = it->second.done;
    if (!done)
      continue; // memref-backed / no done exposed → treat as tied-high
    if (ei.destStageOffset >= stageCE.size())
      continue;
    Value notDone = comb::createOrFoldNot(hwBuilder, loc, done);
    Value stallI = comb::AndOp::create(hwBuilder, loc,
                                         stageCE[ei.destStageOffset], notDone);
    stallBits.push_back(stallI);
  }
  Value stallVal = falseConst;
  if (!stallBits.empty()) {
    stallVal = stallBits.front();
    for (unsigned i = 1; i < stallBits.size(); ++i)
      stallVal =
          comb::OrOp::create(hwBuilder, loc, stallVal, stallBits[i]);
  }
  stallBE.setValue(stallVal);

  return success();
}

//===----------------------------------------------------------------------===//
// Helper: build hw.module ports and create hw.module
//===----------------------------------------------------------------------===//

/// Shared hw.module creation logic used by both sequential and pipeline paths.
/// Accepts any FunctionOpInterface so both `LoopScheduleFuncSequentialOp` and
/// `LoopScheduleFuncPipelineOp` can flow through unchanged.
static hw::HWModuleOp createHWModule(
    mlir::FunctionOpInterface funcOp, OpBuilder &builder, IRMapping &mapping,
    DenseMap<Value, MemPortMapping> &memPortMap,
    unsigned &clkIdx, unsigned &rstIdx, unsigned &startIdx) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();

  SmallVector<hw::PortInfo> ports;
  unsigned inputIdx = 0;

  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (auto memType = dyn_cast<MemRefType>(arg.getType())) {
      std::string baseName = "mem" + std::to_string(idx);
      Type dataType = memType.getElementType();
      PortArgInfo info = makePortArgInfoFromMemref(arg, memType,
                                                    /*isLocalMem=*/false);
      bool multi = info.numPorts > 1;
      for (unsigned port = 0; port < info.numPorts; ++port) {
        std::string portPrefix = multi ? (baseName + "_p" +
                                           std::to_string(port))
                                       : baseName;
        ports.push_back({{builder.getStringAttr(portPrefix + "_rd_data"),
                           dataType,
                           hw::ModulePort::Direction::Input}});
        inputIdx++;
      }
      appendPortOutputPorts(builder, baseName, info, ports);
    } else {
      ports.push_back(
          {{builder.getStringAttr("arg" + std::to_string(idx)), arg.getType(),
            hw::ModulePort::Direction::Input}});
      inputIdx++;
    }
  }

  auto clockType = seq::ClockType::get(ctx);
  ports.push_back(
      {{builder.getStringAttr("clk"), clockType,
        hw::ModulePort::Direction::Input}});
  clkIdx = inputIdx++;

  ports.push_back({{builder.getStringAttr("rst"), builder.getI1Type(),
                     hw::ModulePort::Direction::Input}});
  rstIdx = inputIdx++;

  ports.push_back({{builder.getStringAttr("start"), builder.getI1Type(),
                     hw::ModulePort::Direction::Input}});
  startIdx = inputIdx++;

  // `ready` precedes `done` in the output port list. Both sequential and
  // pipelined func variants drive this port; for sequential it's high in
  // IDLE (no transaction in flight), for pipelined it's high whenever the
  // II window is open (caller can issue a new start). Callers that don't
  // care about back-pressure (e.g. one-shot driver) can ignore it; the
  // testbench wrapper uses it to drive multi-transaction streaming.
  ports.push_back({{builder.getStringAttr("ready"), builder.getI1Type(),
                     hw::ModulePort::Direction::Output}});

  ports.push_back({{builder.getStringAttr("done"), builder.getI1Type(),
                     hw::ModulePort::Direction::Output}});

  for (auto [idx, retType] : llvm::enumerate(funcOp.getResultTypes())) {
    if (isHandleType(retType))
      continue;
    ports.push_back(
        {{builder.getStringAttr("result" + std::to_string(idx)), retType,
          hw::ModulePort::Direction::Output}});
      }

  hw::ModulePortInfo portInfo(ports);
  builder.setInsertionPointAfter(funcOp);

  auto hwMod = hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(funcOp.getName()), portInfo,
      ArrayAttr{}, {}, StringAttr{}, false);

  for (auto attr : funcOp->getAttrs()) {
    if (attr.getName().strref().starts_with("testbench.") ||
        attr.getName().strref() == "top")
      hwMod->setAttr(attr.getName(), attr.getValue());
  }

  // Map function arguments.
  memPortMap.clear();
  Block *hwBody = hwMod.getBodyBlock();
  unsigned hwArgIdx = 0;
  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    if (auto memTy = dyn_cast<MemRefType>(arg.getType())) {
      MemPortMapping mp;
      unsigned numPorts = computeNumPortsFromUsers(arg);
      // Port 0 lands in the top-level fields; remaining ports populate
      // `extraPorts[K-1]`. Address slots are pre-sized so handleHWLoad/
      // Store can index them without re-resizing on first use.
      mp.rdData = hwBody->getArgument(hwArgIdx++);
      mp.addrs.assign(getNumAddrPorts(memTy), Value());
      for (unsigned k = 1; k < numPorts; ++k) {
        PortDrives p;
        p.rdData = hwBody->getArgument(hwArgIdx++);
        p.addrs.assign(getNumAddrPorts(memTy), Value());
        mp.extraPorts.push_back(p);
      }
      memPortMap[arg] = mp;
    } else {
      mapping.map(arg, hwBody->getArgument(hwArgIdx));
      hwArgIdx++;
    }
  }

  return hwMod;
}

/// Build hw.output with memory ports, ready/done handshake, and return values.
static void buildHWOutput(mlir::FunctionOpInterface funcOp, OpBuilder &builder,
                          Location loc, Block *hwBody, IRMapping &mapping,
                          DenseMap<Value, MemPortMapping> &memPortMap,
                          Value readySignal, Value doneSignal) {
  SmallVector<Value> outputs;

  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    auto memType = dyn_cast<MemRefType>(arg.getType());
    if (!memType)
      continue;
    PortArgInfo info = makePortArgInfoFromMemref(arg, memType,
                                                  /*isLocalMem=*/false);
    appendPortOutputValues(builder, loc, info, arg, memPortMap, outputs);
  }

  // Order matches createHWModule: ready, then done, then results.
  outputs.push_back(readySignal);
  outputs.push_back(doneSignal);

  auto returnOp = cast<loopschedule::LoopScheduleReturnOp>(
      funcOp.getFunctionBody().front().getTerminator());
  for (auto retVal : returnOp.getOperands()) {
    if (isHandleType(retVal.getType()))
      continue;
    if (auto mapped = mapping.lookupOrNull(retVal))
      outputs.push_back(mapped);
    else
      outputs.push_back(createZeroConstant(builder, loc, retVal.getType()));
  }

  hw::OutputOp::create(builder, loc, outputs);
}

//===----------------------------------------------------------------------===//
// Function lowering (main entry)
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleToFSMPass::setupFunctionPrelude(
    Block &body, ValueRange funcArgs, OpBuilder &builder, Location loc,
    Value clk, Value rst, ModuleOp enclosingModule,
    llvm::function_ref<bool(Operation *)> isStageOrFrameOp,
    IRMapping &mapping, BackedgeBuilder &funcBB,
    DenseMap<Value, MemPortMapping> &memPortMap,
    SmallVectorImpl<HLMemBackedges> &hlmemBEs,
    loopschedule::HWMemoryLoweringState &memInstState,
    SmallVectorImpl<PortArgInfo> &memrefArgs) {
  // Pass 1: walk the body once, classifying each top-level op:
  //   * skip the return op,
  //   * skip caller-owned scheduling containers (frames / stages — the
  //     `isStageOrFrameOp` predicate distinguishes which kind),
  //   * collect `memref.alloc` for hlmem creation,
  //   * collect `HWMemoryInstanceLoweringInterface` ops for amc lowering,
  //   * clone everything else into the hw.module body via `mapping`.
  SmallVector<memref::AllocOp> localAllocs;
  SmallVector<loopschedule::HWMemoryInstanceLoweringInterface> instanceOps;
  for (auto &op : body) {
    if (isa<loopschedule::LoopScheduleReturnOp>(&op))
      continue;
    if (isStageOrFrameOp(&op))
      continue;
    if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
      localAllocs.push_back(allocOp);
      continue;
    }
    if (auto instOp =
            dyn_cast<loopschedule::HWMemoryInstanceLoweringInterface>(&op)) {
      instanceOps.push_back(instOp);
      continue;
    }
    builder.clone(op, mapping);
  }

  // Pass 2: materialize a `seq.hlmem` for each local memref.alloc, with
  // backedges for addresses, write data, and write enable. The read
  // port's enable is the negation of write enable so a bank with a
  // pending write doesn't also read on the same cycle.
  auto i1 = builder.getI1Type();
  for (auto [i, allocOp] : llvm::enumerate(localAllocs)) {
    auto memType = allocOp.getType();
    auto dataType = memType.getElementType();
    std::string name = "local_mem" + std::to_string(i);
    auto hlmem = seq::HLMemOp::create(builder, loc, clk, rst, name,
                                       memType.getShape(), dataType);

    SmallVector<Backedge> addrBEs;
    SmallVector<Value> addrVals;
    for (Type addrTy : hlmem.getHandle().getType().getAddressTypes()) {
      Backedge be = funcBB.get(addrTy);
      addrBEs.push_back(be);
      addrVals.push_back(Value(be));
    }
    Backedge wrDataBE = funcBB.get(dataType);
    Backedge wrEnBE = funcBB.get(i1);

    Value notWrEn = comb::XorOp::create(
        builder, loc, Value(wrEnBE),
        hw::ConstantOp::create(builder, loc, i1, 1));
    auto readPort = seq::ReadPortOp::create(
        builder, loc, hlmem.getHandle(), ValueRange(addrVals), notWrEn,
        /*latency=*/1);
    seq::WritePortOp::create(builder, loc, hlmem.getHandle(),
                              ValueRange(addrVals), Value(wrDataBE),
                              Value(wrEnBE), /*latency=*/1);

    MemPortMapping mp;
    mp.rdData = readPort.getReadData();
    mp.addrs.assign(getNumAddrPorts(memType), Value());
    mp.wrData = Value();
    mp.wrEn = Value();
    memPortMap[allocOp.getResult()] = mp;

    hlmemBEs.push_back({allocOp.getResult(), std::move(addrBEs), wrDataBE,
                        wrEnBE});
  }

  // Pass 3: lower each amc.instance via the dialect-defined interface.
  // Each instance owns its own hw.instance creation and registers a
  // per-port signal bundle into `memInstState.portMap`.
  for (auto instOp : instanceOps) {
    if (failed(instOp.lowerToHW(builder, memInstState)))
      return failure();
  }
  for (auto &[portValue, signals] : memInstState.portMap) {
    MemPortMapping mp;
    mp.rdData = signals.rdData;
    mp.addrs.assign(signals.addrs.size(), Value());
    mp.done = signals.done;
    memPortMap[portValue] = mp;
  }

  // Pass 4: assemble `memrefArgs` in canonical order — function args
  // first, then local allocs, then amc-instance ports — so that down-
  // stream port-list construction (`createHWModule`, etc.) sees a
  // stable mem0/mem1/... naming.
  for (auto arg : funcArgs) {
    if (auto memType = dyn_cast<MemRefType>(arg.getType()))
      memrefArgs.push_back(
          makePortArgInfoFromMemref(arg, memType, /*isLocalMem=*/false));
  }
  for (auto allocOp : localAllocs) {
    memrefArgs.push_back(makePortArgInfoFromMemref(allocOp.getResult(),
                                                    allocOp.getType(),
                                                    /*isLocalMem=*/true));
  }
  for (auto &[portValue, signals] : memInstState.portMap) {
    PortArgInfo info;
    info.originalArg = portValue;
    info.isAmcPort = true;
    info.latency = signals.latency;
    info.addrWidths.reserve(signals.addrs.size());
    for (Value a : signals.addrs)
      info.addrWidths.push_back(cast<IntegerType>(a.getType()).getWidth());
    info.isRead = signals.rdData != Value();
    info.isWrite = signals.wrEn != Value();
    info.requiresRdEn = signals.rdEn != Value();
    if (info.isRead)
      info.elementType = signals.rdData.getType();
    else if (info.isWrite)
      info.elementType = signals.wrData.getType();
    memrefArgs.push_back(std::move(info));
  }
  return success();
}

void LoopScheduleToFSMPass::resolveFunctionMemoryBackedges(
    OpBuilder &builder, Location loc, Block *hwBody,
    SmallVectorImpl<HLMemBackedges> &hlmemBEs,
    loopschedule::HWMemoryLoweringState &memInstState,
    DenseMap<Value, MemPortMapping> &memPortMap) {
  builder.setInsertionPointToEnd(hwBody);
  auto i1 = builder.getI1Type();
  for (auto &be : hlmemBEs) {
    auto &ports = memPortMap[be.allocResult];
    auto memTy = cast<MemRefType>(be.allocResult.getType());
    auto dataType = memTy.getElementType();
    for (auto [d, beAddr] : llvm::enumerate(be.addrBEs)) {
      auto hlmemAddrTy = cast<IntegerType>(Value(beAddr).getType());
      Value src;
      if (d < ports.addrs.size() && ports.addrs[d])
        src =
            resizeIntTo(builder, loc, ports.addrs[d], hlmemAddrTy.getWidth());
      else
        src = hw::ConstantOp::create(builder, loc, hlmemAddrTy, 0);
      beAddr.setValue(src);
    }
    be.wrDataBE.setValue(ports.wrData
                              ? ports.wrData
                              : hw::ConstantOp::create(builder, loc, dataType,
                                                        0));
    be.wrEnBE.setValue(ports.wrEn
                            ? ports.wrEn
                            : hw::ConstantOp::create(builder, loc, i1, 0));
  }

  for (auto &pb : memInstState.pendingBackedges) {
    auto it = memPortMap.find(pb.portValue);
    Value resolved;
    Type beTy = Value(pb.be).getType();
    auto makeZero = [&]() {
      return hw::ConstantOp::create(builder, loc, beTy, 0);
    };
    auto makeOne = [&]() {
      return hw::ConstantOp::create(builder, loc, beTy, 1);
    };
    using Kind = loopschedule::PortBackedge::Kind;
    switch (pb.kind) {
    case Kind::Addr:
      if (it != memPortMap.end() && pb.addrIdx < it->second.addrs.size() &&
          it->second.addrs[pb.addrIdx])
        resolved =
            resizeIntTo(builder, loc, it->second.addrs[pb.addrIdx],
                        cast<IntegerType>(beTy).getWidth());
      else
        resolved = makeZero();
      break;
    case Kind::RdEn:
      resolved = (it != memPortMap.end() && it->second.rdEn) ? it->second.rdEn
                                                              : makeOne();
      break;
    case Kind::WrData:
      resolved = (it != memPortMap.end() && it->second.wrData)
                      ? it->second.wrData
                      : makeZero();
      break;
    case Kind::WrEn:
      resolved = (it != memPortMap.end() && it->second.wrEn) ? it->second.wrEn
                                                              : makeZero();
      break;
    }
    pb.be.setValue(resolved);
  }
}

LogicalResult LoopScheduleToFSMPass::lowerFunction(loopschedule::LoopScheduleFuncSequentialOp funcOp) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();
  OpBuilder builder(ctx);

  // Bind a per-function operator-library analysis. The analysis ctor is a
  // no-op when the func has no `oplib.library` attribute; the dispatch
  // helpers will then error on any compute op (the strict contract for
  // the rewritten FSM lowering).
  analysis::OperatorLibraryAnalysis ola(funcOp);
  operatorLibrary = &ola;
  auto unbindLibrary =
      llvm::make_scope_exit([&] { operatorLibrary = nullptr; });

  // --- Sequential path (supports nesting) ---

  IRMapping mapping;
  unsigned clkIdx, rstIdx, startIdx;
  auto hwMod = createHWModule(funcOp, builder, mapping, memPortMap, clkIdx,
                              rstIdx, startIdx);
  Block *hwBody = hwMod.getBodyBlock();
  Value clk = hwBody->getArgument(clkIdx);
  Value rst = hwBody->getArgument(rstIdx);
  Value start = hwBody->getArgument(startIdx);

  // Reset the per-hw.module instance-name uniquer.
  instanceUniquer.clear();

  // Resolve the enclosing builtin.module so the operator-library lowering
  // can look up `hw.module.extern` declarations by symbol.
  auto enclosingModule = funcOp->getParentOfType<ModuleOp>();

  builder.setInsertionPointToEnd(hwBody);

  auto i1 = builder.getI1Type();
  BackedgeBuilder funcBB(builder, loc);
  SmallVector<HLMemBackedges> hlmemBEs;
  SymbolTable modSymTab(enclosingModule);
  loopschedule::HWMemoryLoweringState memInstState(clk, rst, funcBB,
                                                    modSymTab);
  SmallVector<PortArgInfo> memrefArgs;
  if (failed(setupFunctionPrelude(
          funcOp.getBody().front(), funcOp.getArguments(), builder, loc, clk,
          rst, enclosingModule,
          [](Operation *op) {
            return isa<LoopScheduleFrameOp, LoopScheduleSequentialOp,
                       LoopSchedulePipelineOp>(op);
          },
          mapping, funcBB, memPortMap, hlmemBEs, memInstState, memrefArgs)))
    return failure();

  // Collect top-level frames.
  SmallVector<LoopScheduleFrameOp> topFrames;
  for (auto &op : funcOp.getBody().front())
    if (auto frameOp = dyn_cast<LoopScheduleFrameOp>(&op))
      topFrames.push_back(frameOp);

  // --- Multi-frame path ---
  if (topFrames.empty())
    return funcOp.emitError("no top-level frames found");

  unsigned numFrames = topFrames.size();

  // Build a flat list of children: one entry per child op across all frames.
  // Each frame's launch ops are inspected; each launch contains exactly one
  // nested LoopInterface. A frame with multiple launches produces multiple
  // entries sharing the same frameIdx. A leaf frame (no launches) produces
  // one entry with kind=-1.
  struct FrameChild {
    unsigned frameIdx;
    int kind; // -1 leaf, 0 sequential, 1 pipeline, 2 call
    LoopScheduleSequentialOp seqOp;
    LoopSchedulePipelineOp pipOp;
    LoopScheduleCallOp callOp;
    LoopScheduleLaunchOp launchOp; // non-null for kind != -1
  };
  SmallVector<FrameChild> entries;
  for (unsigned i = 0; i < numFrames; ++i) {
    auto launches = loopschedule::getLaunchOpsInOrder(topFrames[i]);
    if (launches.empty()) {
      FrameChild fc;
      fc.frameIdx = i;
      fc.kind = -1;
      entries.push_back(fc);
      continue;
    }
    for (auto launch : launches) {
      Operation *child = nullptr;
      for (auto &op : launch.getBodyBlock().getOperations()) {
        if (isa<LoopScheduleYieldOp>(op))
          continue;
        child = &op;
        break;
      }
      FrameChild fc;
      fc.frameIdx = i;
      fc.launchOp = launch;
      if (auto seqOp = dyn_cast_or_null<LoopScheduleSequentialOp>(child)) {
        fc.kind = 0;
        fc.seqOp = seqOp;
        entries.push_back(fc);
      } else if (auto pipOp = dyn_cast_or_null<LoopSchedulePipelineOp>(child)) {
        fc.kind = 1;
        fc.pipOp = pipOp;
        entries.push_back(fc);
      } else if (auto callOp = dyn_cast_or_null<LoopScheduleCallOp>(child)) {
        fc.kind = 2;
        fc.callOp = callOp;
        entries.push_back(fc);
      }
    }
  }

  // Build a per-entry kind vector for createFunctionFSM. The FSM sees one
  // state per entry, not per step.
  SmallVector<int> entryKinds;
  for (auto &e : entries)
    entryKinds.push_back(e.kind);

  // Count non-leaf entries.
  unsigned numChildren = 0;
  SmallVector<int> childIndexForEntry(entries.size(), -1);
  for (unsigned i = 0; i < entries.size(); ++i) {
    if (entries[i].kind >= 0) {
      childIndexForEntry[i] = numChildren;
      numChildren++;
    }
  }

  // Create function-level FSM.
  std::string funcName = funcOp.getName().str();
  std::string fsmName = funcName + "_fsm";
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  createFunctionFSM(builder, loc, fsmName, entryKinds);

  // Instantiate function FSM in HW module body.
  builder.setInsertionPointToEnd(hwBody);
  BackedgeBuilder bb(builder, loc);

  SmallVector<Backedge> childDoneBEs;
  SmallVector<Value> fsmInputs;
  fsmInputs.push_back(start);
  for (unsigned i = 0; i < numChildren; ++i) {
    auto be = bb.get(i1);
    childDoneBEs.push_back(be);
    fsmInputs.push_back(Value(be));
  }

  unsigned numEntries = entries.size();
  SmallVector<Type> fsmResultTypes(1 + numChildren + numEntries, i1);
  auto fsmInst = fsm::HWInstanceOp::create(
      builder, loc, fsmResultTypes,
      builder.getStringAttr(fsmName + "_inst"),
      builder.getAttr<FlatSymbolRefAttr>(fsmName),
      fsmInputs, clk, rst);

  // Extract FSM outputs.
  unsigned fsmOutIdx = 0;
  Value doneSignal = fsmInst.getResult(fsmOutIdx++);
  SmallVector<Value> childStartSignals(numChildren);
  for (unsigned i = 0; i < numChildren; ++i)
    childStartSignals[i] = fsmInst.getResult(fsmOutIdx++);
  SmallVector<Value> entryRunningSignals(numEntries);
  for (unsigned i = 0; i < numEntries; ++i)
    entryRunningSignals[i] = fsmInst.getResult(fsmOutIdx++);

  // Per-entry memory port mappings. Each starts with shared rdData.
  SmallVector<DenseMap<Value, MemPortMapping>> perEntryPorts(numEntries);
  for (unsigned i = 0; i < numEntries; ++i) {
    for (auto &memInfo : memrefArgs) {
      perEntryPorts[i][memInfo.originalArg].rdData =
          memPortMap[memInfo.originalArg].rdData;
    }
  }

  // Track which entries belong to each frame for pre/post-child op cloning.
  // firstEntryForFrame[f] = index of first entry belonging to frame f.
  // lastEntryForFrame[f]  = index of last entry belonging to frame f.
  SmallVector<unsigned> firstEntryForFrame(numFrames, 0);
  SmallVector<unsigned> lastEntryForFrame(numFrames, 0);
  for (unsigned i = 0; i < numEntries; ++i) {
    unsigned f = entries[i].frameIdx;
    if (i == 0 || entries[i - 1].frameIdx != f)
      firstEntryForFrame[f] = i;
    lastEntryForFrame[f] = i;
  }

  // Map from handle SSA values to the child's output SSA values. Used to
  // resolve `loopschedule.await` ops in later frames.
  DenseMap<Value, SmallVector<Value>> funcHandleValueMap;

  // Helper: for a frame, walk its await region and map await results +
  // body-entry args. Should be called before the frame's body work runs.
  auto processFuncAwaitRegion = [&](LoopScheduleFrameOp frame) {
    for (auto &op : frame.getAwaitBlock().getOperations()) {
      if (auto awaitOp = dyn_cast<LoopScheduleAwaitOp>(&op)) {
        SmallVector<Value> childVals;
        for (Value h : awaitOp.getHandles()) {
          auto it = funcHandleValueMap.find(h);
          if (it != funcHandleValueMap.end())
            for (Value v : it->second)
              childVals.push_back(v);
        }
        // Right-align: see the twin `processAwaitRegion` in the
        // sequential path for the rationale.
        unsigned offset = childVals.size() > awaitOp.getNumResults()
                               ? childVals.size() - awaitOp.getNumResults()
                               : 0;
        for (auto [idx, res] : llvm::enumerate(awaitOp.getResults())) {
          if (offset + idx < childVals.size())
            mapping.map(res, childVals[offset + idx]);
        }
      }
    }
    auto awaitYield = frame.getAwaitYield();
    Block &body = frame.getBodyBlock();
    for (auto [arg, val] :
         llvm::zip(body.getArguments(), awaitYield.getOperands())) {
      if (auto mapped = mapping.lookupOrNull(val))
        mapping.map(arg, mapped);
      else
        mapping.map(arg, val);
    }
  };

  // Helper: lower the non-launch `at` body ops of a frame. Launches are
  // handled per-entry as children; `at` ops that carry real compute coexist
  // with launch-holder ats in the same frame. A launch-holder at is one
  // whose body is just a launch (+ yield); those are skipped entirely.
  // Collected static-store drives from `cloneFrameAtBodies`. Each entry's
  // `wrEn` is already gated by the owning frame's 1-cycle header pulse
  // (see caller). After `mergeStepMemPorts` runs, these drives get
  // priority-composed into `memPortMap` so a static store coexisting
  // with a launch in the same frame takes priority during its single
  // firing cycle (and leaves the pipeline's drives untouched otherwise).
  SmallVector<std::pair<Value, MemPortMapping>> staticDrives;

  auto cloneFrameAtBodies = [&](LoopScheduleFrameOp frame,
                                 Value preGate) -> LogicalResult {
    for (auto atOp : frame.getBodyBlock().getOps<LoopScheduleAtOp>()) {
      bool isLaunchHolder = false;
      for (auto &op : atOp.getBodyBlock()) {
        if (isa<LoopScheduleLaunchOp>(&op)) {
          isLaunchHolder = true;
          break;
        }
      }
      if (isLaunchHolder)
        continue;
      for (auto &op : atOp.getBodyBlock()) {
        if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp,
                LoopScheduleLaunchOp>(&op))
          continue;
        // Memory ops must flow through the HW store/load interface so
        // their memref operands get rewritten to the memPortMap ports
        // instead of leaking into the cloned output as dangling
        // references to the (about-to-be-erased) func-op's block args.
        // We drive into a temporary MemPortMapping so the static drive
        // survives the later `mergeStepMemPorts` overwrite and can be
        // priority-composed with the merged ports.
        if (auto storeOp =
                dyn_cast<loopschedule::HWStoreLoweringInterface>(&op)) {
          DenseMap<Value, MemPortMapping> localPorts;
          for (auto &memInfo : memrefArgs)
            localPorts[memInfo.originalArg].rdData =
                memPortMap[memInfo.originalArg].rdData;
          if (failed(handleHWStore(storeOp, builder, mapping, preGate,
                                      localPorts)))
            return failure();
          Value target = storeOp.getMemoryValue();
          staticDrives.emplace_back(target, std::move(localPorts[target]));
          continue;
        }
        if (auto loadOp =
                dyn_cast<loopschedule::HWLoadLoweringInterface>(&op)) {
          if (failed(handleHWLoad(loadOp, builder, mapping, memPortMap,
                                     preGate)))
            return failure();
          continue;
        }
        if (failed(emitComputeOp(&op, builder, mapping, enclosingModule, clk,
                                  rst)))
          return failure();
      }
      // Eagerly map this at-op's external results so later at-ops in the
      // same frame (e.g. a pipeline-launch-holder at a later offset)
      // can resolve any operands that reference them.
      auto atYield = atOp.getYieldOp();
      for (auto [res, val] :
           llvm::zip(atOp.getResults(), atYield.getOperands())) {
        if (mapping.lookupOrNull(res))
          continue;
        if (auto mapped = mapping.lookupOrNull(val))
          mapping.map(res, mapped);
      }
    }
    return success();
  };

  // Lower each entry.
  unsigned loopCounter = 0;
  for (unsigned ei = 0; ei < numEntries; ++ei) {
    auto &entry = entries[ei];
    unsigned frameIdx = entry.frameIdx;
    builder.setInsertionPointToEnd(hwBody);

    // On first entry of this frame, process its await region (map await
    // results from stashed handle-value map, then forward to body entry
    // args). For non-leaf frames, also clone static `at` body ops that
    // coexist with the launch. Leaf frames defer this to lowerFrameBody
    // below.
    if (ei == firstEntryForFrame[frameIdx]) {
      processFuncAwaitRegion(topFrames[frameIdx]);
      if (entry.kind >= 0) {
        // Static at-body stores need a 1-cycle gate that matches the
        // FRAME_i state (before WAIT_i), so they fire exactly once per
        // frame entry and don't contend with the launched child's
        // memory writes during the rest of frame i. `child_start_i` is
        // exactly that pulse — high only during FRAME_i.
        Value headerGate = childStartSignals[childIndexForEntry[ei]];
        if (failed(cloneFrameAtBodies(topFrames[frameIdx], headerGate)))
          return failure();
      }
    }

    if (entry.kind == 0) {
      // Sequential child: build loop tree and create child module.
      auto seqOp = entry.seqOp;
      std::string prefix = "loop" + std::to_string(loopCounter++);

      LoopNode node;
      unsigned innerCounter = 1;
      buildLoopTree(seqOp, node, prefix, innerCounter);

      hw::HWModuleOp childModule;
      SmallVector<Value> childCaptured;
      if (failed(lowerLoopNodeAsModule(node, builder, loc, funcOp, memrefArgs,
                                       mapping, childModule, childCaptured)))
        return failure();

      // Instantiate child module.
      builder.setInsertionPointToEnd(hwBody);
      SmallVector<Value> childInputs;
      childInputs.push_back(clk);
      childInputs.push_back(rst);
      childInputs.push_back(childStartSignals[childIndexForEntry[ei]]);
      for (Value cap : childCaptured)
        childInputs.push_back(mapping.lookup(cap));
      for (auto &memInfo : memrefArgs) {
        bool hasRdInput = memInfo.isAmcPort ? memInfo.isRead : true;
        if (!hasRdInput)
          continue;
        auto &callerMp = memPortMap[memInfo.originalArg];
        childInputs.push_back(callerMp.rdData);
        for (unsigned k = 1; k < memInfo.numPorts; ++k) {
          if (k - 1 < callerMp.extraPorts.size())
            childInputs.push_back(callerMp.extraPorts[k - 1].rdData);
          else
            childInputs.push_back(hw::ConstantOp::create(
                builder, loc, memInfo.elementType, 0));
        }
      }

      auto childInst = hw::InstanceOp::create(
          builder, loc, childModule,
          builder.getStringAttr(prefix + "_inst"),
          childInputs, nullptr);

      // Extract child outputs.
      unsigned outIdx = 0;
      Value childDone = childInst.getResult(outIdx++);
      childDoneBEs[childIndexForEntry[ei]].setValue(childDone);

      // Map seqOp results.
      SmallVector<Value> childResultVals;
      for (auto result : seqOp.getResults()) {
        Value v = childInst.getResult(outIdx++);
        mapping.map(result, v);
        childResultVals.push_back(v);
      }
      if (entry.launchOp) {
        // Propagate child results through the enclosing at's result(s), so a
        // later frame's await-by-handle (which traces through at→frame yield)
        // can resolve the underlying SSA values.
        if (auto launchAt =
                entry.launchOp->getParentOfType<LoopScheduleAtOp>()) {
          auto atYield = launchAt.getYieldOp();
          for (auto [atRes, yOperand] :
               llvm::zip(launchAt.getResults(), atYield.getOperands())) {
            if (yOperand == entry.launchOp.getHandle())
              funcHandleValueMap[atRes] = childResultVals;
          }
        }
        funcHandleValueMap[entry.launchOp.getHandle()] =
            std::move(childResultVals);
      }

      // Extract child memory outputs into per-entry ports, per-port.
      for (auto &memInfo : memrefArgs) {
        auto &mp = perEntryPorts[ei][memInfo.originalArg];
        auto widths = ArrayRef<unsigned>(memInfo.addrWidths);
        bool emitWrite = memInfo.isAmcPort ? memInfo.isWrite : true;
        for (unsigned port = 0; port < memInfo.numPorts; ++port) {
          PortDrivesRef portsR = portRef(mp, port);
          portsR.addrs->assign(widths.size(), Value());
          for (unsigned d = 0; d < widths.size(); ++d)
            (*portsR.addrs)[d] = childInst.getResult(outIdx++);
          if (memInfo.requiresRdEn)
            *portsR.rdEn = childInst.getResult(outIdx++);
          if (emitWrite) {
            *portsR.wrData = childInst.getResult(outIdx++);
            *portsR.wrEn = childInst.getResult(outIdx++);
          }
        }
      }

    } else if (entry.kind == 1) {
      // Pipeline child: lower inline.
      auto pipOp = entry.pipOp;
      std::string pipPrefix = "loop" + std::to_string(loopCounter++);
      Value pipDone;
      if (failed(lowerPipelineChild(pipOp, builder, loc, hwBody, mapping,
                                    clk, rst,
                                    childStartSignals[childIndexForEntry[ei]],
                                    pipPrefix, pipDone,
                                    perEntryPorts[ei], memrefArgs)))
        return failure();
      childDoneBEs[childIndexForEntry[ei]].setValue(pipDone);

      if (entry.launchOp) {
        SmallVector<Value> pipResults;
        for (Value r : pipOp.getResults())
          if (auto m = mapping.lookupOrNull(r))
            pipResults.push_back(m);
        // Propagate pipeline results through the enclosing at's result(s) so a
        // later frame's await-by-handle resolves through at→frame yield.
        if (auto launchAt =
                entry.launchOp->getParentOfType<LoopScheduleAtOp>()) {
          auto atYield = launchAt.getYieldOp();
          for (auto [atRes, yOperand] :
               llvm::zip(launchAt.getResults(), atYield.getOperands())) {
            if (yOperand == entry.launchOp.getHandle())
              funcHandleValueMap[atRes] = pipResults;
          }
        }
        funcHandleValueMap[entry.launchOp.getHandle()] = std::move(pipResults);
      }

    } else if (entry.kind == 2) {
      // Call: instantiate the already-lowered callee hw.module and wire up
      // the start/done handshake plus memref port pass-through.
      auto callOp = entry.callOp;
      auto calleeName = callOp.getCallee();
      auto *calleeSym = SymbolTable::lookupNearestSymbolFrom(
          funcOp, callOp.getCalleeAttr());
      auto calleeMod = dyn_cast_or_null<hw::HWModuleOp>(calleeSym);
      if (!calleeMod)
        return callOp.emitOpError("callee '")
               << calleeName
               << "' has not been lowered to hw.module yet — callees must "
                  "be ordered before their callers in the module";

      // Build instance inputs in createHWModule's order: per-arg — memref
      // args contribute a rd_data input, scalar args contribute the value
      // itself — then clk, rst, start. Track which caller memref each
      // memref-port row corresponds to for the output-extraction loop.
      builder.setInsertionPointToEnd(hwBody);
      SmallVector<Value> childInputs;
      SmallVector<PortArgInfo> calleeMemInfos;
      SmallVector<Value> calleeMemCallerArgs;
      for (auto operand : callOp.getOperands()) {
        if (auto memType = dyn_cast<MemRefType>(operand.getType())) {
          auto it = memPortMap.find(operand);
          if (it == memPortMap.end())
            return callOp.emitOpError(
                "memref operand has no backing port mapping (must be a "
                "caller memref arg or a local memref.alloc)");
          childInputs.push_back(it->second.rdData);
          calleeMemInfos.push_back(
              makePortArgInfoFromMemref(operand, memType,
                                         /*isLocalMem=*/false));
          calleeMemCallerArgs.push_back(operand);
        } else {
          Value v = mapping.lookupOrNull(operand);
          if (!v)
            v = operand;
          childInputs.push_back(v);
        }
      }
      childInputs.push_back(clk);
      childInputs.push_back(rst);
      childInputs.push_back(childStartSignals[childIndexForEntry[ei]]);

      auto childInst = hw::InstanceOp::create(
          builder, loc, calleeMod,
          builder.getStringAttr((calleeName + "_inst").str()),
          childInputs, nullptr);

      // Consume outputs in createHWModule's order: memref outputs per
      // call-operand (addr_*, rd_en?, wr_data, wr_en), then ready, done,
      // then any call results.
      unsigned outIdx = 0;
      for (auto [memIdx, info] : llvm::enumerate(calleeMemInfos)) {
        Value callerArg = calleeMemCallerArgs[memIdx];
        auto &mp = perEntryPorts[ei][callerArg];
        auto widths = ArrayRef<unsigned>(info.addrWidths);
        mp.addrs.assign(widths.size(), Value());
        for (unsigned d = 0; d < widths.size(); ++d)
          mp.addrs[d] = childInst.getResult(outIdx++);
        if (info.requiresRdEn)
          mp.rdEn = childInst.getResult(outIdx++);
        bool emitWrite = info.isAmcPort ? info.isWrite : true;
        if (emitWrite) {
          mp.wrData = childInst.getResult(outIdx++);
          mp.wrEn = childInst.getResult(outIdx++);
        }
      }
      // ready — unused by the caller FSM today; skip.
      outIdx++;
      Value childDone = childInst.getResult(outIdx++);
      childDoneBEs[childIndexForEntry[ei]].setValue(childDone);

      SmallVector<Value> callResultVals;
      for (auto result : callOp.getResults()) {
        Value v = childInst.getResult(outIdx++);
        mapping.map(result, v);
        callResultVals.push_back(v);
      }
      if (entry.launchOp) {
        if (auto launchAt =
                entry.launchOp->getParentOfType<LoopScheduleAtOp>()) {
          auto atYield = launchAt.getYieldOp();
          for (auto [atRes, yOperand] :
               llvm::zip(launchAt.getResults(), atYield.getOperands())) {
            if (yOperand == entry.launchOp.getHandle())
              funcHandleValueMap[atRes] = callResultVals;
          }
        }
        funcHandleValueMap[entry.launchOp.getHandle()] =
            std::move(callResultVals);
      }

    } else {
      // Leaf frame: lower body with entry_running as wrEn gate.
      Value oneGate = entryRunningSignals[ei];
      if (failed(lowerFrameBody(&topFrames[frameIdx].getBodyBlock(), builder,
                                mapping, ArrayRef<Value>(oneGate),
                                perEntryPorts[ei], enclosingModule, clk,
                                rst)))
        return failure();
    }

    // Register frame results only after the last entry for this frame
    // completes.
    if (ei == lastEntryForFrame[frameIdx]) {
      // Pre-scan for loads from local hlmem — skip capture for those.
      DenseSet<Value> localLoadResults;
      topFrames[frameIdx]->walk([&](LoopScheduleLoadOp loadOp) {
        for (auto &memInfo : memrefArgs)
          if (memInfo.originalArg == loadOp.getMemRef() &&
              memInfo.isLocalMem)
            localLoadResults.insert(loadOp.getResult());
      });

      // Forward each at's external results from its yield operands so the
      // frame body yield (whose operands reference at results) resolves.
      for (auto atOp :
           topFrames[frameIdx].getBodyBlock().getOps<LoopScheduleAtOp>()) {
        auto atYield = atOp.getYieldOp();
        for (auto [res, val] :
             llvm::zip(atOp.getResults(), atYield.getOperands())) {
          if (mapping.lookupOrNull(res))
            continue;
          if (auto m = mapping.lookupOrNull(val))
            mapping.map(res, m);
        }
      }

      auto frameYieldOp = cast<LoopScheduleYieldOp>(
          topFrames[frameIdx].getBodyBlock().getTerminator());
      // First: forward handle-typed frame results to any stashed values so
      // later frames that await on the frame's result can resolve the
      // underlying child's SSA outputs.
      for (auto [frameResult, yOperand] :
           llvm::zip(topFrames[frameIdx].getResults(),
                     frameYieldOp.getOperands())) {
        if (!isHandleType(frameResult.getType()))
          continue;
        auto it = funcHandleValueMap.find(yOperand);
        if (it != funcHandleValueMap.end())
          funcHandleValueMap[frameResult] = it->second;
      }
      for (auto [frameResult, yOperand] :
           llvm::zip(topFrames[frameIdx].getResults(),
                     frameYieldOp.getOperands())) {
        // Handles have no HW representation; skip entirely.
        if (isHandleType(frameResult.getType()))
          continue;
        Value val = mapping.lookup(yOperand);
        if (localLoadResults.count(yOperand)) {
          // Local hlmem read (latency=1) provides the delay; skip register.
          mapping.map(frameResult, val);
          continue;
        }
        auto regName = builder.getStringAttr(
            funcName + "_frame" + std::to_string(frameIdx) + "_result_" +
            std::to_string(frameResult.getResultNumber()));
        Value resetVal = createZeroConstant(builder, loc, val.getType());
        auto reg = seq::CompRegClockEnabledOp::create(
            builder, loc, val, clk, entryRunningSignals[ei], rst, resetVal,
            regName);
        mapping.map(frameResult, reg);
      }
    }
  }

  // Merge per-entry memory ports.
  builder.setInsertionPointToEnd(hwBody);
  DenseMap<Value, MemPortMapping> mergedMemPorts;
  for (auto &memInfo : memrefArgs)
    mergedMemPorts[memInfo.originalArg].rdData =
        memPortMap[memInfo.originalArg].rdData;
  mergeStepMemPorts(builder, loc, perEntryPorts, entryRunningSignals,
                    memrefArgs, mergedMemPorts);

  // Priority-compose any static at-body store drives (collected by
  // `cloneFrameAtBodies`) over the merged entry drives. Each static
  // drive's wrEn is already gated by its owning frame's header pulse,
  // so the static fires during that one cycle and the child's drives
  // take over for the rest of the frame.
  for (auto &[memVal, staticMp] : staticDrives) {
    auto &merged = mergedMemPorts[memVal];
    Value sGate = staticMp.wrEn;
    if (!sGate)
      continue;
    if (merged.wrEn) {
      merged.wrEn = comb::OrOp::create(builder, loc, sGate, merged.wrEn);
    } else {
      merged.wrEn = sGate;
    }
    if (staticMp.wrData) {
      if (merged.wrData)
        merged.wrData = comb::MuxOp::create(builder, loc, sGate,
                                              staticMp.wrData, merged.wrData);
      else
        merged.wrData = staticMp.wrData;
    }
    for (auto [d, sAddr] : llvm::enumerate(staticMp.addrs)) {
      if (!sAddr)
        continue;
      if (d >= merged.addrs.size())
        merged.addrs.resize(d + 1);
      if (merged.addrs[d])
        merged.addrs[d] = comb::MuxOp::create(builder, loc, sGate, sAddr,
                                                merged.addrs[d]);
      else
        merged.addrs[d] = sAddr;
    }
  }

  // Copy merged ports to function-level memPortMap.
  for (auto &memInfo : memrefArgs) {
    memPortMap[memInfo.originalArg].addrs =
        mergedMemPorts[memInfo.originalArg].addrs;
    memPortMap[memInfo.originalArg].wrData =
        mergedMemPorts[memInfo.originalArg].wrData;
    memPortMap[memInfo.originalArg].wrEn =
        mergedMemPorts[memInfo.originalArg].wrEn;
  }

  // Resolve hlmem + amc.instance backedges left dangling by the prelude.
  resolveFunctionMemoryBackedges(builder, loc, hwBody, hlmemBEs, memInstState,
                                  memPortMap);

  // Build hw.output. Compute `ready` as "no transaction in flight": a
  // sticky bit set on `start` and cleared on `done`. Sequential funcs
  // accept exactly one transaction at a time, so this matches the FSM's
  // IDLE-vs-RUNNING distinction without having to plumb a state-decoded
  // signal out of the FSM machine.
  builder.setInsertionPointToEnd(hwBody);
  Value falseConstSeq =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
  BackedgeBuilder readyBB(builder, loc);
  Backedge inFlightNextBE = readyBB.get(builder.getI1Type());
  auto inFlightReg = seq::CompRegOp::create(
      builder, loc, Value(inFlightNextBE), clk, rst, falseConstSeq,
      builder.getStringAttr("tx_in_flight"));
  Value inFlight = inFlightReg;
  Value notDoneSeq = comb::createOrFoldNot(builder, loc, doneSignal);
  Value holdInFlight =
      comb::AndOp::create(builder, loc, inFlight, notDoneSeq);
  Value inFlightNext = comb::OrOp::create(builder, loc, start, holdInFlight);
  inFlightNextBE.setValue(inFlightNext);
  Value readySignal = comb::createOrFoldNot(builder, loc, inFlight);

  buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, readySignal,
                doneSignal);

  funcOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Function pipeline lowering (Phase 1: single-transaction, straight-line)
//===----------------------------------------------------------------------===//
//
// Lowers a `loopschedule.func_pipeline` to an `hw.module` whose internals
// are pipelined. Phase 1 restrictions:
//   - No nested loops in the body (enforced by SCFToLoopSchedule).
//   - No iter args (function pipelines don't loop back).
//   - No condition value (always-true; pipeline runs to completion).
//   - No dynamic-latency ops (no launch/expect handling).
//
// The hw.module presents the same start/done handshake as the sequential
// path so callers can ignore the difference. `done` pulses once when the
// last stage's CE fires for the (single) transaction. To support multi-
// transaction streaming the active register would need to track in-flight
// count instead of a single bit; that's Phase 2.
LogicalResult LoopScheduleToFSMPass::lowerFunction(
    loopschedule::LoopScheduleFuncPipelineOp funcOp) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();
  OpBuilder builder(ctx);

  analysis::OperatorLibraryAnalysis ola(funcOp);
  operatorLibrary = &ola;
  auto unbindLibrary =
      llvm::make_scope_exit([&] { operatorLibrary = nullptr; });

  IRMapping mapping;
  unsigned clkIdx, rstIdx, startIdx;
  auto hwMod = createHWModule(funcOp, builder, mapping, memPortMap, clkIdx,
                              rstIdx, startIdx);
  Block *hwBody = hwMod.getBodyBlock();
  Value clk = hwBody->getArgument(clkIdx);
  Value rst = hwBody->getArgument(rstIdx);
  Value start = hwBody->getArgument(startIdx);

  instanceUniquer.clear();
  auto enclosingModule = funcOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToEnd(hwBody);

  // --- Prelude: clone non-stage ops, set up local hlmems + amc instances ---
  auto i1 = builder.getI1Type();
  BackedgeBuilder funcBB(builder, loc);
  SmallVector<HLMemBackedges> hlmemBEs;
  SymbolTable modSymTab(enclosingModule);
  loopschedule::HWMemoryLoweringState memInstState(clk, rst, funcBB,
                                                    modSymTab);
  SmallVector<PortArgInfo> memrefArgs;
  if (failed(setupFunctionPrelude(
          funcOp.getBody().front(), funcOp.getArguments(), builder, loc, clk,
          rst, enclosingModule,
          [](Operation *op) { return isa<LoopScheduleAtOp>(op); }, mapping,
          funcBB, memPortMap, hlmemBEs, memInstState, memrefArgs)))
    return failure();

  // --- Collect stages ---
  SmallVector<LoopScheduleAtOp> stages;
  for (auto &op : funcOp.getBody().front())
    if (auto stageOp = dyn_cast<LoopScheduleAtOp>(&op))
      stages.push_back(stageOp);

  // No stages: degenerate case (constant return). Pipe start through to done
  // by one register cycle so the handshake still has a stable shape. Ready
  // is always high — there's no internal delay to back-pressure on.
  if (stages.empty()) {
    builder.setInsertionPointToEnd(hwBody);
    Value zero = hw::ConstantOp::create(builder, loc, i1, 0);
    Value oneReady =
        hw::ConstantOp::create(builder, loc, i1, 1);
    auto doneReg = seq::CompRegOp::create(
        builder, loc, start, clk, rst, zero,
        builder.getStringAttr("pipe_done"));
    buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, oneReady,
                  doneReg);
    funcOp.erase();
    return success();
  }

  // --- Pipeline machinery ---
  builder.setInsertionPointToEnd(hwBody);
  Value falseConst = hw::ConstantOp::create(builder, loc, i1, 0);

  BackedgeBuilder bb(builder, loc);

  // Phase 2 multi-transaction streaming. A modulo-II counter tracks the
  // II window; caller may assert `start` whenever `ready = (counter == 0)`.
  // The counter is *not* free-running — it sits at 0 until a start is
  // accepted, then runs 0 → 1 → ... → II-1 → 0, after which the next
  // start can fire. This way the existing single-shot testbench (which
  // doesn't yet honor `ready`) still works for II ≥ 1, and a later
  // multi-transaction testbench just needs to wait for `ready` between
  // start pulses.
  uint64_t II = funcOp.getII();
  Value readySignal;
  Value ceGen;
  if (II <= 1) {
    readySignal = hw::ConstantOp::create(builder, loc, i1, 1);
    ceGen = start;
  } else {
    unsigned counterWidth = llvm::Log2_64_Ceil(II);
    Type counterType = IntegerType::get(ctx, counterWidth);
    Value cZero = hw::ConstantOp::create(builder, loc, counterType, 0);
    Value cOne = hw::ConstantOp::create(builder, loc, counterType, 1);
    Value cIIMinusOne =
        hw::ConstantOp::create(builder, loc, counterType, II - 1);

    Backedge counterBE = bb.get(counterType);
    auto counterReg = seq::CompRegOp::create(
        builder, loc, Value(counterBE), clk, rst, cZero,
        builder.getStringAttr("pipe_ii_counter"));

    Value isZero = comb::ICmpOp::create(
        builder, loc, comb::ICmpPredicate::eq, counterReg, cZero);
    Value counterPlusOne =
        comb::AddOp::create(builder, loc, counterReg, cOne);
    Value atMax = comb::ICmpOp::create(
        builder, loc, comb::ICmpPredicate::eq, counterReg, cIIMinusOne);
    Value wrapped =
        comb::MuxOp::create(builder, loc, atMax, cZero, counterPlusOne);

    // Advance only when not idle, OR when accepting a new start.
    Value advanceFromIdle =
        comb::AndOp::create(builder, loc, isZero, start);
    Value advancingMidWindow =
        comb::createOrFoldNot(builder, loc, isZero);
    Value advancing =
        comb::OrOp::create(builder, loc, advanceFromIdle, advancingMidWindow);
    Value counterNext =
        comb::MuxOp::create(builder, loc, advancing, wrapped, cZero);
    counterBE.setValue(counterNext);

    readySignal = isZero;
    ceGen = advanceFromIdle; // = start & ready
  }

  // Stage CE per absolute offset. Stages may live at sparse offsets (0, 1,
  // 4, ...) when multi-cycle ops cause the scheduler to skip cycles. Build
  // a shift register of length `maxOffset + 1` and tap each stage by its
  // own offset; otherwise an N-stage chain would fire the Nth tap on cycle
  // N rather than cycle stages[N].offset.
  unsigned maxOffset = 0;
  for (auto stage : stages)
    maxOffset = std::max(maxOffset, (unsigned)stage.getOffset());

  SmallVector<Value> ceShift(maxOffset + 1);
  ceShift[0] = ceGen;
  for (unsigned i = 1; i <= maxOffset; ++i) {
    auto ceName = builder.getStringAttr(
        "pipe_ce_off_" + std::to_string(i));
    ceShift[i] = seq::CompRegOp::create(
        builder, loc, ceShift[i - 1], clk, rst, falseConst, ceName);
  }

  SmallVector<Value> stageCE(stages.size());
  for (auto [i, stage] : llvm::enumerate(stages))
    stageCE[i] = ceShift[(unsigned)stage.getOffset()];

  // Per-stage memory port mappings.
  SmallVector<DenseMap<Value, MemPortMapping>> perStagePorts(stages.size());
  for (unsigned s = 0; s < stages.size(); ++s)
    for (auto &entry : memPortMap)
      perStagePorts[s][entry.first].rdData = entry.second.rdData;

  // Cross-stage delay registers — see CrossStageValueResolver above.
  CrossStageValueResolver delayResolver(
      builder, loc, hwBody, clk, rst, stages, stageCE, mapping,
      operatorLibrary, "pipe");

  auto isLocalMemref = [&](Value memref) -> bool {
    for (auto &memInfo : memrefArgs)
      if (memInfo.originalArg == memref && memInfo.isLocalMem)
        return true;
    return false;
  };

  // Per-memref max read-stage offset. The testbench uses this to time its
  // per-transaction read shift correctly under multi-transaction streaming
  // (Phase 2). Without it, the TB would have to assume every memory is
  // read at stage 0, which fails any kernel whose II ≥ 2 reads at later
  // stages (e.g. the addmul kernel reads at stages 0 and 1).
  DenseMap<Value, unsigned> memMaxReadStageOff;

  // --- Stage emission: clone ops, build cross-stage registers ---
  for (auto [stageIdx, stageOp] : llvm::enumerate(stages)) {
    Block &body = stageOp.getBodyBlock();
    builder.setInsertionPointToEnd(hwBody);

    DenseSet<Value> localLoadResults;

    for (auto &op : body.getOperations()) {
      if (isa<LoopScheduleYieldOp>(&op))
        continue;

      SmallVector<std::pair<Value, Value>> savedMappings;
      for (Value operand : op.getOperands()) {
        Value delayed = delayResolver.resolveForStage(operand, stageIdx);
        if (!delayed)
          continue;
        Value current = mapping.lookup(operand);
        if (delayed != current) {
          savedMappings.emplace_back(operand, current);
          mapping.map(operand, delayed);
        }
      }

      std::function<LogicalResult(Operation *, Value)> processOp =
          [&](Operation *inner, Value gate) -> LogicalResult {
        if (isa<LoopScheduleYieldOp>(inner))
          return success();
        if (auto ifOp = dyn_cast<LoopScheduleIfOp>(inner)) {
          Value cond = mapping.lookup(ifOp.getCond());
          Value innerGate =
              comb::AndOp::create(builder, loc, gate, cond);
          for (auto &nested : ifOp.getBody().front()) {
            if (auto yieldOp = dyn_cast<LoopScheduleYieldOp>(&nested)) {
              for (auto [res, val] :
                   llvm::zip(ifOp.getResults(), yieldOp.getOperands()))
                mapping.map(res, mapping.lookup(val));
              continue;
            }
            if (failed(processOp(&nested, innerGate)))
              return failure();
          }
          return success();
        }
        if (auto storeOp = dyn_cast<HWStoreLoweringInterface>(inner))
          return handleHWStore(storeOp, builder, mapping, gate,
                               perStagePorts[stageIdx]);
        if (auto loadOp = dyn_cast<HWLoadLoweringInterface>(inner)) {
          if (auto ls = dyn_cast<LoopScheduleLoadOp>(inner))
            if (isLocalMemref(ls.getMemRef()))
              localLoadResults.insert(ls.getResult());
          if (loadOp.requiresReadEnable() && loadOp.getReadLatency() > 0)
            localLoadResults.insert(loadOp.getResult());
          // Track the latest stage offset that issues a read against
          // each memref. The TB indexes its read-shift register chain
          // by this offset.
          unsigned offset = (unsigned)stages[stageIdx].getOffset();
          Value memVal = loadOp.getMemoryValue();
          auto it = memMaxReadStageOff.find(memVal);
          if (it == memMaxReadStageOff.end() || it->second < offset)
            memMaxReadStageOff[memVal] = offset;
          return handleHWLoad(loadOp, builder, mapping,
                              perStagePorts[stageIdx], gate);
        }
        return emitComputeOp(inner, builder, mapping, enclosingModule, clk,
                             rst);
      };

      LogicalResult opResult = processOp(&op, stageCE[stageIdx]);
      for (auto &sm : savedMappings)
        mapping.map(sm.first, sm.second);
      if (failed(opResult))
        return failure();
    }

    auto regOp = cast<LoopScheduleYieldOp>(body.getTerminator());
    for (auto [regIdx, val] : llvm::enumerate(regOp.getOperands())) {
      Value delayed = delayResolver.resolveForStage(val, stageIdx);
      Value mappedVal = delayed ? delayed : mapping.lookup(val);

      if (localLoadResults.count(val)) {
        mapping.map(stageOp.getResult(regIdx), mappedVal);
        continue;
      }
      if (computeOpCycleLatency(val.getDefiningOp(), operatorLibrary) > 0) {
        mapping.map(stageOp.getResult(regIdx), mappedVal);
        continue;
      }

      Value resetVal = createZeroConstant(builder, loc, mappedVal.getType());
      auto regName = builder.getStringAttr(
          "pipe_s" + std::to_string(stageIdx) + "_r" +
          std::to_string(regIdx));
      Value reg = seq::CompRegClockEnabledOp::create(
          builder, loc, mappedVal, clk, stageCE[stageIdx], rst, resetVal,
          regName);
      mapping.map(stageOp.getResult(regIdx), reg);
    }
  }

  builder.setInsertionPointToEnd(hwBody);

  muxStageMemPorts(builder, loc, perStagePorts, stageCE, memPortMap);

  // Annotate per-memref max read stage offset (in mem0/mem1/... port
  // order) so the testbench can build a per-memory read-shift chain.
  // Memrefs that aren't read at all get 0 (the chain[0] register matches
  // the just-issued txn).
  {
    SmallVector<int64_t> readStageOffsets;
    readStageOffsets.reserve(memrefArgs.size());
    for (auto &memInfo : memrefArgs) {
      auto it = memMaxReadStageOff.find(memInfo.originalArg);
      readStageOffsets.push_back(
          it == memMaxReadStageOff.end() ? 0 : (int64_t)it->second);
    }
    hwMod->setAttr("loopschedule.mem_read_stages",
                    builder.getI64ArrayAttr(readStageOffsets));
  }

  // Done. The last stage's CE pulses on the same cycle as the last store
  // commits (writes are non-blocking and only visible the following cycle),
  // so the testbench would observe pre-write memory if we exposed the raw
  // CE. Register `done` for one extra cycle so consumers see it after all
  // writes have settled — same trick the loop-pipeline path uses.
  Value doneSignal = seq::CompRegOp::create(
      builder, loc, stageCE.back(), clk, rst, falseConst,
      builder.getStringAttr("pipe_done"));

  // Resolve hlmem + amc.instance backedges left dangling by the prelude.
  resolveFunctionMemoryBackedges(builder, loc, hwBody, hlmemBEs, memInstState,
                                  memPortMap);

  builder.setInsertionPointToEnd(hwBody);
  buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, readySignal,
                doneSignal);

  funcOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

/// Rewrite returned memref.alloc values into plain function arguments so they
/// appear as regular mem<i> testbench ports instead of becoming internal
/// seq.hlmem instances. The testbench harness (and the Allo Python driver)
/// discovers output memories by their port index, so returned allocs must sit
/// alongside the input memref args in the final hw.module port list.
template <typename FuncOpT>
static void hoistReturnedAllocsToArgs(FuncOpT funcOp) {
  if (funcOp.getBody().empty())
    return;
  auto *block = &funcOp.getBody().front();
  auto returnOp = dyn_cast<loopschedule::LoopScheduleReturnOp>(block->getTerminator());
  if (!returnOp)
    return;
  // Collect returned memref allocs and their operand indices.
  SmallVector<memref::AllocOp> hoisted;
  SmallVector<unsigned> droppedReturnIdxs;
  for (auto [idx, retVal] : llvm::enumerate(returnOp.getOperands())) {
    if (!isa<MemRefType>(retVal.getType()))
      continue;
    auto allocOp = retVal.template getDefiningOp<memref::AllocOp>();
    if (!allocOp)
      continue;
    hoisted.push_back(allocOp);
    droppedReturnIdxs.push_back(idx);
  }
  if (hoisted.empty())
    return;
  // Append a new block argument for each hoisted alloc, replace uses, and
  // erase the alloc.
  auto funcType = funcOp.getFunctionType();
  SmallVector<Type> newInputs(funcType.getInputs().begin(),
                              funcType.getInputs().end());
  for (auto allocOp : hoisted) {
    Type memTy = allocOp.getType();
    newInputs.push_back(memTy);
    auto newArg = block->addArgument(memTy, allocOp.getLoc());
    allocOp.getResult().replaceAllUsesWith(newArg);
    allocOp.erase();
  }
  // Drop returned operands (in descending order so indices stay valid).
  SmallVector<Value> newReturnOperands(returnOp.getOperands().begin(),
                                       returnOp.getOperands().end());
  for (int i = (int)droppedReturnIdxs.size() - 1; i >= 0; --i)
    newReturnOperands.erase(newReturnOperands.begin() + droppedReturnIdxs[i]);
  OpBuilder b(returnOp);
  auto newReturn = loopschedule::LoopScheduleReturnOp::create(b, returnOp.getLoc(),
                                            newReturnOperands);
  (void)newReturn;
  returnOp.erase();
  // Update the function type.
  SmallVector<Type> newResults(funcType.getResults().begin(),
                               funcType.getResults().end());
  for (int i = (int)droppedReturnIdxs.size() - 1; i >= 0; --i)
    newResults.erase(newResults.begin() + droppedReturnIdxs[i]);
  funcOp.setType(
      FunctionType::get(funcOp.getContext(), newInputs, newResults));
}

/// Flatten multi-dim memref arguments (and local allocs) to 1-D so that the
/// hw.module / testbench contract only ever deals with a single `mem<i>_addr`
/// port. Upstream `flatten-memref` only knows about memref.load/store, not
/// loopschedule.load/store, so we do the rewrite here where we have full
/// ownership of the operations that actually touch the memref.
template <typename FuncOpT>
static LogicalResult flattenMultiDimMemrefs(FuncOpT funcOp) {
  if (funcOp.getBody().empty())
    return success();
  auto *block = &funcOp.getBody().front();
  auto *ctx = funcOp.getContext();

  SmallVector<Value> toFlatten;
  for (auto arg : block->getArguments()) {
    if (auto mt = dyn_cast<MemRefType>(arg.getType()))
      if (mt.getRank() > 1)
        toFlatten.push_back(arg);
  }
  funcOp.walk([&](memref::AllocOp allocOp) {
    auto mt = allocOp.getType();
    if (mt.getRank() > 1)
      toFlatten.push_back(allocOp.getResult());
  });
  if (toFlatten.empty())
    return success();

  auto linearize = [&](OpBuilder &b, Location loc, ValueRange indices,
                       ArrayRef<int64_t> shape) -> Value {
    int64_t total = 1;
    for (auto d : shape)
      total *= d;
    unsigned bits =
        std::max<unsigned>(1, llvm::Log2_64_Ceil(std::max<int64_t>(total, 2)));
    Type idxType = IntegerType::get(ctx, bits);
    Value acc =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(idxType, 0));
    for (size_t i = 0; i < indices.size(); ++i) {
      int64_t tail = 1;
      for (size_t j = i + 1; j < shape.size(); ++j)
        tail *= shape[j];
      Value idx = indices[i];
      auto idxTy = cast<IntegerType>(idx.getType());
      if (idxTy.getWidth() < bits)
        idx = arith::ExtUIOp::create(b, loc, idxType, idx);
      else if (idxTy.getWidth() > bits)
        idx = arith::TruncIOp::create(b, loc, idxType, idx);
      if (tail > 1) {
        Value tailConst = arith::ConstantOp::create(
            b, loc, b.getIntegerAttr(idxType, tail));
        idx = arith::MulIOp::create(b, loc, idx, tailConst);
      }
      acc = arith::AddIOp::create(b, loc, acc, idx);
    }
    return acc;
  };

  for (Value v : toFlatten) {
    auto oldType = cast<MemRefType>(v.getType());
    auto shape = oldType.getShape();
    int64_t total = 1;
    for (auto d : shape) {
      if (d == ShapedType::kDynamic)
        return funcOp.emitError(
            "cannot flatten memref with dynamic shape for FSM backend");
      total *= d;
    }
    auto newType = MemRefType::get({total}, oldType.getElementType());

    SmallVector<Operation *> users(v.getUsers().begin(), v.getUsers().end());
    v.setType(newType);
    for (auto *user : users) {
      OpBuilder b(user);
      if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(user)) {
        Value linear = linearize(b, loadOp.getLoc(), loadOp.getIndices(), shape);
        auto newLoad = LoopScheduleLoadOp::create(
            b, loadOp.getLoc(), oldType.getElementType(), v,
            ValueRange{linear});
        loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
        loadOp.erase();
      } else if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(user)) {
        Value linear =
            linearize(b, storeOp.getLoc(), storeOp.getIndices(), shape);
        LoopScheduleStoreOp::create(b, storeOp.getLoc(),
                                    storeOp.getValueToStore(), v,
                                    ValueRange{linear});
        storeOp.erase();
      } else {
        return user->emitError(
            "unsupported user of multi-dim memref during FSM flatten");
      }
    }
  }

  // Update the function signature: any block args we retyped need to be
  // reflected in the function type.
  SmallVector<Type> newInputs;
  for (auto arg : block->getArguments())
    newInputs.push_back(arg.getType());
  funcOp.setType(FunctionType::get(ctx, newInputs,
                                   funcOp.getFunctionType().getResults()));
  return success();
}

void LoopScheduleToFSMPass::runOnOperation() {
  auto moduleOp = getOperation();

  SmallVector<loopschedule::LoopScheduleFuncSequentialOp> seqFuncs;
  SmallVector<loopschedule::LoopScheduleFuncPipelineOp> pipeFuncs;
  moduleOp.walk(
      [&](loopschedule::LoopScheduleFuncSequentialOp f) { seqFuncs.push_back(f); });
  moduleOp.walk(
      [&](loopschedule::LoopScheduleFuncPipelineOp f) { pipeFuncs.push_back(f); });

  for (auto funcOp : seqFuncs)
    hoistReturnedAllocsToArgs(funcOp);
  for (auto funcOp : pipeFuncs)
    hoistReturnedAllocsToArgs(funcOp);

  if (!disableFlattenMemrefs) {
    for (auto funcOp : seqFuncs) {
      if (failed(flattenMultiDimMemrefs(funcOp))) {
        signalPassFailure();
        return;
      }
    }
    for (auto funcOp : pipeFuncs) {
      if (failed(flattenMultiDimMemrefs(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

  for (auto funcOp : seqFuncs) {
    if (failed(lowerFunction(funcOp))) {
      signalPassFailure();
      return;
    }
  }
  for (auto funcOp : pipeFuncs) {
    if (failed(lowerFunction(funcOp))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
circt::createLoopScheduleToFSMPass() {
  return std::make_unique<LoopScheduleToFSMPass>();
}

void circt::registerLoopScheduleToFSM() {
  // Defined out-of-line to keep the registration in a single TU; the
  // generated GEN_PASS_REGISTRATION_LOOPSCHEDULETOFSM inline function
  // collides with `circt::registerCIRCTConversionPasses()` if exposed
  // from the public header.
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> {
        return circt::createLoopScheduleToFSMPass();
      });
}
