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
#include <limits>

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
  // Separate write-completion pulse for read+write dynamic ports (see
  // HWPortSignals::wrDone). When non-null, `done` is read-only and store
  // expects attribute against `wrDone` instead.
  Value wrDone;
  // Memory-driven same-cycle acceptance level for dynamic ports (see
  // HWPortSignals::ready). Null ⇒ tied high (no issue backpressure).
  Value ready;
  // Memory-driven write-/read-drain levels for posted AXI faces (see
  // HWPortSignals::wrIdle/rdIdle). A fenced load ANDs `wrIdle` into its
  // rd_en (RAW); a fenced store ANDs `rdIdle` into its wr_en (WAR). Null ⇒
  // tied high (fence is a no-op for non-posted ports).
  Value rdIdle;
  Value wrIdle;
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
  // True iff the port exposes a `ready` acceptance level (dynamic amc
  // ports); plumbed through loop modules like `done`.
  bool hasReady = false;
  // True iff the port exposes a completion `done` / split write-completion
  // `wr_done` (rw dyn faces); plumbed through loop modules. Historically
  // `done` threading was gated on isRead, starving write-only and rw
  // ports' completion handshakes inside loop modules.
  bool hasDone = false;
  bool hasWrDone = false;
  // True iff the port exposes write-/read-drain levels (`wr_idle`/`rd_idle`,
  // posted AXI faces); plumbed through loop modules like `ready` so a fenced
  // access inside a loop can gate its rd_en/wr_en on the drain.
  bool hasRdIdle = false;
  bool hasWrIdle = false;
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
  // External BRAM-style ports carry a read enable so the memory's output
  // register HOLDS the last read value between reads (like ram_1rw's
  // content_en) instead of tracking the address bus every cycle. Local
  // hlmems are internal seq.hlmem instances with no module-level port.
  info.requiresRdEn = !isLocalMem;
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

/// Is `v` (a loop result or a value forwarded through the launch/at/frame
/// yield plumbing) ever actually consumed? Values threaded through yields
/// reach consumers only via an await WITH results (handleValueMap) or the
/// enclosing terminator; a pure-ordering `loopschedule.await` (no results)
/// does not read them. Used to decide whether a loop module may cut its
/// done output through on the final advance edge: that edge is the SAME
/// posedge the loop's result registers latch, so a parent latching a value
/// derived from those results on its own advance edge would read the
/// pre-edge (stale) value — only loops whose results are dead may cut
/// through. Conservative: any unrecognized use counts as consumed.
static bool loopScheduleValueConsumed(Value v, unsigned depth = 0) {
  if (depth > 16)
    return true;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (isa<LoopScheduleYieldOp>(user)) {
      Operation *parent = user->getParentOp();
      unsigned idx = use.getOperandNumber();
      if (auto launch = dyn_cast<LoopScheduleLaunchOp>(parent)) {
        if (loopScheduleValueConsumed(launch.getHandle(), depth + 1))
          return true;
        continue;
      }
      if (auto at = dyn_cast<LoopScheduleAtOp>(parent)) {
        if (idx < at->getNumResults() &&
            loopScheduleValueConsumed(at->getResult(idx), depth + 1))
          return true;
        continue;
      }
      if (auto frame = dyn_cast<LoopScheduleFrameOp>(parent)) {
        if (idx < frame->getNumResults() &&
            loopScheduleValueConsumed(frame->getResult(idx), depth + 1))
          return true;
        continue;
      }
      return true;
    }
    if (auto await = dyn_cast<LoopScheduleAwaitOp>(user)) {
      if (await->getNumResults() > 0)
        return true; // await-with-value delivers the results
      continue;      // pure ordering await
    }
    return true; // terminator, arith use, anything else — consumed
  }
  return false;
}

/// May this loop's FSM latch its iter_arg inits on the START edge (the
/// IDLE/DONE entry bypass) instead of one cycle later in COND? Unsafe when
/// an init is produced by an at-op in the SAME frame that launches the
/// loop and its capture register only becomes readable after the launch
/// cycle — e.g. an accumulator init loaded from memory at at-0 with the
/// launch at at-1: the capture register latches on the same posedge the
/// start pulse fires, so an entry-edge init latch would read the PREVIOUS
/// invocation's value. Such loops keep the COND entry state (one settle
/// cycle). Inits that are constants, block args, or values from earlier
/// frames are stable well before the start pulse.
static bool entryInitsReadyAtStart(LoopScheduleSequentialOp seqOp) {
  auto launch = seqOp->getParentOfType<LoopScheduleLaunchOp>();
  if (!launch)
    return true; // top-level loop: inits are function-scope values
  auto launchAt = launch->getParentOfType<LoopScheduleAtOp>();
  unsigned launchOffset = launchAt ? (unsigned)launchAt.getOffset() : 0;
  auto frame = launch->getParentOfType<LoopScheduleFrameOp>();
  if (!frame)
    return true;
  for (Value init : seqOp.getInits()) {
    Operation *def = init.getDefiningOp();
    if (!def || !frame->isAncestor(def))
      continue; // stable before the frame started
    auto defAt = dyn_cast<LoopScheduleAtOp>(def);
    if (!defAt)
      return false; // unmodeled same-frame producer — be conservative
    Value inner = defAt.getYieldOp()
                      .getOperands()[cast<OpResult>(init).getResultNumber()];
    Operation *producer = inner.getDefiningOp();
    unsigned lat = 1;
    if (producer)
      lat = isa<loopschedule::HWLoadLoweringInterface>(producer)
                ? 1
                : getOpCycleLatency(producer);
    // Capture register readable the cycle AFTER offset+latency.
    if ((unsigned)defAt.getOffset() + lat + 1 > launchOffset)
      return false;
  }
  return true;
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
  // The cross-stage resolver's convention is "data live during stage
  // J + latency + 1" (a latency-1 BRAM read issued at J is live at J+1,
  // like a latency-0 value's stage register). A dynamic load's expect
  // FIFO presents the data live on the consume cycle J + declared
  // latency, so report declaredLatency - 1: the consumer at the expect's
  // dest stage reads the FIFO head directly, and any later consumer gets
  // a delay register that captures the head at the consume edge.
  if (auto load = dyn_cast<LoadInterface>(def))
    if (load.isDynamic()) {
      unsigned lat = load.getLatency();
      return lat > 0 ? lat - 1 : 0;
    }
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

/// Value-sensitive variant of `computeOpCycleLatency`: looks through
/// `loopschedule.if` wrappers to the op that actually produces the value.
/// Predication gates side effects but adds no cycles, so an if-wrapped
/// multi-cycle operator's result is live exactly when the bare operator's
/// would be. Without the look-through, an if-wrapped latency-L result
/// reports 0 and picks up a spurious stage register on top of the
/// operator's internal pipeline, skewing every downstream consumer by one
/// cycle (first seen when loop flattening began predicating compute
/// epilogues, e.g. gesummv's `y[i] = alpha*accA + beta*accB`).
static unsigned
computeValueCycleLatency(Value v,
                         analysis::OperatorLibraryAnalysis *operatorLibrary) {
  Operation *def = v.getDefiningOp();
  while (auto ifOp = dyn_cast_or_null<LoopScheduleIfOp>(def)) {
    auto yieldOp =
        dyn_cast<LoopScheduleYieldOp>(ifOp.getBody().front().getTerminator());
    if (!yieldOp)
      return 0;
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (idx >= yieldOp.getNumOperands())
      return 0;
    v = yieldOp.getOperand(idx);
    def = v.getDefiningOp();
  }
  return computeOpCycleLatency(def, operatorLibrary);
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
    return computeValueCycleLatency(yieldOp.getOperand(idx), operatorLibrary);
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

/// One extern operator instance shared by several scheduled ops that the
/// binding pass proved can never issue in the same cycle (same
/// `loopschedule.operator` and `loopschedule.binding`). The first user
/// creates the instance; each subsequent user rewrites the instance's
/// operand ports to `mux(userGate, userOperand, previousDrive)` and
/// narrows the CE port to the AND of the users' CEs (hw.module bodies
/// are graph regions, so the later-defined muxes may legally drive the
/// earlier instance).
struct SharedOperatorInstance {
  hw::InstanceOp instOp;
  /// Instance input index for the extern port carrying `oplib.operand j`
  /// (index j into this vector), -1 when the operator has no such port.
  SmallVector<int> operandPortForIdx;
  /// Instance input index of the `oplib.enable` port, -1 if none.
  int ceInputIdx = -1;
  /// Current CE drive (ANDed as users join).
  Value ce;
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
  /// `launchStartOffsets[j] <= launchAtOffsets[j]` is the cycle where
  /// launch j's child_start actually pulses (the early-start peephole may
  /// pull a pipeline launch ahead of its scheduled at-offset); the frame's
  /// cycle-state count still covers the SCHEDULED offset so static-op
  /// issue/capture cycles are untouched.
  fsm::MachineOp createSequentialFSM(OpBuilder &builder, Location loc,
                                     StringRef fsmName, unsigned numFrames,
                                     ArrayRef<unsigned> waitFrameIndices,
                                     ArrayRef<unsigned> launchAtOffsets,
                                     ArrayRef<unsigned> launchStartOffsets,
                                     ArrayRef<unsigned> frameLatencies,
                                     bool foldLastFrame, bool condBypass,
                                     bool entryBypass);

  /// Create linear run-once FSM for sequencing top-level function frames.
  /// frameChildKind[i]: -1 = leaf, 0 = sequential child, 1 = pipeline child,
  /// -2 = stateless (an await-only frame: its ordering is already enforced
  /// by the preceding WAIT, so it gets no state — its frame_running output
  /// is declared but never driven).
  /// entryLatencies[i] is the leaf entry's frame latency: leaves with
  /// latency L > 1 hold their state for L cycles and expose per-cycle
  /// outputs appended after frame_running_*; entryCycleOutBase[i] receives
  /// each entry's base index into that region (-1 = single-cycle).
  ///
  /// entryGroup[i] identifies the frame each entry came from; entries of
  /// one group are CONCURRENT siblings (multiple launches of one frame).
  /// A group shares one FRAME/WAIT state pair: FRAME pulses every
  /// member's child_start together and WAIT holds until the group's
  /// combined done input (one FSM input per waiting group, built by the
  /// caller as the all-members-done conjunction) rises. Groups must be
  /// contiguous and ascending in entry order. Singleton groups reproduce
  /// the historical per-entry chain exactly.
  fsm::MachineOp createFunctionFSM(OpBuilder &builder, Location loc,
                                    StringRef fsmName,
                                    ArrayRef<int> frameChildKind,
                                    ArrayRef<unsigned> entryLatencies,
                                    ArrayRef<unsigned> entryGroup,
                                    SmallVectorImpl<int> &entryCycleOutBase);

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
                               ModuleOp moduleOp, Value clk, Value rst,
                               struct SeqDynCtx *seqDyn = nullptr,
                               struct SeqPortMuxCtx *seqMux = nullptr,
                               Value opCE = {});

  /// Materialize a single compute op via the operator-library dispatch.
  /// Used by every internal cloning site that previously called
  /// `builder.clone(op, mapping)` for arith ops. `op` is the original op
  /// (not yet cloned). On success, populates `mapping` for the original
  /// op's results.
  LogicalResult emitComputeOp(Operation *op, OpBuilder &builder,
                              IRMapping &mapping, ModuleOp moduleOp,
                              Value clk, Value rst, Value opCE = {},
                              Value shareGate = {});

  /// `lowerAtBody` is a method (not a free function) so it can access
  /// `operatorLibrary` and `instanceUniquer` directly.
  LogicalResult lowerAtBody(Block *body, OpBuilder &builder,
                            IRMapping &mapping, ArrayRef<Value> cycleGates,
                            unsigned baseCycle,
                            DenseMap<Value, MemPortMapping> &memPorts,
                            ModuleOp moduleOp, Value clk, Value rst,
                            struct SeqDynCtx *seqDyn = nullptr,
                            struct SeqPortMuxCtx *seqMux = nullptr,
                            Value opCE = {});

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

  /// Extern operator instances shared by binding id within the hw.module
  /// currently being generated (reset alongside `instanceUniquer`). Ops
  /// stamped with the same (operator, `loopschedule.binding`) pair by the
  /// binding pass reuse one instance: each joining user's operands are
  /// muxed in under its activation gate, and the clock-enable narrows to
  /// the AND of the users' CEs. The AND is sound because every stall term
  /// in a module is gated on its own frame/stage being active, so a
  /// non-owning user's CE reads idle-high — and cross-frame users are
  /// separated by full pipeline drains (frame WAIT semantics), so at most
  /// one user ever has values in flight.
  llvm::DenseMap<std::pair<mlir::StringAttr, int64_t>, SharedOperatorInstance>
      sharedOperators;
};

//===----------------------------------------------------------------------===//
// Load/store helpers
//===----------------------------------------------------------------------===//

static bool isSeqDynAccess(Operation *op);

/// Context for lowering sequential frames that contain MORE THAN ONE static
/// access to the same memory port. The frame's accesses share one set of
/// address/write-data/write-enable slots in `MemPortMapping`; without this
/// context each access overwrites the previous one's drives (last writer
/// wins), so only the final access's address ever reaches the port and every
/// load result aliases the raw `rd_data` wire (caught by the fib kernel:
/// `A[i] = A[i-1] + A[i-2]` silently dropped all loop stores). With the
/// context, each access on a contended port muxes its drives in under its
/// own issue-cycle gate, write enables OR together, and each load's result
/// is captured into a register at its data-valid cycle (with a live-cycle
/// bypass) so later-cycle consumers read the held value.
///
/// Ports with a single static access are intentionally left on the direct
/// drive path: their address is stable for the whole frame, the raw
/// `rd_data` mapping is correct, and the generated HW stays identical to
/// what this pass emitted before the context existed.
struct SeqPortMuxCtx {
  /// (memory value, binding port) pairs with >1 static access in the frame.
  llvm::DenseSet<std::pair<Value, unsigned>> multi;
  /// Memories whose rd_data wire is registered: their data is valid the
  /// cycle AFTER issue. Under the BRAM-port contract this is every memory
  /// — local seq.hlmem and amc instance rams by construction, external
  /// memref ports because they present latency-1 registered reads (the
  /// testbench models them the same way). The set exists so a future
  /// combinational-read port kind can opt out.
  llvm::DenseSet<Value> registeredMems;
  /// Per-cycle gates of the enclosing frame (frame_cycle_<i>_<c>).
  ArrayRef<Value> cycleGates;
  /// Issue cycle of the access currently being lowered (set by the caller
  /// before each handleHWLoad/handleHWStore call).
  unsigned cycle = 0;
  Value clk;
  Value rst;
  std::string regPrefix;
  unsigned counter = 0;
};

/// Populate `multi` with the (memory value, binding port) pairs that have
/// more than one static (fixed-latency) access in `frameBody`. Dynamic
/// accesses go through the SeqDynCtx handshake path (which already
/// OR-accumulates its drives) and launched children lower in their own
/// modules, so both are excluded.
static void collectFrameAccessCounts(
    Block &frameBody,
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> &counts) {
  frameBody.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (isa<LoopScheduleLaunchOp, LoopScheduleSequentialOp,
            LoopSchedulePipelineOp>(op))
      return WalkResult::skip();
    if (isSeqDynAccess(op))
      return WalkResult::advance();
    if (auto loadOp = dyn_cast<loopschedule::HWLoadLoweringInterface>(op))
      ++counts[{loadOp.getMemoryValue(), getBindingPort(loadOp)}];
    else if (auto storeOp =
                 dyn_cast<loopschedule::HWStoreLoweringInterface>(op))
      ++counts[{storeOp.getMemoryValue(), getBindingPort(storeOp)}];
    return WalkResult::advance();
  });
}


/// Lower any `HWLoadLoweringInterface` op: drive the read addresses on
/// the memory port mapping (port selected by `loopschedule.binding`,
/// default 0) and map the load result to the port's read data.
static LogicalResult
handleHWLoad(loopschedule::HWLoadLoweringInterface loadOp, OpBuilder &builder,
             IRMapping &mapping,
             DenseMap<Value, MemPortMapping> &memPorts,
             Value rdEnGate = nullptr, SeqPortMuxCtx *seqMux = nullptr) {
  Location loc = loadOp->getLoc();
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
  bool contended =
      seqMux && seqMux->multi.contains({loadOp.getMemoryValue(), port});
  assert((!contended || rdEnGate) &&
         "contended-port load lowering requires the issue-cycle gate");
  pref.addrs->resize(widths.size());
  for (auto [d, idx] : llvm::enumerate(loadOp.getIndices())) {
    Value addr = mapping.lookup(idx);
    addr = resizeIntTo(builder, loc, addr, widths[d]);
    if (contended) {
      Value prev = (*pref.addrs)[d];
      if (!prev)
        prev = hw::ConstantOp::create(builder, loc, addr.getType(), 0);
      addr = comb::MuxOp::create(builder, loc, rdEnGate, addr, prev);
    }
    (*pref.addrs)[d] = addr;
  }
  // Drive the port's read enable with this load's issue gate. AMC ports
  // require it (assert below); memref ports use it for the BRAM-port
  // hold contract (emitted only when the port declares requiresRdEn —
  // local hlmem entries simply ignore the slot).
  assert((rdEnGate || !loadOp.requiresReadEnable()) &&
         "HW load requires an explicit read-enable gate");
  if (rdEnGate) {
    // NB: no RAW fence here (the sequential path folds the drain into `ready`
    // in lowerSeqDynAccess). Gating a pipelined load's rd_en on wr_idle
    // self-deadlocks: the stall freezes the whole pipeline, including the store
    // whose drain would raise wr_idle. A pipelined intra-loop AXI RAW is left
    // to program order / the engine committing at the W beat; a completion-
    // based fence would be needed to make it robust to a reordering slave.
    *pref.rdEn = (contended && *pref.rdEn)
                     ? (Value)comb::OrOp::create(builder, loc, *pref.rdEn,
                                                 rdEnGate)
                     : rdEnGate;
  }
  // On a contended port the raw rd_data wire only carries this load's data
  // during its data-valid cycle — afterwards the port serves the frame's
  // next access. Capture the data into a register on the valid cycle and
  // hand consumers a live-cycle bypass mux.
  if (contended) {
    unsigned wireLatency =
        seqMux->registeredMems.contains(loadOp.getMemoryValue()) ? 1 : 0;
    unsigned dataCycle = seqMux->cycle + wireLatency;
    Value liveGate;
    if (dataCycle < seqMux->cycleGates.size()) {
      liveGate = seqMux->cycleGates[dataCycle];
    } else {
      // The data-valid cycle falls past the enclosing frame's last cycle
      // (a registered read issued on the frame's final cycle): the FSM has
      // already advanced, but the BRAM holds this read's data until the
      // port's NEXT enabled read commits — which is at least one cycle
      // after any subsequent issue. A one-cycle-delayed copy of the ISSUE
      // gate therefore fires exactly within the data-valid window,
      // regardless of which frame the FSM is in by then. Falling through
      // to the raw rd_data wire here would silently hand cross-frame
      // consumers a later access's data.
      Value issueGate = seqMux->cycleGates[seqMux->cycle];
      Value zero1 =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
      auto gateName = builder.getStringAttr(
          seqMux->regPrefix + "_ldcapgate_" + std::to_string(seqMux->counter));
      liveGate = seq::CompRegOp::create(builder, loc, issueGate, seqMux->clk,
                                        seqMux->rst, zero1, gateName);
    }
    Value resetVal = createZeroConstant(builder, loc, pref.rdData->getType());
    auto regName = builder.getStringAttr(
        seqMux->regPrefix + "_ldcap_" + std::to_string(seqMux->counter++));
    Value captured = seq::CompRegClockEnabledOp::create(
        builder, loc, *pref.rdData, seqMux->clk, liveGate, seqMux->rst,
        resetVal, regName);
    mapping.map(loadOp.getResult(),
                comb::MuxOp::create(builder, loc, liveGate, *pref.rdData,
                                    captured));
    return success();
  }
  mapping.map(loadOp.getResult(), *pref.rdData);
  return success();
}

/// Lower any `HWStoreLoweringInterface` op: drive the addresses, write
/// data, and write enable (gated by `wrEnGate`) on the bound port.
static LogicalResult
handleHWStore(loopschedule::HWStoreLoweringInterface storeOp,
               OpBuilder &builder, IRMapping &mapping, Value wrEnGate,
               DenseMap<Value, MemPortMapping> &memPorts,
               SeqPortMuxCtx *seqMux = nullptr) {
  assert(wrEnGate && "handleHWStore requires a wrEnGate");
  Location loc = storeOp->getLoc();
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
  bool contended =
      seqMux && seqMux->multi.contains({storeOp.getMemoryValue(), port});
  pref.addrs->resize(widths.size());
  for (auto [d, idx] : llvm::enumerate(storeOp.getIndices())) {
    Value addr = mapping.lookup(idx);
    addr = resizeIntTo(builder, loc, addr, widths[d]);
    if (contended) {
      Value prev = (*pref.addrs)[d];
      if (!prev)
        prev = hw::ConstantOp::create(builder, loc, addr.getType(), 0);
      addr = comb::MuxOp::create(builder, loc, wrEnGate, addr, prev);
    }
    (*pref.addrs)[d] = addr;
  }
  Value wrData = mapping.lookup(storeOp.getValueToStore());
  if (contended && *pref.wrData)
    wrData = comb::MuxOp::create(builder, loc, wrEnGate, wrData, *pref.wrData);
  *pref.wrData = wrData;
  // NB: no WAR fence here. Unlike the sequential path (lowerSeqDynAccess),
  // gating a store's wr_en on rd_idle in a PIPELINE deadlocks an rw port: the
  // store would wait for all reads to drain while a RAW-fenced load waits for
  // this store to drain — a cross-direction cycle the overlapping schedule
  // cannot break. WAR on a pipelined AXI port instead relies on program order
  // (the earlier load's AR issues before this store's AW). The RAW fence in
  // handleHWLoad is deadlock-free because writes drain autonomously.
  *pref.wrEn = (contended && *pref.wrEn)
                   ? (Value)comb::OrOp::create(builder, loc, *pref.wrEn,
                                               wrEnGate)
                   : wrEnGate;
  return success();
}

//===----------------------------------------------------------------------===//
// Sequential-frame dynamic-access handshake
//===----------------------------------------------------------------------===//

/// Context for lowering BARE dynamic accesses (variable-latency AMC dyn
/// ports) inside sequential frames. The sequential FSM historically treated
/// them as fixed-latency ops — enables gated only on frame-cycle pulses and
/// unconditional state advance — which silently drops requests when the
/// port's `ready` is low and samples data before `done`. With this context,
/// each dynamic access gets a one-shot ready-gated issue and the FSM stalls
/// (via the collected `stallTerms`, OR-reduced into the sequential FSM's
/// `stall` input) until the access's completion pulse arrives. Null where
/// bare dynamic accesses are unsupported (function-level frames), in which
/// case they are a pass error.
struct SeqDynCtx {
  Value clk;
  Value rst;
  Block *hwBody;               // insertion anchor for latches/registers
  BackedgeBuilder *bb;
  ArrayRef<Value> cycleGates;  // enclosing frame's per-cycle gates
  Value frameActive;           // enclosing frame's active level (latch reset)
  SmallVectorImpl<Value> *stallTerms;
  std::string namePrefix;
  unsigned *counter;           // unique naming across accesses
  // Iteration-advance edge (iter_advance & !stall, may be null). With the
  // COND bypass the FSM loops from the last frame state straight back into
  // FRAME_0, so frameActive never drops between iterations — the one-shot
  // accepted/seen latches must ALSO clear on this edge or iteration N+1's
  // accesses would see themselves as already issued/completed.
  Value advanceEdge;
  // Per-port "previous read's registered `seen`" for THIS frame (null ⇒ no
  // chaining). Multi-outstanding reads on one dyn port share the port's
  // `done` pulse; to attribute each pulse to the right read (responses are
  // in issue order on a single-ID master) a read may only capture once the
  // previous read on its port has been seen. Keyed by the port value; reset
  // per frame because the `seen` latches reset when the frame deactivates.
  llvm::DenseMap<mlir::Value, mlir::Value> *prevReadSeen = nullptr;
};

/// Lower one bare dynamic load/store in a sequential frame with a full
/// ready/done handshake. Sequential steps serialize, so at most one access
/// per port is in flight: the pipeline's per-expect counters/FIFOs collapse
/// to one accepted-latch, one done-latch, and (for loads) one data-capture
/// register per access.
///
/// Semantics (budget = the op's scheduled cycle_latency, a MINIMUM):
///   - issue: the enable pulses only on `gate & ready & !accepted`; the FSM
///     stalls at the issue state while `!accepted & !ready` (no drops).
///     The accepted-latch makes the issue one-shot even while the state is
///     held by an unrelated stall.
///   - completion: the FSM stalls at the access's LAST budget state until
///     the port's completion pulse has been seen (done for loads; wr_done
///     when the rw face splits it, else done for stores). An on-time pulse
///     bypasses the stall (zero overhead for fixed-latency-like behavior).
///   - data: rd_data is captured on the completion pulse; consumers (which
///     execute at states after the last budget state) read the register.
static LogicalResult
lowerSeqDynAccess(Operation *op, OpBuilder &builder, IRMapping &mapping,
                  DenseMap<Value, MemPortMapping> &memPorts, Value gate,
                  unsigned atOffset, SeqDynCtx *seqDyn) {
  if (!seqDyn)
    return op->emitError(
        "dynamic memory accesses outside loops are not yet supported by the "
        "sequential FSM lowering; place the access in a loop (or pipeline "
        "the enclosing loop)");

  auto loadOp = dyn_cast<loopschedule::HWLoadLoweringInterface>(op);
  auto storeOp = dyn_cast<loopschedule::HWStoreLoweringInterface>(op);
  Value memVal = loadOp ? loadOp.getMemoryValue() : storeOp.getMemoryValue();
  auto it = memPorts.find(memVal);
  if (it == memPorts.end())
    return op->emitError("unmapped memory");
  MemPortMapping &mp = it->second;

  Location loc = op->getLoc();
  auto i1 = builder.getI1Type();
  OpBuilder hb(builder.getContext());
  hb.setInsertionPointToEnd(seqDyn->hwBody);
  Value zero = hw::ConstantOp::create(hb, loc, i1, 0);

  unsigned lat = getOpCycleLatency(op);
  unsigned lastC = atOffset + (lat > 0 ? lat - 1 : 0);
  if (lastC >= seqDyn->cycleGates.size())
    lastC = seqDyn->cycleGates.size() - 1;
  Value lastGate = seqDyn->cycleGates[lastC];

  Value ready = mp.ready; // null => tied high (no backpressure)
  // RAW/WAR fence: a fenced access may not issue until the opposite
  // direction's drain is idle (a load waits for wr_idle, a store for
  // rd_idle). Folding the drain into `ready` reuses the one-shot-issue /
  // ready-stall machinery below, so completion accounting stays exact. Null
  // drain (non-posted port) leaves it a no-op.
  if (op->hasAttr("fence")) {
    Value drain = loadOp ? mp.wrIdle : mp.rdIdle;
    if (drain)
      ready = ready ? (Value)comb::AndOp::create(hb, loc, ready, drain, false)
                    : drain;
  }
  Value done = mp.done;
  if (storeOp) {
    if (mp.wrDone) {
      // rw faces split write completion out (AXI dyn rw ports).
      done = mp.wrDone;
    } else if (mp.rdData) {
      // Readable port without a split write-done (on-chip rw arbiter
      // faces): `done` covers READS only — a store completion-stall would
      // deadlock. Keep the fixed-latency budget for stores (bounded
      // on-chip latency), with only the ready/one-shot issue machinery.
      done = Value();
    }
    // Write-only ports: `done` IS the store completion — stall on it.
  }
  std::string base =
      seqDyn->namePrefix + "_seqdyn" + std::to_string((*seqDyn->counter)++);

  // Latch hold gate: latches persist while the frame stays active and the
  // loop is not advancing to its next iteration (the COND-bypass FSM keeps
  // frameActive high across back-to-back iterations, so the advance edge
  // is the only reset between them).
  Value latchHold = seqDyn->frameActive;
  if (seqDyn->advanceEdge) {
    Value notAdvance = comb::createOrFoldNot(hb, loc, seqDyn->advanceEdge);
    latchHold = comb::AndOp::create(hb, loc, latchHold, notAdvance, false);
  }

  // Accepted-latch: one-shot issue. Set on the issue handshake, cleared
  // when the frame deactivates (between frames) or the iteration advances.
  // Registered, so an enable pulse under a same-cycle stall from another
  // access still marks this access issued.
  Backedge accNextBE = seqDyn->bb->get(i1);
  auto accReg =
      seq::CompRegOp::create(hb, loc, Value(accNextBE), seqDyn->clk,
                             seqDyn->rst, zero, hb.getStringAttr(base + "_acc"));
  Value notAcc = comb::createOrFoldNot(hb, loc, accReg);
  Value issueHandshake = comb::AndOp::create(
      hb, loc, gate,
      ready ? (Value)comb::AndOp::create(hb, loc, ready, notAcc, false)
            : notAcc,
      false);
  Value accNext = comb::AndOp::create(
      hb, loc, comb::OrOp::create(hb, loc, accReg, issueHandshake, false),
      latchHold, false);
  accNextBE.setValue(accNext);

  // Ready stall: want to issue but the port can't take it.
  if (ready) {
    Value notReady = comb::createOrFoldNot(hb, loc, ready);
    Value readyStall = comb::AndOp::create(
        hb, loc, comb::AndOp::create(hb, loc, gate, notAcc, false), notReady,
        false);
    seqDyn->stallTerms->push_back(readyStall);
  }

  // Done-latch: sticky completion seen since this access issued. The pulse
  // is attributed with the REGISTERED accepted bit (a same-cycle issue of a
  // later access on the same port cannot steal it) AND the not-yet-seen bit
  // (the FIRST post-issue pulse belongs to this access; later pulses on the
  // shared port belong to later accesses and must not re-trigger the data
  // capture).
  Value doneMine;
  Value seenFed;
  if (done) {
    Backedge seenNextBE = seqDyn->bb->get(i1);
    auto seenReg = seq::CompRegOp::create(hb, loc, Value(seenNextBE),
                                          seqDyn->clk, seqDyn->rst, zero,
                                          hb.getStringAttr(base + "_seen"));
    Value notSeenReg = comb::createOrFoldNot(hb, loc, seenReg);
    doneMine = comb::AndOp::create(
        hb, loc, comb::AndOp::create(hb, loc, done, accReg, false),
        notSeenReg, false);
    // Multi-outstanding read attribution: this port's `done` pulse belongs to
    // the OLDEST un-seen read (single-ID master ⇒ responses in issue order).
    // Gate this load's capture on the previous read's REGISTERED `seen` so it
    // only fires on the NEXT pulse, not the one the previous read already
    // consumed. Then record this read's `seen` as the chain tail. Loads only —
    // stores don't capture data and the tested sequential kernels keep one
    // store per port in flight.
    if (loadOp && seqDyn->prevReadSeen) {
      if (Value prev = seqDyn->prevReadSeen->lookup(memVal))
        doneMine = comb::AndOp::create(hb, loc, doneMine, prev, false);
      (*seqDyn->prevReadSeen)[memVal] = seenReg;
    }
    Value seenNext = comb::AndOp::create(
        hb, loc, comb::OrOp::create(hb, loc, seenReg, doneMine, false),
        latchHold, false);
    seenNextBE.setValue(seenNext);
    // Live bypass: a pulse arriving exactly at the check state stalls
    // nothing.
    seenFed = comb::OrOp::create(hb, loc, seenReg, doneMine, false);

    // Completion stall: hold the last budget state until the pulse landed.
    Value notSeen = comb::createOrFoldNot(hb, loc, seenFed);
    Value doneStall = comb::AndOp::create(hb, loc, lastGate, notSeen, false);
    seqDyn->stallTerms->push_back(doneStall);
  }

  // Drive the port with MERGE semantics: several serialized accesses share
  // one port's drive slot per frame (the static helpers overwrite it — the
  // pre-existing sequential clobbering bug: only the last-lowered access
  // ever reached the port). Enables are disjoint by construction (distinct
  // frame-cycle gates, one-shot accepted latches), so each access muxes its
  // address/data in under its own enable and ORs its enable in.
  unsigned port = getBindingPort(op);
  PortDrivesRef pref = portRef(mp, port);
  SmallVector<unsigned> widths = loadOp ? loadOp.getAddrWidths()
                                        : storeOp.getAddrWidths();
  auto indices = loadOp ? loadOp.getIndices() : storeOp.getIndices();
  if (indices.size() != widths.size())
    return op->emitError("memory access index count (")
           << indices.size() << ") does not match memory rank ("
           << widths.size() << ")";
  pref.addrs->resize(widths.size());
  for (auto [d, idx] : llvm::enumerate(indices)) {
    Value addr = mapping.lookup(idx);
    addr = resizeIntTo(builder, loc, addr, widths[d]);
    Value &slot = (*pref.addrs)[d];
    slot = slot ? (Value)comb::MuxOp::create(builder, loc, issueHandshake,
                                             addr, slot)
                : addr;
  }

  if (loadOp) {
    if (loadOp.requiresReadEnable()) {
      Value &en = *pref.rdEn;
      en = en ? (Value)comb::OrOp::create(builder, loc, en, issueHandshake,
                                          false)
              : issueHandshake;
    }
    // FWFT stream pop (declared-zero read latency + read-enable): the beat
    // is valid ON the pop cycle and the FIFO head advances right after, so
    // the doneMine-timed capture below would latch the NEXT beat (accReg
    // registers one cycle behind the pop). Capture at issue instead —
    // issueHandshake is gated on the port's beat-valid `ready`, so the
    // captured datum is exactly the popped beat. Acceptance is completion:
    // the ready-stall above already holds the state until the beat lands.
    if (loadOp.requiresReadEnable() && loadOp.getReadLatency() == 0 &&
        pref.rdData && *pref.rdData) {
      auto capReg = seq::CompRegClockEnabledOp::create(
          hb, loc, *pref.rdData, seqDyn->clk, issueHandshake, seqDyn->rst,
          hw::ConstantOp::create(hb, loc, (*pref.rdData).getType(), 0),
          hb.getStringAttr(base + "_cap"));
      mapping.map(loadOp.getResult(), capReg);
      return success();
    }
    // Map the result to a capture register: data is valid ON the completion
    // pulse and consumers execute at later (post-stall) states.
    if (done && pref.rdData && *pref.rdData) {
      Value capCE = doneMine;
      auto capReg = seq::CompRegClockEnabledOp::create(
          hb, loc, *pref.rdData, seqDyn->clk, capCE, seqDyn->rst,
          hw::ConstantOp::create(hb, loc, (*pref.rdData).getType(), 0),
          hb.getStringAttr(base + "_cap"));
      mapping.map(loadOp.getResult(), capReg);
    } else {
      mapping.map(loadOp.getResult(), *pref.rdData);
    }
    return success();
  }

  // Data-less dynamic stores (amc.burst_copy: the data plane lives inside
  // the copy engine) drive only the enable.
  if (Value toStore = storeOp.getValueToStore()) {
    Value wrData = mapping.lookup(toStore);
    Value &dataSlot = *pref.wrData;
    dataSlot = dataSlot ? (Value)comb::MuxOp::create(builder, loc,
                                                     issueHandshake, wrData,
                                                     dataSlot)
                        : wrData;
  }
  Value &wrEnSlot = *pref.wrEn;
  wrEnSlot = wrEnSlot ? (Value)comb::OrOp::create(builder, loc, wrEnSlot,
                                                  issueHandshake, false)
                      : issueHandshake;
  return success();
}

/// True if `op` is a variable-latency (dynamic) memory access that needs
/// the sequential handshake path.
static bool isSeqDynAccess(Operation *op) {
  if (auto li = dyn_cast<loopschedule::LoadInterface>(op))
    return li.isDynamic();
  if (auto si = dyn_cast<loopschedule::StoreInterface>(op))
    return si.isDynamic();
  return false;
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
emitHwInstanceFromOperator(
    Operation *origOp, OpBuilder &builder, IRMapping &mapping,
    ModuleOp moduleOp, oplib::HwInstanceOp templateInst, Value clk, Value rst,
    llvm::StringMap<unsigned> &uniquer, StringRef opName, Value opCE = {},
    Value shareGate = {},
    llvm::DenseMap<std::pair<mlir::StringAttr, int64_t>,
                   SharedOperatorInstance> *sharedOps = nullptr) {
  auto externOp = moduleOp.lookupSymbol<hw::HWModuleExternOp>(
      templateInst.getModuleNameAttr().getValue());
  if (!externOp)
    return origOp->emitOpError("operator '")
           << opName << "' references unknown extern module @"
           << templateInst.getModuleName();

  auto inputAttrs = externOp.getAllInputAttrs();
  auto outputAttrs = externOp.getAllOutputAttrs();

  // Map each op-result to the instance output carrying `oplib.result = N`.
  auto mapResults = [&](hw::InstanceOp instOp) {
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
  };

  // Shared-instance path: the binding pass proved same-(operator, binding)
  // ops never issue in the same cycle, so a later user joins the first
  // user's instance — operands muxed in under this user's activation gate,
  // CE narrowed to the AND of the users' CEs (see SharedOperatorInstance).
  auto bindingAttr =
      origOp->getAttrOfType<IntegerAttr>("loopschedule.binding");
  // Debug escape hatch: LOOPSCHEDULE_NO_SHARED_OPERATORS=1 gives every
  // user its own instance (isolates shared-join bugs from schedule bugs).
  bool shareable = bindingAttr && shareGate && sharedOps &&
                   !::getenv("LOOPSCHEDULE_NO_SHARED_OPERATORS");
  std::pair<mlir::StringAttr, int64_t> shareKey;
  if (shareable) {
    shareKey = {builder.getStringAttr(opName), bindingAttr.getInt()};
    auto it = sharedOps->find(shareKey);
    if (it != sharedOps->end()) {
      SharedOperatorInstance &shared = it->second;
      for (auto [j, idx] : llvm::enumerate(shared.operandPortForIdx)) {
        if (idx < 0)
          continue;
        if (j >= origOp->getNumOperands())
          return origOp->emitOpError("operator '")
                 << opName << "' shared instance expects operand " << j;
        Value mine = mapping.lookup(origOp->getOperand(j));
        Value cur = shared.instOp->getOperand(idx);
        Value muxed = comb::MuxOp::create(builder, origOp->getLoc(),
                                          shareGate, mine, cur);
        shared.instOp->setOperand(idx, muxed);
      }
      if (shared.ceInputIdx >= 0) {
        Value oneI1Local = hw::ConstantOp::create(
            builder, origOp->getLoc(), builder.getI1Type(), (int64_t)1);
        Value myCE = opCE ? opCE : oneI1Local;
        if (myCE != shared.ce) {
          Value newCE = comb::AndOp::create(builder, origOp->getLoc(),
                                            shared.ce, myCE);
          shared.instOp->setOperand(shared.ceInputIdx, newCE);
          shared.ce = newCE;
        }
      }
      mapResults(shared.instOp);
      return success();
    }
  }

  Value oneI1 = hw::ConstantOp::create(builder, origOp->getLoc(),
                                       builder.getI1Type(), (int64_t)1);
  Value clkValue = clk;

  SmallVector<Value> instanceOperands;
  SharedOperatorInstance sharedEntry;
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
      // Pipelined operators (e.g. int_mul_pipe_*) shift internally every
      // enabled cycle. In a context that can stall (a pipeline waiting on
      // dyn-latency expects, a sequential frame stalled on a handshake),
      // the operator must freeze with the stage/frame registers — a free-
      // running CE lets in-flight values march out of the operator pipe
      // during the stall and be lost or double-consumed.
      driver = opCE ? opCE : oneI1;
      sharedEntry.ceInputIdx = (int)instanceOperands.size();
      sharedEntry.ce = driver;
    } else if (portAttrs) {
      if (auto opIdxAttr =
              dyn_cast_or_null<IntegerAttr>(portAttrs.get("oplib.operand"))) {
        unsigned j = opIdxAttr.getInt();
        if (j >= origOp->getNumOperands())
          return origOp->emitOpError("operator '")
                 << opName << "' extern port operand index " << j
                 << " exceeds op operand count";
        driver = mapping.lookup(origOp->getOperand(j));
        if (sharedEntry.operandPortForIdx.size() <= j)
          sharedEntry.operandPortForIdx.resize(j + 1, -1);
        sharedEntry.operandPortForIdx[j] = (int)instanceOperands.size();
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

  if (shareable) {
    sharedEntry.instOp = instOp;
    (*sharedOps)[shareKey] = sharedEntry;
  }

  mapResults(instOp);
  return success();
}

/// Dispatch a single compute op to either a comb-op materialization or an
/// hw.instance materialization based on the operator's hw_match.
static LogicalResult
emitOpFromOperatorLibrary(Operation *origOp, OpBuilder &builder,
                          IRMapping &mapping, ModuleOp moduleOp, Value clk,
                          Value rst, llvm::StringMap<unsigned> &uniquer,
                          analysis::OperatorLibraryAnalysis &ola,
                          Value opCE = {}, Value shareGate = {},
                          llvm::DenseMap<std::pair<mlir::StringAttr, int64_t>,
                                         SharedOperatorInstance> *sharedOps =
                              nullptr) {
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
                                        inst, clk, rst, uniquer, opName,
                                        opCE, shareGate, sharedOps);
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
    Value rst, SeqDynCtx *seqDyn, SeqPortMuxCtx *seqMux, Value opCE) {
  if (seqMux)
    seqMux->cycle = baseCycle;
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
    if (isSeqDynAccess(inner))
      return lowerSeqDynAccess(inner, builder, mapping, memPorts, gate,
                               baseCycle, seqDyn);
    if (auto storeOp = dyn_cast<HWStoreLoweringInterface>(inner))
      return handleHWStore(storeOp, builder, mapping, gate, memPorts, seqMux);
    if (auto loadOp = dyn_cast<HWLoadLoweringInterface>(inner))
      return handleHWLoad(loadOp, builder, mapping, memPorts, gate, seqMux);
    return emitComputeOp(inner, builder, mapping, moduleOp, clk, rst, opCE,
                         /*shareGate=*/gate);
  };
  for (auto &op : *body) {
    if (failed(processOp(&op, pickGate(baseCycle))))
      return failure();
  }
  return success();
}

LogicalResult LoopScheduleToFSMPass::emitComputeOp(
    Operation *op, OpBuilder &builder, IRMapping &mapping, ModuleOp moduleOp,
    Value clk, Value rst, Value opCE, Value shareGate) {
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
                                    instanceUniquer, *operatorLibrary, opCE,
                                    shareGate, &sharedOperators);
}

LogicalResult LoopScheduleToFSMPass::lowerFrameBody(
    Block *frameBody, OpBuilder &builder, IRMapping &mapping,
    ArrayRef<Value> cycleGates,
    DenseMap<Value, MemPortMapping> &memPorts, ModuleOp moduleOp,
    Value clk, Value rst, SeqDynCtx *seqDyn, SeqPortMuxCtx *seqMux,
    Value opCE) {
  for (auto &op : *frameBody) {
    if (isa<LoopScheduleYieldOp>(&op))
      continue;
    if (auto atOp = dyn_cast<LoopScheduleAtOp>(&op)) {
      unsigned offset = (unsigned)atOp.getOffset();
      if (failed(lowerAtBody(&atOp.getBodyBlock(), builder, mapping, cycleGates,
                             offset, memPorts, moduleOp, clk, rst, seqDyn,
                             seqMux, opCE)))
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
    ArrayRef<unsigned> launchStartOffsets,
    ArrayRef<unsigned> frameLatencies, bool foldLastFrame, bool condBypass,
    bool entryBypass) {
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
  assert(launchStartOffsets.size() == launchAtOffsets.size() &&
         "launchStartOffsets must have one entry per launch");
  for (unsigned j = 0; j < launchAtOffsets.size(); ++j)
    assert(launchStartOffsets[j] <= launchAtOffsets[j] &&
           "a launch may only start EARLIER than its scheduled at-offset");

  unsigned numWaits = waitFrameIndices.size();

  // A folded last frame (pure-comb iter_arg advance — see the fold
  // predicate in lowerLoopNodeAsModule) gets no FSM states: its ops are
  // combinational into the iter_arg registers' D inputs, so iter_advance
  // simply fires one frame earlier. Its frame_active_<i> output is still
  // declared (keeping the wrapper's result indexing intact) but is never
  // driven high.
  unsigned numStateFrames = foldLastFrame ? numFrames - 1 : numFrames;
  assert(numStateFrames >= 1 && "folded loop must keep at least one frame");

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
  assert((!foldLastFrame || (frameWaitIdx[numFrames - 1].empty() &&
                             frameLats[numFrames - 1] == 1)) &&
         "folded last frame must be launch-free and single-cycle");

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

  // Inputs: start, cond, cond_next, child_done_0..C-1, stall.
  // `cond_next` is the loop condition evaluated on the iter_args' NEXT
  // values (the feedback wires that latch on the iteration-advance edge).
  // With condBypass the last state loops back directly to FRAME_0 /
  // DONE on it, skipping the per-iteration COND state; without the
  // bypass the wrapper drives it with constant 0 and it is unused.
  // `stall` freezes the frame-state machinery: transitions between frame
  // cycle states (and out of WAIT) hold while it is high. It is the
  // OR of the per-dynamic-access ready/done stall terms computed in the
  // module body (low whenever no frame is active, so IDLE/COND/DONE need
  // no guard).
  SmallVector<Type> inputTypes;
  inputTypes.push_back(i1); // start
  inputTypes.push_back(i1); // cond
  inputTypes.push_back(i1); // cond_next
  for (unsigned j = 0; j < numWaits; ++j)
    inputTypes.push_back(i1); // child_done_j
  inputTypes.push_back(i1);   // stall

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
  argNames.push_back(builder.getStringAttr("cond_next"));
  for (unsigned j = 0; j < numWaits; ++j)
    argNames.push_back(
        builder.getStringAttr("child_done_" + std::to_string(j)));
  argNames.push_back(builder.getStringAttr("stall"));
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

  // --- IDLE ---
  // IDLE (and, with the bypass, DONE) outputs first_iter = the START
  // input rather than the first_iter variable: the iter_arg registers'
  // clock-enable includes first_iter, so the inits latch exactly on the
  // start cycle, and the wrapper's cond input — which reads the
  // iter_args through the first_iter mux — is automatically evaluated
  // over the inits while start is high. With condBypass that lets IDLE
  // jump straight into FRAME_0 (or DONE for a zero-trip run), skipping
  // the entry COND state entirely; COND then has no predecessors and is
  // not emitted.
  auto entryOut = [&]() -> SmallVector<Value> {
    SmallVector<Value> v = buildOut(falseVal, false, -1, -1, -1, -1);
    v[1] = machine.getArgument(0); // first_iter = start
    return v;
  };
  assert(!entryBypass || condBypass);
  auto emitEntryTransitions = [&](Block *tb) {
    fb.setInsertionPointToEnd(tb);
    if (!entryBypass) {
      fsm::TransitionOp::create(
          fb, loc, StringRef("COND"),
          [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
          [&]() { fsm::UpdateOp::create(fb, loc, fiVar, trueVal); });
      return;
    }
    fsm::TransitionOp::create(
        fb, loc, StringRef(frameStateName(0, 0)),
        [&]() {
          Value g = comb::AndOp::create(fb, loc, machine.getArgument(0),
                                        machine.getArgument(1));
          fsm::ReturnOp::create(fb, loc, g);
        },
        [&]() { fsm::UpdateOp::create(fb, loc, fiVar, falseVal); });
    // Zero-trip run: report done without entering the body.
    fsm::TransitionOp::create(
        fb, loc, StringRef("DONE"),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
        []() {});
  };
  {
    auto st = fsm::StateOp::create(fb, loc, "IDLE");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    fsm::OutputOp::create(fb, loc, entryOut());
    emitEntryTransitions(&st.getTransitions().front());
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // --- COND ---
  if (!entryBypass) {
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

  Value condNextArg = machine.getArgument(2);
  Value stallArg = machine.getArgument(3 + numWaits);
  // Guard body returning !stall — frame-state transitions hold under stall
  // (fsm.machine stays in the current state when no guard is true).
  auto notStallGuard = [&]() {
    Value ns = comb::createOrFoldNot(fb, loc, stallArg);
    fsm::ReturnOp::create(fb, loc, ns);
  };

  // Helper: emit the last frame's exit transitions. `extraGuard` (may be
  // null) is ANDed into every guard alongside !stall — the WAIT exit
  // passes its all-child-dones term through it.
  //
  // Without condBypass we go back to COND so the condition is evaluated
  // with the UPDATED iter_args (the iter_arg register latches on the
  // clock edge leaving the last frame). Checking `cond` at the last
  // frame instead would race: the register hasn't updated yet, so cond
  // still reflects the OLD iter_args — producing one extra spurious
  // iteration.
  //
  // With condBypass the wrapper evaluates the condition on the
  // registers' D wires (`cond_next`), which is exactly the value the
  // iter_args latch on this edge — so we can skip COND and loop straight
  // back into FRAME_0 (or leave to DONE), saving one cycle per
  // iteration. COND remains for loop entry from IDLE.
  auto emitLastFrameTransition = [&](Block *tb,
                                     llvm::function_ref<Value()> extraGuard) {
    fb.setInsertionPointToEnd(tb);
    auto guardWith = [&](bool negateCondNext, bool useCondNext) {
      return [&, negateCondNext, useCondNext]() {
        Value g = comb::createOrFoldNot(fb, loc, stallArg);
        if (extraGuard)
          g = comb::AndOp::create(fb, loc, g, extraGuard());
        if (useCondNext) {
          Value cn = negateCondNext
                         ? comb::createOrFoldNot(fb, loc, condNextArg)
                         : condNextArg;
          g = comb::AndOp::create(fb, loc, g, cn);
        }
        fsm::ReturnOp::create(fb, loc, g);
      };
    };
    if (!condBypass) {
      fsm::TransitionOp::create(fb, loc, StringRef("COND"),
                                guardWith(false, false), []() {});
      return;
    }
    fsm::TransitionOp::create(fb, loc, StringRef(frameStateName(0, 0)),
                              guardWith(false, true), []() {});
    fsm::TransitionOp::create(fb, loc, StringRef("DONE"),
                              guardWith(true, true), []() {});
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
  // this frame — latching is done in the hardware wrapper), and WAIT_i
  // exits directly into the next frame (or COND for the last frame).
  // Otherwise the last cycle state transitions directly to the next
  // frame (or COND for the last frame).
  //
  // iter_advance fires in the last cycle state of the last frame when
  // it has no launches. When the last frame ends in a WAIT there is no
  // state left to host a Moore iter_advance (held high across a
  // multi-cycle WAIT it would re-latch the iter_args every cycle), so
  // the WRAPPER computes the advance clock-enable from the WAIT exit
  // condition itself (in-WAIT AND all child dones — see
  // lowerLoopNodeAsModule).
  //
  // Multiple launches in the same frame run concurrently: each fires
  // its start pulse at its own atOffset, and the single WAIT waits for
  // all of them to finish. Launches at the same atOffset fire in the
  // same cycle.
  for (unsigned i = 0; i < numStateFrames; ++i) {
    bool isLast = (i + 1 == numStateFrames);
    unsigned L = frameLats[i];
    bool frameHasLaunch = !frameWaitIdx[i].empty();

    // Precompute per-cycle start/live lists for this frame.
    // frameWaitIdx[i] holds global launch indices for launches in frame i;
    // launchStartOffsets[j] is the cycle where launch j's child_start
    // pulses (<= its scheduled at-offset — the early-start peephole).
    SmallVector<SmallVector<unsigned>> startsPerCycle(L);
    SmallVector<SmallVector<unsigned>> livesPerCycle(L);
    for (int j : frameWaitIdx[i]) {
      unsigned o = launchStartOffsets[(unsigned)j];
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
                                  StringRef(frameStateName(i, c + 1)),
                                  notStallGuard, []() {});
      } else if (frameHasLaunch) {
        fsm::TransitionOp::create(fb, loc, StringRef(waitStateName(i)),
                                  notStallGuard, []() {});
      } else if (isLast) {
        emitLastFrameTransition(tb, nullptr);
      } else {
        fsm::TransitionOp::create(fb, loc, StringRef(leaveTargetName),
                                  notStallGuard, []() {});
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
    // is sampling. WAIT exits directly into the next frame (or COND for
    // the last frame — the iter_arg latch enable is computed in the
    // wrapper from this same exit condition, see lowerLoopNodeAsModule);
    // there is no POST settle state.
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
      // AND of child_done_<j> for every launch in this frame (!stall is
      // added by the transition emitters; access stall terms are
      // frame-gated so it rarely binds in WAIT).
      auto allDonesGuard = [&]() -> Value {
        Value guard;
        for (int j : frameWaitIdx[i]) {
          Value done = machine.getArgument(3 + (unsigned)j);
          guard = guard ? comb::AndOp::create(fb, loc, guard, done)
                        : done;
        }
        return guard;
      };
      if (isLast) {
        emitLastFrameTransition(tb, allDonesGuard);
      } else {
        fsm::TransitionOp::create(
            fb, loc, StringRef(leaveTargetName),
            [&]() {
              Value guard = allDonesGuard();
              Value ns = comb::createOrFoldNot(fb, loc, stallArg);
              guard =
                  guard ? comb::AndOp::create(fb, loc, guard, ns) : ns;
              fsm::ReturnOp::create(fb, loc, guard);
            },
            []() {});
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
    SmallVector<Value> v = buildOut(trueVal, false, -1, -1, -1, -1);
    if (condBypass)
      v[1] = machine.getArgument(0); // first_iter = start (re-entry latch)
    fsm::OutputOp::create(fb, loc, v);
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    if (condBypass) {
      // With the early-done cut-through the parent's WAIT exits on our
      // final advance edge — its next start pulse lands while we are
      // still transiting DONE→IDLE, so DONE must accept start exactly
      // like IDLE. A zero-trip re-entry holds DONE (done stays high for
      // the new invocation).
      fsm::TransitionOp::create(
          fb, loc, StringRef(frameStateName(0, 0)),
          [&]() {
            Value g = comb::AndOp::create(fb, loc, machine.getArgument(0),
                                          machine.getArgument(1));
            fsm::ReturnOp::create(fb, loc, g);
          },
          [&]() { fsm::UpdateOp::create(fb, loc, fiVar, falseVal); });
      fsm::TransitionOp::create(
          fb, loc, StringRef("DONE"),
          [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
          []() {});
    }
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
    ArrayRef<int> frameChildKind, ArrayRef<unsigned> entryLatencies,
    ArrayRef<unsigned> entryGroup, SmallVectorImpl<int> &entryCycleOutBase) {
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();
  unsigned numFrames = frameChildKind.size();

  // Count non-leaf frames to determine child_start outputs.
  unsigned numChildren = 0;
  SmallVector<int> childIndexForFrame(numFrames, -1);
  for (unsigned i = 0; i < numFrames; ++i) {
    if (frameChildKind[i] >= 0) {
      childIndexForFrame[i] = numChildren;
      numChildren++;
    }
  }

  // Partition entries into contiguous groups (concurrent siblings of one
  // frame). Each group's members are launched together and awaited
  // together; the FSM takes ONE done input per group that waits.
  struct Group {
    unsigned first;                 // first entry index
    SmallVector<unsigned> members;  // entry indices
    bool hasChild = false;          // any kind >= 0 member
    bool allStateless = true;       // every member kind == -2
    int doneArg = -1;               // FSM input index (set below)
  };
  SmallVector<Group> groupList;
  for (unsigned i = 0; i < numFrames; ++i) {
    if (groupList.empty() || entryGroup[i] != entryGroup[groupList.back().first])
      groupList.push_back({i, {}, false, true, -1});
    Group &g = groupList.back();
    g.members.push_back(i);
    if (frameChildKind[i] >= 0)
      g.hasChild = true;
    if (frameChildKind[i] != -2)
      g.allStateless = false;
  }
  unsigned numWaitGroups = 0;
  for (Group &g : groupList)
    if (g.hasChild)
      g.doneArg = 1 + (int)numWaitGroups++;

  // Multi-cycle LEAF entries (latency > 1, e.g. two same-port stores the
  // scheduler serialized into `at 0` / `at 1`) hold their state for L
  // cycles and expose one dedicated cycle output per cycle, mirroring the
  // sequential FSM's frame-cycle outputs. `entryCycleOutBase[i]` is the
  // entry's base index into that appended output region (-1 = none).
  auto latencyOf = [&](unsigned i) -> unsigned {
    unsigned l = i < entryLatencies.size() ? entryLatencies[i] : 1;
    return frameChildKind[i] < 0 ? std::max(l, 1u) : 1u;
  };
  entryCycleOutBase.assign(numFrames, -1);
  unsigned totalCycleOuts = 0;
  for (unsigned i = 0; i < numFrames; ++i) {
    if (latencyOf(i) > 1) {
      entryCycleOutBase[i] = (int)totalCycleOuts;
      totalCycleOuts += latencyOf(i);
    }
  }

  // Inputs: start, group_done_0, ..., group_done_{numWaitGroups-1} (one
  // per group that launches children; the caller supplies the group's
  // all-members-done conjunction).
  SmallVector<Type> inputTypes(1 + numWaitGroups, i1);
  // Outputs: done, child_start_0..N, frame_running_0..M, then the appended
  // per-cycle outputs of multi-cycle leaf entries.
  SmallVector<Type> outputTypes(1 + numChildren + numFrames + totalCycleOuts,
                                i1);

  auto funcType = FunctionType::get(ctx, inputTypes, outputTypes);
  auto machine =
      fsm::MachineOp::create(builder, loc, fsmName, "IDLE", funcType);

  // Arg names.
  SmallVector<Attribute> argNameAttrs;
  argNameAttrs.push_back(builder.getStringAttr("start"));
  for (unsigned i = 0; i < numWaitGroups; ++i)
    argNameAttrs.push_back(
        builder.getStringAttr("group_done_" + std::to_string(i)));
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
  for (unsigned i = 0; i < numFrames; ++i)
    for (unsigned c = 0; entryCycleOutBase[i] >= 0 && c < latencyOf(i); ++c)
      resNameAttrs.push_back(builder.getStringAttr(
          "frame_cycle_" + std::to_string(i) + "_" + std::to_string(c)));
  machine.setResNamesAttr(builder.getArrayAttr(resNameAttrs));

  OpBuilder fb(ctx);
  fb.setInsertionPointToEnd(&machine.getBody().front());

  Value trueVal = hw::ConstantOp::create(fb, loc, i1, 1);
  Value falseVal = hw::ConstantOp::create(fb, loc, i1, 0);

  // Helper to build output vector.
  // out[0] = done, out[1..numChildren] = child_start,
  // out[1+numChildren..] = frame_running, then the per-cycle outputs
  // (activeCycleOut is a GLOBAL index into that appended region, -1 none).
  // The child-start and frame-running actives are SETS: a group state
  // raises them for every member at once.
  auto makeOutput = [&](bool done, ArrayRef<int> activeChildStarts,
                        ArrayRef<int> activeFrameRunnings,
                        int activeCycleOut = -1) -> SmallVector<Value> {
    SmallVector<Value> vals;
    vals.push_back(done ? trueVal : falseVal);
    for (unsigned i = 0; i < numChildren; ++i)
      vals.push_back(llvm::is_contained(activeChildStarts, (int)i) ? trueVal
                                                                   : falseVal);
    for (unsigned i = 0; i < numFrames; ++i)
      vals.push_back(llvm::is_contained(activeFrameRunnings, (int)i)
                         ? trueVal
                         : falseVal);
    for (unsigned c = 0; c < totalCycleOuts; ++c)
      vals.push_back((int)c == activeCycleOut ? trueVal : falseVal);
    return vals;
  };

  // States are named after the group's FIRST entry index, so singleton
  // groups keep the historical FRAME_<i>/WAIT_<i> names. All-stateless
  // groups get no states: successor chains skip them.
  auto nextEmittedState = [&](unsigned g) -> std::string {
    for (unsigned j = g + 1; j < groupList.size(); ++j)
      if (!groupList[j].allStateless)
        return "FRAME_" + std::to_string(groupList[j].first);
    return "DONE";
  };
  int firstEmitted = -1;
  for (unsigned g = 0; g < groupList.size(); ++g)
    if (!groupList[g].allStateless) {
      firstEmitted = (int)groupList[g].first;
      break;
    }

  // --- IDLE ---
  {
    auto st = fsm::StateOp::create(fb, loc, "IDLE");
    Block *ob = st.ensureOutput(fb);
    ob->getTerminator()->erase();
    fb.setInsertionPointToEnd(ob);
    // NOTE: an IDLE->WAIT entry-launch bypass (starting the first child
    // straight from IDLE) was tried and reverted: both the Mealy- and the
    // wrapper-pulse variant reshaped the one-hot FSM enough to flip a
    // marginal state-decode -> LUTRAM -> DSP-operand path in atax from
    // +0.17ns to about -0.3ns at the 2ns target — one cycle per kernel
    // invocation is not worth an Fmax cliff.
    fsm::OutputOp::create(fb, loc, makeOutput(false, {}, {}));
    Block *tb = &st.getTransitions().front();
    fb.setInsertionPointToEnd(tb);
    std::string entryTarget =
        firstEmitted < 0 ? std::string("DONE")
                         : "FRAME_" + std::to_string(firstEmitted);
    fsm::TransitionOp::create(
        fb, loc, StringRef(entryTarget),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
        []() {});
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // --- FRAME_g and WAIT_g states (one pair per group) ---
  for (unsigned g = 0; g < groupList.size(); ++g) {
    Group &grp = groupList[g];
    if (grp.allStateless)
      continue; // stateless await-only group
    std::string frameName = "FRAME_" + std::to_string(grp.first);
    std::string nextState = nextEmittedState(g);
    bool isLeaf = !grp.hasChild;

    // The group's active sets: every member's child_start (non-leaf
    // members) pulses in FRAME_g, and every member's frame_running holds
    // through FRAME_g (+ WAIT_g).
    SmallVector<int> groupChildStarts, groupRunnings;
    for (unsigned i : grp.members) {
      if (frameChildKind[i] >= 0)
        groupChildStarts.push_back(childIndexForFrame[i]);
      if (frameChildKind[i] != -2)
        groupRunnings.push_back((int)i);
    }

    // FRAME_g, plus FRAME_g_C1..C{L-1} chained cycle states for multi-cycle
    // leaf entries: frame_running stays high across the whole chain and
    // each cycle state additionally raises its frame_cycle_<i>_<c> output,
    // so `at K` bodies get a genuine per-cycle issue gate (two same-port
    // stores serialized by the scheduler need K to really be cycle K).
    // Leaf groups are always singletons (leaf entries only arise from
    // launch-less frames), so the per-entry cycle-out model is unchanged.
    {
      unsigned lat = latencyOf(grp.first);
      for (unsigned c = 0; c < lat; ++c) {
        std::string stateName =
            c == 0 ? frameName : frameName + "_C" + std::to_string(c);
        std::string succ =
            (c + 1 < lat)
                ? frameName + "_C" + std::to_string(c + 1)
                : (isLeaf ? nextState : "WAIT_" + std::to_string(grp.first));
        auto st = fsm::StateOp::create(fb, loc, stateName);
        Block *ob = st.ensureOutput(fb);
        ob->getTerminator()->erase();
        fb.setInsertionPointToEnd(ob);
        int cycleSlot = entryCycleOutBase[grp.first] >= 0
                            ? entryCycleOutBase[grp.first] + (int)c
                            : -1;
        fsm::OutputOp::create(
            fb, loc,
            makeOutput(false, isLeaf ? ArrayRef<int>{} : groupChildStarts,
                       groupRunnings, cycleSlot));
        Block *tb = &st.getTransitions().front();
        fb.setInsertionPointToEnd(tb);
        fsm::TransitionOp::create(fb, loc, StringRef(succ));
        fb.setInsertionPointToEnd(&machine.getBody().front());
      }
    }

    // WAIT_g (child-launching groups only): hold until the group's
    // combined done input (all members done) rises.
    if (!isLeaf) {
      std::string waitName = "WAIT_" + std::to_string(grp.first);
      auto st = fsm::StateOp::create(fb, loc, waitName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      // frame_running stays high during WAIT
      fsm::OutputOp::create(fb, loc, makeOutput(false, {}, groupRunnings));
      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      unsigned doneArgIdx = (unsigned)grp.doneArg;
      fsm::TransitionOp::create(
          fb, loc, StringRef(nextState),
          [&]() {
            fsm::ReturnOp::create(fb, loc, machine.getArgument(doneArgIdx));
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
    fsm::OutputOp::create(fb, loc, makeOutput(true, {}, {}));
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
  // For amc-port memrefs, also plumb the per-port `done` signal in.
  // The memory module emits an honest 1-cycle pulse on the cycle the
  // arbiter actually services this consumer; the loop FSM uses it to
  // freeze the pipeline (and capture the live rdData) until every
  // consumer of a shared multi-output arbiter has been served. Memref-
  // backed entries (function-arg memrefs / local hlmems) don't expose
  // a done signal — those reads complete in fixed latency and don't
  // need a handshake.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasDone)
      continue;
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_done"),
                       builder.getI1Type(),
                       hw::ModulePort::Direction::Input}});
    inputIdx++;
  }
  // Split write-completion for rw dyn faces; plumbed like `done`.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasWrDone)
      continue;
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_wr_done"),
                       builder.getI1Type(),
                       hw::ModulePort::Direction::Input}});
    inputIdx++;
  }
  // Same-cycle acceptance level for dynamic ports (issue backpressure for
  // posted accesses); plumbed like `done` above.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasReady)
      continue;
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_ready"),
                       builder.getI1Type(),
                       hw::ModulePort::Direction::Input}});
    inputIdx++;
  }
  // Write-/read-drain levels for posted AXI faces (RAW/WAR fence); plumbed
  // like `ready` above. rd_idle first, then wr_idle.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasRdIdle)
      continue;
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_rd_idle"),
                       builder.getI1Type(),
                       hw::ModulePort::Direction::Input}});
    inputIdx++;
  }
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasWrIdle)
      continue;
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_wr_idle"),
                       builder.getI1Type(),
                       hw::ModulePort::Direction::Input}});
    inputIdx++;
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
  // Mirror the order of the `mem*_done` input ports added above.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasDone)
      continue;
    localMemPortMap[memInfo.originalArg].done = hwBody->getArgument(argIdx++);
  }
  // Mirror the order of the `mem*_wr_done` input ports added above.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasWrDone)
      continue;
    localMemPortMap[memInfo.originalArg].wrDone =
        hwBody->getArgument(argIdx++);
  }
  // Mirror the order of the `mem*_ready` input ports added above.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasReady)
      continue;
    localMemPortMap[memInfo.originalArg].ready = hwBody->getArgument(argIdx++);
  }
  // Mirror the order of the `mem*_rd_idle` / `mem*_wr_idle` input ports.
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasRdIdle)
      continue;
    localMemPortMap[memInfo.originalArg].rdIdle = hwBody->getArgument(argIdx++);
  }
  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    if (!memInfo.isAmcPort || !memInfo.hasWrIdle)
      continue;
    localMemPortMap[memInfo.originalArg].wrIdle = hwBody->getArgument(argIdx++);
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
/// step_running signals. Loops over each declared port of the memref
/// independently — accesses bound to port K only contend with other
/// port-K accesses.
///
/// `stepGroups` (optional) declares which steps can run CONCURRENTLY:
/// sibling launches of one function frame share a group id and their
/// gates are high simultaneously. For such steps a pure gate-keyed
/// priority mux would let the first active step's idle (zero) drives
/// shadow another concurrent step's real access to a port it doesn't
/// even use, so their addr/wrData select on the step's ACTUAL enables
/// instead (amc ports announce every access through wr_en/rd_en; a step
/// with no access contributes 0 and never wins). Steps in distinct
/// groups — or all steps, when `stepGroups` is empty (the sequential
/// loop path) — are mutually exclusive and keep the exact historical
/// raw-gate selection. Enables always merge as OR of gate-qualified
/// terms, which is equivalent to the historical mux under exclusive
/// gates and correct under concurrency.
static void mergeStepMemPorts(
    OpBuilder &builder, Location loc,
    ArrayRef<DenseMap<Value, MemPortMapping>> perStepPorts,
    ArrayRef<Value> stepRunningSignals,
    ArrayRef<PortArgInfo> memrefArgs,
    DenseMap<Value, MemPortMapping> &mergedPorts,
    ArrayRef<unsigned> stepGroups = {}) {

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
        Value gate = stepRunningSignals[i];
        // Gate-qualified enables for this step.
        Value stepWrEn = pv.wrEn ? pv.wrEn
            : hw::ConstantOp::create(builder, loc, i1, 0);
        Value gatedWrEn = comb::AndOp::create(builder, loc, gate, stepWrEn);
        Value gatedRdEn;
        if (memInfo.requiresRdEn) {
          Value stepRdEn = pv.rdEn ? pv.rdEn
              : hw::ConstantOp::create(builder, loc, i1, 0);
          gatedRdEn = comb::AndOp::create(builder, loc, gate, stepRdEn);
        }
        // Address/data selector: the step's live access. Amc ports always
        // announce an access through an enable (wr_en for stores, rd_en
        // for loads — a write-only port's wr_en IS its only enable), so a
        // concurrent step that never touches the port contributes 0 and
        // never wins. Plain memref ports read passively (no rd_en), so
        // they keep the historical raw-gate selection.
        // Address/data selector. Steps whose gates are mutually
        // exclusive (different concurrency groups — the historical
        // sequential chain) keep the raw gate: it is exact and imposes
        // no requirements on how the step models its enables. Steps that
        // can run CONCURRENTLY (same group: sibling launches of one
        // frame) qualify with their actual enables, so a concurrent step
        // that never touches this port (rd_en = wr_en = 0) cannot shadow
        // the owning step's address. Amc ports always announce accesses
        // through an enable (wr_en for stores, rd_en for loads); plain
        // memref ports read passively, but local memrefs are never
        // shared across concurrent siblings (the binder gives concurrent
        // accesses distinct ports), so the enable term only ever
        // tightens amc-port selection.
        bool concurrentStep =
            !stepGroups.empty() &&
            llvm::count(stepGroups, stepGroups[i]) > 1;
        Value sel = gate;
        if (concurrentStep && memInfo.isAmcPort) {
          sel = gatedWrEn;
          if (gatedRdEn)
            sel = comb::OrOp::create(builder, loc, sel, gatedRdEn);
        }
        for (auto [d, w] : llvm::enumerate(memInfo.addrWidths)) {
          Type addrType = IntegerType::get(ctx, w);
          Value stepAddr = (pv.addrs && d < pv.addrs->size() && (*pv.addrs)[d])
              ? (*pv.addrs)[d]
              : hw::ConstantOp::create(builder, loc, addrType, 0);
          addrs[d] = comb::MuxOp::create(builder, loc, sel, stepAddr,
                                          addrs[d]);
        }
        Value stepWrData = pv.wrData ? pv.wrData
            : hw::ConstantOp::create(builder, loc, dataType, 0);
        wrData = comb::MuxOp::create(builder, loc, sel, stepWrData, wrData);
        wrEn = comb::OrOp::create(builder, loc, gatedWrEn, wrEn);
        if (memInfo.requiresRdEn)
          rdEn = comb::OrOp::create(builder, loc, gatedRdEn, rdEn);
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
    // Zero-address ports are real: stream beat faces (burst_pop/burst_push)
    // drive rdEn/wrData/wrEn with no address slots. Only skip a memref with
    // no consumer-side drives at all.
    bool anyDrive = needsRdEn;
    for (auto &stagePorts : perStagePorts) {
      auto it = stagePorts.find(memref);
      if (it == stagePorts.end())
        continue;
      const MemPortMapping &ports = it->second;
      if (ports.wrEn || ports.wrData ||
          llvm::any_of(ports.addrs, [](Value v) { return (bool)v; }))
        anyDrive = true;
    }
    if (!anyDrive)
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
    // A read-only zero-address port (a pop-only stream face) has no data
    // type to mux — leave wrData untouched.
    Value wrData = existing.wrData;
    if (!wrData && dataType)
      wrData = hw::ConstantOp::create(builder, loc, dataType, 0);
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

/// Minimum at-offset at which pipeline iter_arg `argIdx` is read inside
/// `pipOp` (users outside any `loopschedule.at` count as offset 0), or
/// UINT_MAX when the argument is never read. Shared between the early
/// child_start peephole's init-timing guard and the iter-arg preload
/// depth in `lowerPipelineChild` — the two MUST agree: the peephole only
/// guarantees a same-frame-produced init is readable by cycle
/// `start + firstUse`, and the preload samples the init at exactly that
/// cycle. (This lowering assumes stage at-offsets are contiguous from 0,
/// i.e. offset == position in the stage list — the same invariant the
/// stageCE indexing relies on elsewhere.)
static unsigned firstIterArgUseOffset(loopschedule::LoopSchedulePipelineOp pipOp,
                                      unsigned argIdx) {
  BlockArgument arg = pipOp.getStagesBlock().getArgument(argIdx);
  unsigned firstUse = UINT_MAX;
  for (auto *user : arg.getUsers()) {
    LoopScheduleAtOp userAt;
    Operation *anc = user;
    while (anc && anc != pipOp.getOperation()) {
      if ((userAt = dyn_cast<LoopScheduleAtOp>(anc)))
        break;
      anc = anc->getParentOp();
    }
    firstUse =
        std::min(firstUse, userAt ? (unsigned)userAt.getOffset() : 0u);
  }
  return firstUse;
}

/// Earliest cycle within `frameOp` at which `pipOp`'s child_start may pulse
/// without an iter_arg INIT latching a stale value.
///
/// A same-frame value reaches a launched child through the capture chain
/// `resultNeedsCapture` builds: a producer in `at O` (or anything it feeds
/// combinationally) is register-readable no earlier than cycle O+2, because
/// the deepest capture in its cone latches during O+1. The feedback-register
/// preload in `lowerPipelineChild` samples an init at cycle
/// `start + firstIterArgUseOffset(arg)` and latches it at the END of that
/// cycle, so the value must be readable DURING it:
///
///     start + firstUse >= O + 2
///
/// The SCHEDULE only guarantees the init is COMBINATIONALLY available at
/// `O + operator latency`, which is why this floor is needed. It bites
/// exactly when firstUse is small, i.e. for the induction variable (whose
/// preload depth is forced to 0 because the loop condition reads it
/// unregistered) of a loop with a runtime LOWER bound. Inits that are
/// constants, block args, or frame-external values -- every loop with a
/// static lower bound -- give no constraint at all.
///
/// Deliberately does NOT constrain what the pipeline BODY reads: those
/// crossings are modelled by the scheduler and re-checked by the early
/// child_start peephole's guard (b2).
static unsigned
minLegalLaunchStart(loopschedule::LoopSchedulePipelineOp pipOp,
                    loopschedule::LoopScheduleFrameOp frameOp) {
  Operation *launchHolder =
      frameOp.getBodyBlock().findAncestorOpInBlock(*pipOp.getOperation());
  unsigned minStart = 0;
  for (auto [argIdx, init] : llvm::enumerate(pipOp.getInits())) {
    Operation *def = init.getDefiningOp();
    if (!def || !frameOp->isAncestor(def))
      continue; // stable before the frame started
    Operation *top = frameOp.getBodyBlock().findAncestorOpInBlock(*def);
    if (!top || top == launchHolder)
      continue; // inside our own launch: moves with the child
    auto defAt = dyn_cast<LoopScheduleAtOp>(top);
    if (!defAt)
      continue;
    unsigned firstUse = firstIterArgUseOffset(pipOp, argIdx);
    if (firstUse == UINT_MAX)
      continue; // never read
    unsigned ready = (unsigned)defAt.getOffset() + 2;
    minStart = std::max(minStart, ready > firstUse ? ready - firstUse : 0u);
  }
  return minStart;
}

LogicalResult LoopScheduleToFSMPass::lowerLoopNodeAsModule(
    const LoopNode &node, OpBuilder &builder, Location loc,
    loopschedule::LoopScheduleFuncSequentialOp funcOp, ArrayRef<PortArgInfo> memrefArgs,
    IRMapping &parentMapping,
    hw::HWModuleOp &outModule,
    SmallVectorImpl<Value> &capturedVals) {
  // `sharedOperators` / `instanceUniquer` are scoped to the hw.module
  // CURRENTLY being emitted, and this function runs mid-emission of the
  // caller's module. Save the caller's in-progress state and restore it on
  // exit: the child's clear-and-populate must never leak back — a caller
  // op emitted after this returns would join a shared operator instance
  // living inside the CHILD module's region (invalid cross-region IR; the
  // nested-stream two_nests regression), and instance names would collide.
  auto savedShared = std::move(sharedOperators);
  auto savedUniquer = std::move(instanceUniquer);
  auto restoreScopes = llvm::make_scope_exit([&] {
    sharedOperators = std::move(savedShared);
    instanceUniquer = std::move(savedUniquer);
  });

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
  sharedOperators.clear();

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

  // Port contention is a BODY-global property, not per-frame: the frames
  // share each port's address/rd_data wires, so two frames that each make a
  // single access to the same port still contend — a later frame's access
  // retargets the port while an earlier frame's load result may still be
  // consumed (its raw rd_data mapping would silently deliver the later
  // access's data). Compute the multi-access set across ALL frames and use
  // it for every frame's SeqPortMuxCtx so each such access gets the
  // gate-muxed drive and the data-valid capture register.
  llvm::DenseSet<std::pair<Value, unsigned>> bodyMultiAccess;
  {
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> counts;
    // Walk EVERY region of each frame: await-frames keep their post-await
    // work in the `do` region, which getBodyBlock() doesn't cover.
    for (auto frameOp : frames)
      for (Region &region : frameOp->getRegions())
        for (Block &block : region)
          collectFrameAccessCounts(block, counts);
    for (auto &entry : counts)
      if (entry.second > 1)
        bodyMultiAccess.insert(entry.first);
  }

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
      unsigned offset = slot.atOffset;
      // --- Late child_start correction (CORRECTNESS, not an optimisation) ---
      // A launch reads its frame-produced operands out of the
      // launch-consumer capture registers, which latch during cycle O+1 for
      // a producer at at-offset O and are register-readable from O+2. The
      // SCHEDULE, by contrast, places a launch as soon as its operands are
      // COMBINATIONALLY available (O + operator latency), which is a cycle
      // or two too early whenever a pipeline iter_arg init comes out of the
      // same frame and is read at a small stage offset. The IV of a loop
      // with a runtime LOWER bound is exactly that case:
      // `for k in range(rowptr[i], rowptr[i+1])` inits its IV from an
      // at-result, and the IV's preload depth is forced to 0 because the
      // loop condition reads it combinationally -- so the preload samples
      // the capture register on the start edge and latches the PREVIOUS
      // outer iteration's bound. Every row then walked from the previous
      // row's base: plausible numbers, silently wrong.
      //
      // Push such a launch to the first legal cycle. Frame cycle states
      // extend to cover it, so the static ops keep their issue/capture
      // cycles. Loops whose inits are constants or frame-external values --
      // i.e. every loop with a static lower bound -- are unaffected, which
      // is why this leaves existing designs bit-identical.
      if (slot.pipIdx >= 0)
        offset = std::max(
            offset, minLegalLaunchStart(node.pipelineChildren[slot.pipIdx],
                                        frames[i]));
      launchAtOffsets.push_back(offset);
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

  // --- Early child_start peephole ---
  // A pipeline launch scheduled at at-offset K > 0 usually sits there
  // because a same-frame static op produces one of its iter_arg inits
  // (e.g. an accumulator init loaded from memory). But the pipeline does
  // not read an init until the FIRST STAGE THAT USES the corresponding
  // iter_arg (the feedback-register preload in `lowerPipelineChild`
  // samples it on the start pulse delayed by `firstIterArgUseOffset`),
  // so the start pulse can move ahead of the init's ready cycle — often
  // all the way to at-0 — overlapping the pipeline's first stages with
  // the frame's remaining preamble cycles and shaving K - newStart
  // cycles per iteration. The frame keeps its full cycle count
  // (static-op issue/capture cycles are untouched); only the
  // child_start pulse and the child_active window move.
  //
  // Guards:
  //  (a) single-launch frames with a PIPELINE child only (sequential-loop
  //      and call children sample captured inputs from their start),
  //  (b) init timing: newStart + firstUseStage(iter_arg) >= ready cycle
  //      of every same-frame-produced init (ready = the cycle its
  //      launch-consumer capture register is readable),
  //  (c) port disjointness: pipeline stages that would overlap the
  //      frame's cycle states must not touch any (port, binding) the
  //      frame's static ops drive (static addresses are held through the
  //      whole frame by the merge muxes).
  SmallVector<unsigned> launchStartOffsets(launchAtOffsets.begin(),
                                           launchAtOffsets.end());
  for (unsigned i = 0; i < numFrames; ++i) {
    auto &slots = node.frameLaunches[i];
    if (slots.size() != 1 || slots[0].pipIdx < 0)
      continue;
    // The scheduled offset AFTER the late-start correction above: the
    // peephole may only move a launch earlier within the window the
    // correction left legal.
    unsigned schedOffset = launchAtOffsets[(unsigned)frameWaitIdx[i].front()];
    if (schedOffset == 0)
      continue;
    // (d) dynamic-latency accesses in the frame: a variable-latency access
    // (an amc.burst_copy fill, a dyn AXI load) scheduled before the child
    // carries a completion dependence the scheduler encoded as the child's
    // at-offset — the access's done-stall holds its last budget cycle
    // until the access actually completes, and the child sits after that
    // budget precisely so it reads what the access produced. An
    // early-started child escapes the stall (it fires before the stalled
    // cycle is even reached) and overlaps the access: the tiled
    // burst_copy kernel's compute overlapped its fill and read a stale
    // staging buffer. The port-disjointness guard below cannot see this
    // conflict — the access's engine-owned port is a different SSA value
    // from the pipeline's ports even when both back the same ram — so be
    // conservative: any dynamic access in this frame (launch-wrapped or
    // not) keeps the child at its scheduled offset.
    bool hasDynAccess = false;
    frames[i].getBodyBlock().walk([&](Operation *op) {
      if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(op))
        return WalkResult::skip(); // the child's own body doesn't count
      bool dyn = false;
      if (auto l = dyn_cast<loopschedule::LoadInterface>(op))
        dyn = l.isDynamic();
      else if (auto s = dyn_cast<loopschedule::StoreInterface>(op))
        dyn = s.isDynamic();
      if (!dyn)
        return WalkResult::advance();
      hasDynAccess = true;
      return WalkResult::interrupt();
    });
    if (hasDynAccess)
      continue;
    auto pipOp = node.pipelineChildren[slots[0].pipIdx];
    LoopScheduleFrameOp frameOp = frames[i];
    Block &pipBlock = pipOp.getStagesBlock();

    // (b) earliest legal start from the init-timing constraints.
    unsigned minStart = 0;
    bool analyzable = true;
    for (auto [argIdx, init] : llvm::enumerate(pipOp.getInits())) {
      Operation *def = init.getDefiningOp();
      if (!def || !frameOp->isAncestor(def))
        continue; // defined before the frame — stable, no constraint
      auto defAt = dyn_cast<LoopScheduleAtOp>(def);
      if (!defAt) {
        analyzable = false;
        break;
      }
      Value inner = defAt.getYieldOp()
                        .getOperands()[cast<OpResult>(init).getResultNumber()];
      Operation *producer = inner.getDefiningOp();
      // Only memory loads are modeled (the doitgen-style accumulator
      // init); any other same-frame producer keeps the scheduled offset.
      if (!producer ||
          !isa<loopschedule::HWLoadLoweringInterface>(producer)) {
        analyzable = false;
        break;
      }
      // Load issued at the at's offset; rd_data valid one cycle later and
      // latched into the launch-consumer capture register at the end of
      // that cycle — register-readable the cycle after.
      unsigned ready = (unsigned)defAt.getOffset() + 2;
      // First pipeline stage that reads this iter_arg (shared with the
      // preload-depth computation in lowerPipelineChild).
      unsigned firstUse = firstIterArgUseOffset(pipOp, argIdx);
      if (firstUse == UINT_MAX)
        continue; // init never read
      minStart = std::max(minStart, ready > firstUse ? ready - firstUse : 0);
    }

    // (b2) earliest legal start from the values the pipeline BODY reads.
    // Inits are not the only thing a pipeline consumes from its frame: the
    // stages can reference a same-frame `at` result directly — the hoisted
    // `r: index = idx[i]` gather spelling puts the row index load in the
    // outer frame and reads it from stage 0 to form the inner address.
    // Such a value reaches the child through the launch-consumer capture
    // register, which `resultNeedsCapture` gates on cycle O+1 (O = the
    // producing at's offset), making it register-readable from cycle O+2.
    // A body stage at offset S runs in cycle `start + 1 + S` (child_start
    // pulses in `start`, `pipN_active` is its registered form), so the
    // start pulse must satisfy `start + 1 + S >= O + 2`. Without this the
    // peephole hoisted the launch on top of the load feeding it and the
    // pipeline's FIRST iteration read the PREVIOUS outer iteration's
    // value — one stale element per inner loop, silently.
    Operation *launchHolder =
        frameOp.getBodyBlock().findAncestorOpInBlock(*pipOp.getOperation());
    pipBlock.walk([&](Operation *user) {
      if (!analyzable)
        return;
      for (Value operand : user->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || !frameOp->isAncestor(def))
          continue; // defined before the frame — stable, no constraint
        Operation *defTop =
            frameOp.getBodyBlock().findAncestorOpInBlock(*def);
        if (defTop == launchHolder)
          continue; // inside our own launch: moves with the child
        auto defAt = dyn_cast_or_null<LoopScheduleAtOp>(defTop);
        // Only `at` results cross from the frame into the child; a
        // producer we cannot place in a cycle keeps the scheduled offset.
        if (!defAt) {
          analyzable = false;
          return;
        }
        // The stage that reads it (users outside any stage `at` count as
        // stage 0, matching `firstIterArgUseOffset`).
        unsigned stage = 0;
        for (Operation *anc = user; anc && anc != pipOp.getOperation();
             anc = anc->getParentOp())
          if (auto stageAt = dyn_cast<LoopScheduleAtOp>(anc)) {
            stage = (unsigned)stageAt.getOffset();
            break;
          }
        unsigned readable = (unsigned)defAt.getOffset() + 1;
        minStart = std::max(minStart, readable > stage ? readable - stage : 0);
      }
    });

    if (!analyzable || minStart >= schedOffset)
      continue;

    // (c) port disjointness over the overlap window.
    DenseSet<std::pair<Value, unsigned>> framePortSet;
    for (auto atOp : frameOp.getBodyBlock().getOps<LoopScheduleAtOp>()) {
      bool isLaunchHolder = llvm::any_of(
          atOp.getBodyBlock(),
          [](Operation &op) { return isa<LoopScheduleLaunchOp>(op); });
      if (isLaunchHolder)
        continue;
      atOp.walk([&](Operation *op) {
        if (auto l = dyn_cast<loopschedule::HWLoadLoweringInterface>(op))
          framePortSet.insert({l.getMemoryValue(), getBindingPort(op)});
        else if (auto s =
                     dyn_cast<loopschedule::HWStoreLoweringInterface>(op))
          framePortSet.insert({s.getMemoryValue(), getBindingPort(op)});
      });
    }
    bool portConflict = false;
    unsigned window = schedOffset - minStart;
    for (auto stageAt : pipBlock.getOps<LoopScheduleAtOp>()) {
      if ((unsigned)stageAt.getOffset() >= window)
        continue;
      stageAt.walk([&](Operation *op) {
        Value mem;
        if (auto l = dyn_cast<loopschedule::HWLoadLoweringInterface>(op))
          mem = l.getMemoryValue();
        else if (auto s =
                     dyn_cast<loopschedule::HWStoreLoweringInterface>(op))
          mem = s.getMemoryValue();
        else
          return;
        if (framePortSet.contains({mem, getBindingPort(op)}))
          portConflict = true;
      });
    }
    if (portConflict)
      continue;
    launchStartOffsets[(unsigned)frameWaitIdx[i].front()] = minStart;
  }

  // --- Fold a trailing iter-advance-only frame ---
  // SCFToLoopSchedule ends every sequential loop with a frame that only
  // computes the next iter_arg values (e.g. `iv + 1` feeding an
  // iter_arg_update). Those ops are pure comb into the iter_arg
  // registers' D inputs and the frame's results are read only through
  // the terminator (feedback + loop results), both of which use the
  // combinational alias — so the frame needs no FSM state of its own.
  // Folding it fires iter_advance in the previous frame's exit instead,
  // saving one cycle per loop iteration. Bail when the frame produces
  // the loop condition (COND consumes it from a state that would no
  // longer exist), launches children, spans multiple cycles, or touches
  // memory.
  bool foldLastFrame = false;
  if (numFrames >= 2 && frameWaitIdx[numFrames - 1].empty() &&
      frameLatencies[numFrames - 1] == 1) {
    LoopScheduleFrameOp lastFrame = frames[numFrames - 1];
    bool pure = terminatorOp.getCondition().getDefiningOp() !=
                lastFrame.getOperation();
    if (pure) {
      lastFrame.getBodyBlock().walk([&](Operation *op) {
        if (isa<LoopScheduleAtOp, LoopScheduleYieldOp,
                LoopScheduleIterArgUpdateOp>(op))
          return WalkResult::advance();
        if (isMemoryEffectFree(op))
          return WalkResult::advance();
        pure = false;
        return WalkResult::interrupt();
      });
    }
    foldLastFrame = pure;
  }

  // --- Decide the COND bypass ---
  // With the bypass the last state loops straight back into FRAME_0 (or
  // out to DONE) guarded on `cond_next` — the loop condition evaluated on
  // the iter_arg registers' D wires — skipping the per-iteration COND
  // state. That is only sound when the condition is a pure combinational
  // function of the iter_args and loop-invariant values: a load, a
  // launch result, or an operator that lowers to a registered
  // hw.instance would not have its next-iteration value available on the
  // advance edge. Walk the condition's backward slice (through frame/at
  // result indirection) and check every op lowers combinationally.
  auto lowersCombinationally = [&](Operation *op) -> bool {
    if (isFreeArithOp(op))
      return true;
    auto operatorAttr =
        op->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    if (!operatorLibrary || !operatorAttr)
      return getOpCycleLatency(op) <= 1; // plain-clone fallback path
    StringRef opName = operatorLibrary->getOperatorBySymbol(operatorAttr);
    oplib::HwMatchOp hwMatch = operatorLibrary->getHwMatchOp(opName);
    if (!hwMatch)
      return false;
    for (auto &o : *hwMatch.getBodyBlock())
      if (isa<oplib::HwInstanceOp>(o))
        return false;
    return true;
  };
  bool condBypass = [&]() -> bool {
    SmallVector<Value> stack{terminatorOp.getCondition()};
    DenseSet<Value> visited;
    while (!stack.empty()) {
      Value v = stack.pop_back_val();
      if (!visited.insert(v).second)
        continue;
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        Block *owner = barg.getOwner();
        if (owner == &seqOp.getScheduleBlock())
          continue; // iter_arg — substituted with its feedback wire
        if (auto ownerFrame =
                dyn_cast<LoopScheduleFrameOp>(owner->getParentOp())) {
          // Await-frame body arg: follow the await-yield operand.
          stack.push_back(
              ownerFrame.getAwaitYield().getOperands()[barg.getArgNumber()]);
          continue;
        }
        if (!seqOp->isAncestor(owner->getParentOp()))
          continue; // defined above the loop — invariant
        return false;
      }
      Operation *def = v.getDefiningOp();
      if (!seqOp->isAncestor(def))
        continue; // invariant
      if (auto defFrame = dyn_cast<LoopScheduleFrameOp>(def)) {
        auto yieldOp2 = cast<LoopScheduleYieldOp>(
            defFrame.getBodyBlock().getTerminator());
        stack.push_back(yieldOp2.getOperands()[cast<OpResult>(v)
                                                   .getResultNumber()]);
        continue;
      }
      if (auto defAt = dyn_cast<LoopScheduleAtOp>(def)) {
        stack.push_back(defAt.getYieldOp().getOperands()[cast<OpResult>(v)
                                                             .getResultNumber()]);
        continue;
      }
      if (def->getNumRegions() != 0 || !isMemoryEffectFree(def) ||
          !lowersCombinationally(def))
        return false;
      for (Value o : def->getOperands())
        stack.push_back(o);
    }
    return true;
  }();

  // --- Create FSM machine ---
  std::string fsmName = node.prefix + "_fsm";
  builder.setInsertionPointToEnd(moduleOp.getBody());
  (void)createSequentialFSM(builder, loc, fsmName, numFrames, waitFrameIndices,
                            launchAtOffsets, launchStartOffsets,
                            frameLatencies, foldLastFrame, condBypass,
                            condBypass && entryInitsReadyAtStart(seqOp));

  // --- Create FSM instance with backedges ---
  hw.setInsertionPointToEnd(hwBody);
  BackedgeBuilder bb(hw, loc);

  Backedge condBE = bb.get(i1);
  // cond re-evaluated on the iter_arg feedback wires; resolved after the
  // feedback values exist (constant 0 when the bypass is off).
  Backedge condNextBE = bb.get(i1);

  // One backedge per wait step for its child_done input.
  SmallVector<Backedge> childDoneBEs;
  childDoneBEs.reserve(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    childDoneBEs.push_back(bb.get(i1));

  // Stall input: OR of the per-dynamic-access ready/done stall terms
  // collected while lowering frame bodies (resolved below).
  Backedge stallBE = bb.get(i1);
  SmallVector<Value> seqStallTerms;
  unsigned seqDynCounter = 0;

  // Build instance inputs: start, cond, cond_next, child_done_0..C-1,
  // stall.
  SmallVector<Value> instInputs;
  instInputs.push_back(startSignal);
  instInputs.push_back(Value(condBE));
  instInputs.push_back(Value(condNextBE));
  for (auto &be : childDoneBEs)
    instInputs.push_back(Value(be));
  instInputs.push_back(Value(stallBE));

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

  // While stalled (a dynamic access waiting on ready/done), everything
  // that advances with the FSM must hold — the FSM freezes its state, and
  // the module-side registers freeze via CE &= notStall.
  Value notStallSeq = comb::createOrFoldNot(hw, loc, Value(stallBE));

  // Raw per-trip advance term: the FSM's iter_advance output. When the
  // last state-emitting frame ends in a WAIT there is no state left to
  // host a Moore iter_advance (WAIT has no POST settle state, and
  // asserting iter_advance throughout a multi-cycle WAIT would re-latch
  // the iter_args every cycle) — compute the advance from the WAIT exit
  // condition itself: in-WAIT (a launch is active but no cycle state is)
  // AND every one of the frame's child dones — exactly the FSM's WAIT_i
  // exit guard, so the latch fires on the same edge the state leaves
  // WAIT.
  Value advanceRaw = fsmIterAdvance;
  unsigned lastStateFrame = foldLastFrame ? numFrames - 2 : numFrames - 1;
  if (!frameWaitIdx[lastStateFrame].empty()) {
    Value notFrameCycle = comb::createOrFoldNot(
        hw, loc, fsmFrameActives[lastStateFrame]);
    Value inWait = comb::AndOp::create(
        hw, loc,
        fsmChildActives[(unsigned)frameWaitIdx[lastStateFrame].front()],
        notFrameCycle);
    Value allDones;
    for (int j : frameWaitIdx[lastStateFrame]) {
      Value d = Value(childDoneBEs[(unsigned)j]);
      allDones = allDones ? comb::AndOp::create(hw, loc, allDones, d) : d;
    }
    Value advTerm = comb::AndOp::create(hw, loc, inWait, allDones);
    advanceRaw = comb::OrOp::create(hw, loc, advanceRaw, advTerm);
  }
  // The advance EDGE (advance & !stall): the posedge on which the
  // iter_args latch and the state loops back. Also clears the seq-dyn
  // one-shot latches — with the COND bypass, frameActive stays high
  // across back-to-back iterations, so this edge is their only
  // between-iterations reset.
  Value iterAdvanceEdge =
      comb::AndOp::create(hw, loc, advanceRaw, notStallSeq);
  // Clock enable for iter_arg registers: advance exactly once per loop
  // trip. first_iter forces the init load on entry to a new loop
  // invocation.
  Value ce = comb::OrOp::create(hw, loc, advanceRaw, fsmFirstIter);
  ce = comb::AndOp::create(hw, loc, ce, notStallSeq);

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
    // A folded last frame has no FSM state (its frame_active output is
    // never driven), and its results are only consumed combinationally
    // through the terminator — skip the dead capture registers.
    if (foldLastFrame && i + 1 == numFrames)
      continue;
    // Freeze under stall: the gate is a held FSM output while stalled, and
    // capturing mid-stall would latch values before a dynamic access's
    // done delivered them.
    frameCaptureGate[i] = comb::AndOp::create(
        hw, loc, fsmFrameCycleGates[i].back(), notStallSeq);
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
      perFramePorts[i][memInfo.originalArg].done =
          localMemPorts[memInfo.originalArg].done;
      perFramePorts[i][memInfo.originalArg].wrDone =
          localMemPorts[memInfo.originalArg].wrDone;
      perFramePorts[i][memInfo.originalArg].ready =
          localMemPorts[memInfo.originalArg].ready;
      perFramePorts[i][memInfo.originalArg].rdIdle =
          localMemPorts[memInfo.originalArg].rdIdle;
      perFramePorts[i][memInfo.originalArg].wrIdle =
          localMemPorts[memInfo.originalArg].wrIdle;
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

    // Per-port read-completion chain for this frame (see SeqDynCtx).
    DenseMap<Value, Value> framePrevReadSeen;

    // Contended-port handling for the static accesses lowered below (see
    // SeqPortMuxCtx). Launched children are excluded by the collector —
    // they drive their own port slots through the child-instance muxing.
    SeqPortMuxCtx seqMux;
    seqMux.multi = bodyMultiAccess;
    for (auto &memInfo : memrefArgs)
      seqMux.registeredMems.insert(memInfo.originalArg);
    seqMux.cycleGates = fsmFrameCycleGates[frameIdx];
    seqMux.clk = clk;
    seqMux.rst = rst;
    seqMux.regPrefix = node.prefix + "_f" + std::to_string(frameIdx);
    SeqPortMuxCtx *seqMuxPtr = seqMux.multi.empty() ? nullptr : &seqMux;

    // Is this at-op's `res` consumed at an at-offset the capture register
    // can actually serve, or inside a launch-holder at-op in the same
    // frame? The register is clocked at `myOffset + 1` and therefore only
    // READABLE from `myOffset + 2` on, so only a consumer that far out
    // needs it; a consumer at exactly `myOffset + 1` keeps the
    // combinational value.
    //
    // That is safe because an at-result's combinational mapping stays
    // valid from `myOffset + 1` through the REST of the frame's cycle
    // states, not just for one cycle: a port with one static access holds
    // its address frame-wide (see SeqPortMuxCtx), a contended port's load
    // result is already a live-bypassed capture register, a dynamic
    // access's result is a completion-clocked `_cap` register, and
    // anything else is a pure function of frame-stable values. What the
    // hold register buys is validity ACROSS the frame boundary — during
    // WAIT_i the launched child owns the port and rd_data moves — which is
    // why a launch-holder consumer still counts at any offset: the child
    // runs after every cycle state of the frame.
    auto resultNeedsCapture = [&](Value atResult, unsigned myOffset) {
      for (auto *user : atResult.getUsers()) {
        Operation *anc = frameBody.findAncestorOpInBlock(*user);
        if (!anc)
          continue;
        auto otherAt = dyn_cast<LoopScheduleAtOp>(anc);
        if (!otherAt)
          continue;
        if ((unsigned)otherAt.getOffset() > myOffset + 1)
          return true;
        for (auto &op : otherAt.getBodyBlock())
          if (isa<LoopScheduleLaunchOp>(&op))
            return true;
      }
      return false;
    };

    // Capture registers created for this frame's at-results, each with the
    // at-offset from which its value is READABLE (the cycle after its
    // clock enable). Applied to `localMapping` lazily — see the flush at
    // the top of the at-op loop — because rewriting the mapping the moment
    // the register is created would hand the register to the at-op that
    // executes in the register's own clock cycle, which reads the value
    // the PREVIOUS loop iteration left behind.
    struct PendingCapture {
      unsigned readableFrom;
      Value atResult;
      Value reg;
    };
    SmallVector<PendingCapture> pendingCaptures;
    auto flushCaptures = [&](unsigned upToOffset) {
      llvm::erase_if(pendingCaptures, [&](const PendingCapture &pc) {
        if (pc.readableFrom > upToOffset)
          return false;
        localMapping.map(pc.atResult, pc.reg);
        return true;
      });
    };

    // Results produced by a dynamic (seqDyn) access in this frame. Their
    // value is already held in a completion-clocked `_cap` register, so the
    // fixed-offset result-capture latch below must NOT re-sample them (it
    // would fire at atOffset+1, before the variable-latency access actually
    // completes, and capture a stale beat).
    llvm::DenseSet<Value> frameSeqDynResults;

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
      // Hand over any capture register that has become readable by this
      // cycle. Everything still pending is a register clocked in THIS
      // cycle or later; this at-op's operands must stay combinational.
      flushCaptures(atOffset);
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
        if (isSeqDynAccess(&op)) {
          SeqDynCtx seqDyn{clk,
                           rst,
                           hwBody,
                           &bb,
                           fsmFrameCycleGates[frameIdx],
                           fsmFrameActives[frameIdx],
                           &seqStallTerms,
                           node.prefix,
                           &seqDynCounter,
                           iterAdvanceEdge,
                           &framePrevReadSeen};
          if (failed(lowerSeqDynAccess(&op, hw, localMapping, framePorts,
                                       atGate, atOffset, &seqDyn)))
            return failure();
          for (Value r : op.getResults())
            frameSeqDynResults.insert(r);
          continue;
        }
        if (auto loadOp =
                dyn_cast<loopschedule::HWLoadLoweringInterface>(&op)) {
          if (seqMuxPtr)
            seqMuxPtr->cycle = atOffset;
          if (failed(handleHWLoad(loadOp, hw, localMapping, framePorts,
                                     atGate, seqMuxPtr)))
            return failure();
          continue;
        }
        if (auto storeOp =
                dyn_cast<loopschedule::HWStoreLoweringInterface>(&op)) {
          if (seqMuxPtr)
            seqMuxPtr->cycle = atOffset;
          if (failed(handleHWStore(storeOp, hw, localMapping, atGate,
                                      framePorts, seqMuxPtr)))
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
        Value captureGate = comb::AndOp::create(
            hw, loc, fsmFrameCycleGates[frameIdx][atOffset + 1],
            notStallSeq);
        hw.setInsertionPointToEnd(hwBody);
        for (auto res : atOp.getResults()) {
          if (!resultNeedsCapture(res, atOffset))
            continue;
          // A dynamic access already latched its data into a stable
          // completion-clocked register; re-capturing at the fixed
          // atOffset+1 cycle would read it before the access completes.
          // The consumer already maps to that register, so leave it be.
          Value yielded = atYield.getOperands()[res.getResultNumber()];
          if (frameSeqDynResults.contains(yielded))
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
          // Readable from atOffset + 2: the enable is atOffset + 1's gate.
          pendingCaptures.push_back({atOffset + 2, res, latched});
        }
      }
    }
    // Everything left over is readable by the time the frame's own cycle
    // states are done: the frame yield, the launched child (which runs in
    // WAIT_i) and later frames all read the held value.
    flushCaptures(std::numeric_limits<unsigned>::max());
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

        // A blocking dynamic access (e.g. an amc.burst_copy) whose
        // completion budget ends in the state that launches this child may
        // still be draining: the done-stall holds the STATE, but the entry
        // pulse alone would launch the child concurrently with the access.
        // Defer the start until the stall clears, latching the pulse so it
        // is not lost mid-stall. Without an active stall this is a
        // passthrough (fire == the original pulse).
        {
          hw.setInsertionPointToEnd(hwBody);
          Value zero1 = hw::ConstantOp::create(hw, loc, i1, 0);
          auto pendReg = seq::CompRegOp::create(
              hw, loc, zero1, clk, rst, zero1,
              hw.getStringAttr(node.prefix + "_w" + std::to_string(waitIdx) +
                               "_start_pend"));
          Value raw = comb::OrOp::create(hw, loc, slotChildStart,
                                         pendReg.getResult(), false);
          Value fire = comb::AndOp::create(hw, loc, raw, notStallSeq, false);
          Value notFire = comb::createOrFoldNot(hw, loc, fire);
          pendReg->setOperand(
              0, comb::AndOp::create(hw, loc, raw, notFire, false));
          slotChildStart = fire;
        }

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
          // Mirror the loop module's `mem*_done` input order.
          for (auto &memInfo : memrefArgs) {
            if (!memInfo.isAmcPort || !memInfo.hasDone)
              continue;
            auto &callerMp =
                perFramePorts[frameIdx][memInfo.originalArg];
            Value d = callerMp.done
                          ? callerMp.done
                          : hw::ConstantOp::create(hw, loc,
                                                     hw.getI1Type(), 0);
            childInputs.push_back(d);
          }
          // Mirror the loop module's `mem*_wr_done` input order.
          for (auto &memInfo : memrefArgs) {
            if (!memInfo.isAmcPort || !memInfo.hasWrDone)
              continue;
            auto &callerMp =
                perFramePorts[frameIdx][memInfo.originalArg];
            Value d = callerMp.wrDone
                          ? callerMp.wrDone
                          : hw::ConstantOp::create(hw, loc,
                                                     hw.getI1Type(), 0);
            childInputs.push_back(d);
          }
          // Mirror the loop module's `mem*_ready` input order. A missing
          // caller signal ties acceptance high (no backpressure).
          for (auto &memInfo : memrefArgs) {
            if (!memInfo.isAmcPort || !memInfo.hasReady)
              continue;
            auto &callerMp =
                perFramePorts[frameIdx][memInfo.originalArg];
            Value rdy = callerMp.ready
                            ? callerMp.ready
                            : hw::ConstantOp::create(hw, loc,
                                                       hw.getI1Type(), 1);
            childInputs.push_back(rdy);
          }
          // Mirror the loop module's `mem*_rd_idle` / `mem*_wr_idle` input
          // order. Missing caller signals tie the drain high (fence no-op).
          for (auto &memInfo : memrefArgs) {
            if (!memInfo.isAmcPort || !memInfo.hasRdIdle)
              continue;
            auto &callerMp = perFramePorts[frameIdx][memInfo.originalArg];
            Value v = callerMp.rdIdle
                          ? callerMp.rdIdle
                          : hw::ConstantOp::create(hw, loc, hw.getI1Type(), 1);
            childInputs.push_back(v);
          }
          for (auto &memInfo : memrefArgs) {
            if (!memInfo.isAmcPort || !memInfo.hasWrIdle)
              continue;
            auto &callerMp = perFramePorts[frameIdx][memInfo.originalArg];
            Value v = callerMp.wrIdle
                          ? callerMp.wrIdle
                          : hw::ConstantOp::create(hw, loc, hw.getI1Type(), 1);
            childInputs.push_back(v);
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
      DenseMap<Value, Value> framePrevReadSeen;
      SeqDynCtx seqDyn{clk,
                       rst,
                       hwBody,
                       &bb,
                       fsmFrameCycleGates[frameIdx],
                       fsmFrameActives[frameIdx],
                       &seqStallTerms,
                       node.prefix,
                       &seqDynCounter,
                       iterAdvanceEdge,
                       &framePrevReadSeen};
      SeqPortMuxCtx seqMux;
      seqMux.multi = bodyMultiAccess;
      for (auto &memInfo : memrefArgs)
        seqMux.registeredMems.insert(memInfo.originalArg);
      seqMux.cycleGates = fsmFrameCycleGates[frameIdx];
      seqMux.clk = clk;
      seqMux.rst = rst;
      seqMux.regPrefix = node.prefix + "_f" + std::to_string(frameIdx);
      if (failed(lowerFrameBody(&frameOp.getBodyBlock(), hw, localMapping,
                                 fsmFrameCycleGates[frameIdx],
                                 perFramePorts[frameIdx], moduleOp, clk,
                                 rst, &seqDyn,
                                 seqMux.multi.empty() ? nullptr : &seqMux,
                                 /*opCE=*/notStallSeq)))
        return failure();
    }

    // Detect whether this frame contains any memref loads. All memref
    // reads have latency=1 (local hlmems by construction, external ports
    // by the BRAM-port contract): `rd_data` during cycle N reflects the
    // addr from cycle N-1. So during the cycle where frame_active_<i> is
    // high, rd_data is stale (reflecting the prior state's addr). We must
    // capture rd_data ONE CYCLE LATER, when it correctly reflects the addr
    // driven during the frame. For frames without loads, the normal gate
    // works fine.
    bool frameHasMemLoad = false;
    frameOp->walk([&](LoopScheduleLoadOp loadOp) {
      frameHasMemLoad = true;
      return WalkResult::interrupt();
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
      if (frameHasMemLoad) {
        Value falseConstCap =
            hw::ConstantOp::create(hw, loc, hw.getI1Type(), 0);
        captureGate = seq::CompRegClockEnabledOp::create(
            hw, loc, frameCaptureGate[frameIdx], clk, notStallSeq, rst,
            falseConstCap,
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

  // Resolve the FSM stall input: OR of every dynamic access's ready/done
  // stall term (constant 0 when the loop has no dynamic accesses — the
  // FSM then behaves exactly as before).
  {
    hw.setInsertionPointToEnd(hwBody);
    Value stallVal;
    if (seqStallTerms.empty()) {
      stallVal = hw::ConstantOp::create(hw, loc, i1, 0);
    } else {
      stallVal = seqStallTerms.front();
      for (unsigned k = 1; k < seqStallTerms.size(); ++k)
        stallVal =
            comb::OrOp::create(hw, loc, stallVal, seqStallTerms[k], false);
    }
    stallBE.setValue(stallVal);
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
  SmallVector<Value> iterArgFeedbacks(iterArgRegs.size());
  for (unsigned i = 0; i < iterArgRegs.size(); ++i) {
    Value termArg = iterArgPhaseResults[i];
    auto combIt = frameResultComb.find(termArg);
    Value feedback = combIt != frameResultComb.end()
                         ? combIt->second
                         : localMapping.lookup(termArg);
    iterArgFeedbacks[i] = feedback;
    Value init = localMapping.lookup(seqOp.getInits()[i]);
    Value muxed = comb::MuxOp::create(hw, loc, fsmFirstIter, init, feedback);
    iterArgRegs[i].getDefiningOp()->setOperand(0, muxed);
  }

  // --- Resolve cond_next ---
  // Re-lower the condition's source slice with every iter_arg mapped to
  // its feedback wire (the value latching on the advance edge). The
  // bypass pre-analysis guaranteed the slice is pure comb of iter_args
  // and loop-invariant values, so emitComputeOp only clones comb ops.
  if (!condBypass) {
    condNextBE.setValue(hw::ConstantOp::create(hw, loc, i1, 0));
  } else {
    IRMapping condNextMapping;
    std::function<Value(Value)> resolveNext = [&](Value v) -> Value {
      if (Value m = condNextMapping.lookupOrNull(v))
        return m;
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        Block *owner = barg.getOwner();
        if (owner == &seqOp.getScheduleBlock()) {
          Value f = iterArgFeedbacks[barg.getArgNumber()];
          condNextMapping.map(v, f);
          return f;
        }
        if (auto ownerFrame =
                dyn_cast<LoopScheduleFrameOp>(owner->getParentOp())) {
          Value r = resolveNext(
              ownerFrame.getAwaitYield().getOperands()[barg.getArgNumber()]);
          condNextMapping.map(v, r);
          return r;
        }
        Value m = localMapping.lookup(v);
        condNextMapping.map(v, m);
        return m;
      }
      Operation *def = v.getDefiningOp();
      if (!seqOp->isAncestor(def)) {
        Value m = localMapping.lookup(v);
        condNextMapping.map(v, m);
        return m;
      }
      if (auto defFrame = dyn_cast<LoopScheduleFrameOp>(def)) {
        auto yieldOp2 = cast<LoopScheduleYieldOp>(
            defFrame.getBodyBlock().getTerminator());
        Value r = resolveNext(
            yieldOp2.getOperands()[cast<OpResult>(v).getResultNumber()]);
        condNextMapping.map(v, r);
        return r;
      }
      if (auto defAt = dyn_cast<LoopScheduleAtOp>(def)) {
        Value r = resolveNext(defAt.getYieldOp()
                                  .getOperands()[cast<OpResult>(v)
                                                     .getResultNumber()]);
        condNextMapping.map(v, r);
        return r;
      }
      for (Value o : def->getOperands())
        (void)resolveNext(o);
      hw.setInsertionPointToEnd(hwBody);
      if (failed(emitComputeOp(def, hw, condNextMapping, moduleOp, clk, rst,
                               /*opCE=*/{})))
        llvm_unreachable("cond_next slice op failed to lower; the bypass "
                         "pre-analysis admitted a non-comb op");
      return condNextMapping.lookup(v);
    };
    condNextBE.setValue(resolveNext(terminatorOp.getCondition()));
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
  // For amc ports we also copy `rdEn` so the stall-aware gate built in
  // `lowerPipelineChild` from the per-port `done` handshake propagates
  // through the loop module's `mem*_rd_en` output — without this, the
  // output defaults to 0 and the AMC arbiter never sees a request.
  for (auto &memInfo : memrefArgs) {
    localMemPorts[memInfo.originalArg].addrs =
        mergedMemPorts[memInfo.originalArg].addrs;
    localMemPorts[memInfo.originalArg].wrData =
        mergedMemPorts[memInfo.originalArg].wrData;
    localMemPorts[memInfo.originalArg].wrEn =
        mergedMemPorts[memInfo.originalArg].wrEn;
    if (memInfo.requiresRdEn)
      localMemPorts[memInfo.originalArg].rdEn =
          mergedMemPorts[memInfo.originalArg].rdEn;
  }

  // --- Build module output ---
  hw.setInsertionPointToEnd(hwBody);
  // Early done: the loop is finished exactly when it takes an advance
  // edge with a false next-condition — one cycle before the FSM's Moore
  // DONE output. Cutting done through on that edge lets the parent's
  // WAIT exit a cycle earlier per invocation. Both operands already
  // exist (no new logic depth beyond one AND/OR).
  //
  // Only sound when the loop's results are dead: the advance edge is the
  // SAME posedge the result (iter_arg) registers latch their final
  // values, so a parent that consumed our results on its own advance
  // edge would read the pre-edge (stale) value. Loops whose results feed
  // downstream consumers keep the Moore DONE (one settle cycle).
  bool resultsDead = llvm::none_of(seqOp.getResults(), [](Value r) {
    return loopScheduleValueConsumed(r);
  });
  Value moduleDone = fsmDone;
  if (condBypass && resultsDead) {
    Value notCondNext =
        comb::createOrFoldNot(hw, loc, Value(condNextBE));
    Value earlyDone =
        comb::AndOp::create(hw, loc, iterAdvanceEdge, notCondNext);
    moduleDone = comb::OrOp::create(hw, loc, fsmDone, earlyDone);
  }
  buildLoopModuleOutput(hw, loc, localMemPorts, memrefArgs, moduleDone,
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
    unsigned issueStageOffset;
    bool isLoad;
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
    auto issue = launch->getParentOfType<LoopScheduleAtOp>();
    if (!issue)
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
    // Dynamic loads and dynamic stores are both wrapped in launch/expect; the
    // stall machinery only needs the memory port value, which both interfaces
    // expose. Loads additionally get a small data FIFO (see below).
    Value memVal;
    bool isLoad = false;
    if (auto loadIface = dyn_cast<HWLoadLoweringInterface>(payload)) {
      memVal = loadIface.getMemoryValue();
      isLoad = true;
    } else if (auto storeIface = dyn_cast<HWStoreLoweringInterface>(payload)) {
      memVal = storeIface.getMemoryValue();
    } else {
      return;
    }
    expectInfos.push_back({(unsigned)dest.getOffset(),
                           (unsigned)issue.getOffset(), isLoad, memVal});
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

  // Per-expect completion tracking. The memory emits an honest 1-cycle
  // pulse on `done` per request, with read data valid ON the pulse (the
  // on-chip done chain matches the data latency; the AXI adapter delays
  // done to its registered rd_data). With the expect planted `latency`
  // stages after the issue, several iterations' requests are in flight
  // on one port at once, so a single sticky bit cannot attribute dones
  // to iterations. Instead each expect owns a small counter plus — for
  // loads — a data FIFO sized to the issue→dest stage distance:
  //   * every done pulse increments the counter and pushes the live
  //     rdData;
  //   * the consume cycle (stageCE[dest] & !stall) pops;
  //   * the expect's stage stalls only while the counter is empty AND
  //     no done is arriving this cycle, so on-time dones (on-chip
  //     single-consumer ports) pass through with zero overhead via the
  //     combinational empty-FIFO bypass.
  // Attribution is positional: a port serves requests in issue order,
  // so the k-th done belongs to the k-th unconsumed iteration. When
  // several expects share one port we must additionally know WHICH expect
  // issued the k-th request; see the issue-window analysis below (a
  // rotation for a window narrower than II, an issue-order tag FIFO
  // otherwise). One done per expect per iteration either way —
  // if-predicated accesses on a shared port would break the count and are
  // not supported.
  //
  // Posted (no_wait) accesses on ports with acceptance backpressure: each
  // entry is (un-stall-gated issue gate, port ready). The Phase 3B stall
  // ORs in `rawGate & !ready` so a posted request is never dropped by a
  // busy memory; the enable itself is gated `& ready` at the issue site.
  SmallVector<std::pair<Value, Value>> postedReadyStalls;
  // Per-expect combinational "may consume this cycle" levels, indexed
  // like expectInfos; null when the expect has no done to wait on.
  SmallVector<Value> expectValid(expectInfos.size());
  // Load-data splices to apply to the issue stage's per-stage port copy
  // once `perStagePorts` is initialized below.
  struct DataSplice {
    unsigned issueStage;
    Value portValue;
    Value data;
  };
  SmallVector<DataSplice> dataSplices;
  // Group expects by (port, done stream) for done distribution. Ports with
  // a split write-done (read+write dyn faces, e.g. an rw AXI port) put load
  // expects on the read `done` and store expects on `wrDone`: each
  // direction completes in issue order WITHIN itself, but cross-direction
  // interleaving (pipeline fill issues several loads before the first
  // store's stage is reached) would break a shared positional rotation.
  llvm::MapVector<std::pair<Value, unsigned>, SmallVector<unsigned>>
      portExpects;
  for (auto [eIdx, ei] : llvm::enumerate(expectInfos)) {
    auto it = memPorts.find(ei.portValue);
    if (it == memPorts.end() || !it->second.done)
      continue;
    if (ei.destStageOffset >= stages.size())
      continue;
    unsigned wrStream = (!ei.isLoad && it->second.wrDone) ? 1 : 0;
    portExpects[{ei.portValue, wrStream}].push_back(eIdx);
  }
  for (auto &[groupKey, eIdxs] : portExpects) {
    Value portValue = groupKey.first;
    auto &groupPort = memPorts.find(portValue)->second;
    Value done = groupKey.second ? groupPort.wrDone : groupPort.done;
    llvm::stable_sort(eIdxs, [&](unsigned a, unsigned b) {
      return expectInfos[a].issueStageOffset <
             expectInfos[b].issueStageOffset;
    });
    // Done distribution for ports shared by several expects.
    //
    // THE INVARIANT: responses on a single-ID port come back in REQUEST
    // ISSUE ORDER, so the k-th done belongs to the k-th request the port
    // accepted. What needs deciding is which expect issued that request.
    //
    // Expect r issues in cycle o_r + II*i (o_r = its issue at-offset, i =
    // the iteration), and a stall freezes every stage together, so the
    // order in which the port sees requests is the static sort of
    // {o_r + II*i}. That sequence is the plain cyclic rotation
    // o_0, o_1, ..., o_{k-1}, o_0, ... exactly when the expects' whole
    // ISSUE WINDOW fits inside one II:
    //
    //     max(o_r) - min(o_r) < II
    //
    // i.e. every iteration's requests are all issued before the next
    // iteration's first one. That is the vadd/saxpy shape on a shared
    // bundle — two INDEPENDENT reads, which the modulo scheduler packs
    // into ADJACENT at-offsets (window 1, II 2) — and there the rotation
    // is exact, prologue and epilogue included (iteration 0 issues all k
    // in order; the last iteration likewise), so keep it: it is one
    // log2(k)-bit register.
    //
    // A DEPENDENT pair does not fit. `x[colidx[j]]`'s address IS
    // `colidx[j]`'s data, so the scheduler cannot issue it until that
    // read's expect — a whole `--axi-port-latency` window downstream,
    // plus a cycle of address arithmetic — while II stays at 2 because
    // the port still only accepts one request per cycle. The window is 17
    // wide, so the port carries floor(window/II) requests from the
    // earlier expect before the later expect's first one and a rotation
    // is off by that many for the rest of the run: every response lands
    // in the wrong expect's FIFO and the loop reads plausible garbage.
    //
    // For a window that wide, CARRY THE TAG instead of assuming it: a
    // FIFO of expect ranks, pushed on each expect's issue pulse, popped
    // by each done. Its head names the expect the arriving done belongs
    // to — in true issue order, through prologue, epilogue and stalls
    // alike. `gatedStageCE[o_r]` is exactly that issue pulse: a dynamic
    // access's enable is `gatedStageCE & ready` and Phase 3B stalls on
    // `stageCE[o_r] & !ready`, so gatedStageCE[o_r] high implies ready
    // high and the two coincide cycle for cycle. The head is a plain
    // register with no bypass, which is sound because a done never
    // arrives in the same cycle as the request it answers (the expect
    // sits at least one stage past its launch) — and it keeps the tag out
    // of the `notStall` cone.
    SmallVector<Value> doneFor(eIdxs.size(), done);
    if (eIdxs.size() > 1) {
      unsigned issueWindow = expectInfos[eIdxs.back()].issueStageOffset -
                             expectInfos[eIdxs.front()].issueStageOffset;
      if (issueWindow < II) {
        unsigned k = eIdxs.size();
        auto rrTy = IntegerType::get(ctx, llvm::Log2_64_Ceil(k));
        Value rrZero = hw::ConstantOp::create(hwBuilder, loc, rrTy, 0);
        Value rrLast = hw::ConstantOp::create(hwBuilder, loc, rrTy, k - 1);
        Value rrOne = hw::ConstantOp::create(hwBuilder, loc, rrTy, 1);
        Backedge rrBE = bb.get(rrTy);
        Value rr = Value(rrBE);
        Value atLast = comb::ICmpOp::create(
            hwBuilder, loc, comb::ICmpPredicate::eq, rr, rrLast);
        Value rrInc = comb::MuxOp::create(
            hwBuilder, loc, atLast, rrZero,
            comb::AddOp::create(hwBuilder, loc, rr, rrOne, false));
        Value rrHeld = comb::MuxOp::create(hwBuilder, loc, done, rrInc, rr);
        Value rrNext =
            comb::MuxOp::create(hwBuilder, loc, startSignal, rrZero, rrHeld);
        auto rrName = hwBuilder.getStringAttr(
            (namePrefix + "_done_rr_" + std::to_string(eIdxs.front())).str());
        Value rrReg = seq::CompRegOp::create(hwBuilder, loc, rrNext, clk, rst,
                                             rrZero, rrName);
        rrBE.setValue(rrReg);
        for (unsigned rank = 0; rank < k; ++rank) {
          Value isMine = comb::ICmpOp::create(
              hwBuilder, loc, comb::ICmpPredicate::eq, rrReg,
              hw::ConstantOp::create(hwBuilder, loc, rrTy, rank));
          doneFor[rank] = comb::AndOp::create(hwBuilder, loc, done, isMine);
        }
      } else {
        unsigned k = eIdxs.size();
        if (llvm::any_of(eIdxs, [&](unsigned e) {
              return expectInfos[e].issueStageOffset >= stages.size();
            }))
          return pipOp.emitError(
              "dynamic access issues outside the pipeline's stage list");
        auto tagTy = IntegerType::get(ctx, llvm::Log2_64_Ceil(k));
        // Depth. Expect r can hold at most floor((dest_r - o_r)/II) + 1
        // requests issued-but-not-done: its dest stage stalls the WHOLE
        // pipeline (freezing every issue) until its done has arrived, so
        // no iteration passes dest_r with its request outstanding, and
        // the iterations in flight between o_r and dest_r are that many.
        // Summing over the group bounds the FIFO exactly, so it can
        // never overflow and the push needs no full-check.
        unsigned tagDepth = 0;
        for (unsigned eIdx : eIdxs) {
          auto &ei = expectInfos[eIdx];
          unsigned reach = ei.destStageOffset > ei.issueStageOffset
                               ? ei.destStageOffset - ei.issueStageOffset
                               : 1;
          tagDepth += reach / std::max<uint64_t>(1, II) + 1;
        }
        auto tcTy = IntegerType::get(
            ctx, std::max(1u, llvm::Log2_64_Ceil(tagDepth + 1)));
        Value tagZero = hw::ConstantOp::create(hwBuilder, loc, tagTy, 0);
        Value tcZero = hw::ConstantOp::create(hwBuilder, loc, tcTy, 0);
        Value tcOne = hw::ConstantOp::create(hwBuilder, loc, tcTy, 1);
        Backedge tcBE = bb.get(tcTy);
        Value tcount = Value(tcBE);
        Value tagEmpty = comb::ICmpOp::create(
            hwBuilder, loc, comb::ICmpPredicate::eq, tcount, tcZero);
        Value tagNotEmpty = comb::createOrFoldNot(hwBuilder, loc, tagEmpty);
        // At most one rank pushes per cycle: two accesses to one port in
        // the same mod-II class are rejected outright below ("accessing
        // the same memory on the same cycle"), and stages fire one cycle
        // apart otherwise. The reverse walk still makes the choice
        // deterministic (lowest rank wins) rather than undefined.
        Value tagPush = falseConst;
        Value tagIn = tagZero;
        for (unsigned rank = k; rank-- > 0;) {
          Value issue =
              gatedStageCE[expectInfos[eIdxs[rank]].issueStageOffset];
          tagPush = comb::OrOp::create(hwBuilder, loc, tagPush, issue, false);
          tagIn = comb::MuxOp::create(
              hwBuilder, loc, issue,
              hw::ConstantOp::create(hwBuilder, loc, tagTy, rank), tagIn);
        }
        Value tagPop =
            comb::AndOp::create(hwBuilder, loc, done, tagNotEmpty, false);
        Value tcInc =
            comb::MuxOp::create(hwBuilder, loc, tagPush, tcOne, tcZero);
        Value tcDec =
            comb::MuxOp::create(hwBuilder, loc, tagPop, tcOne, tcZero);
        Value tcHeld = comb::SubOp::create(
            hwBuilder, loc,
            comb::AddOp::create(hwBuilder, loc, tcount, tcInc, false), tcDec,
            false);
        Value tcNext =
            comb::MuxOp::create(hwBuilder, loc, startSignal, tcZero, tcHeld);
        Value tcReg = seq::CompRegOp::create(
            hwBuilder, loc, tcNext, clk, rst, tcZero,
            hwBuilder.getStringAttr((namePrefix + "_done_tag_count_" +
                                     std::to_string(eIdxs.front()))
                                        .str()));
        tcBE.setValue(tcReg);
        // Shift-register FIFO, head at entry[0] — the same shape as the
        // per-expect data FIFO below. A pop consumes the head first, so a
        // simultaneous push lands at count-1; otherwise at count.
        SmallVector<Value> tagEntries;
        SmallVector<Backedge> tagBEs;
        for (unsigned i = 0; i < tagDepth; ++i) {
          tagBEs.push_back(bb.get(tagTy));
          tagEntries.push_back(seq::CompRegOp::create(
              hwBuilder, loc, Value(tagBEs.back()), clk, rst, tagZero,
              hwBuilder.getStringAttr(
                  (namePrefix + "_done_tag_" + std::to_string(eIdxs.front()) +
                   "_" + std::to_string(i))
                      .str())));
        }
        Value tagWrIdx = comb::MuxOp::create(
            hwBuilder, loc, tagPop,
            comb::SubOp::create(hwBuilder, loc, tcReg, tcOne, false), tcReg);
        for (unsigned i = 0; i < tagDepth; ++i) {
          Value upper = (i + 1 < tagDepth) ? tagEntries[i + 1] : tagZero;
          Value shifted =
              comb::MuxOp::create(hwBuilder, loc, tagPop, upper, tagEntries[i]);
          Value wsel = comb::ICmpOp::create(
              hwBuilder, loc, comb::ICmpPredicate::eq, tagWrIdx,
              hw::ConstantOp::create(hwBuilder, loc, tcTy, i));
          Value writeI =
              comb::AndOp::create(hwBuilder, loc, tagPush, wsel, false);
          tagBEs[i].setValue(
              comb::MuxOp::create(hwBuilder, loc, writeI, tagIn, shifted));
        }
        // An empty FIFO means the done answers a request this group never
        // issued (a posted access with no expect on the same done
        // stream). Give it to nobody: the waiting expect then stalls,
        // which is a hang rather than a silent mis-attribution.
        for (unsigned rank = 0; rank < k; ++rank) {
          Value isMine = comb::ICmpOp::create(
              hwBuilder, loc, comb::ICmpPredicate::eq, tagEntries[0],
              hw::ConstantOp::create(hwBuilder, loc, tagTy, rank));
          doneFor[rank] = comb::AndOp::create(
              hwBuilder, loc, done,
              comb::AndOp::create(hwBuilder, loc, tagNotEmpty, isMine, false),
              false);
        }
      }
    }
    for (auto [rank, eIdx] : llvm::enumerate(eIdxs)) {
      auto &ei = expectInfos[eIdx];
      Value doneE = doneFor[rank];
      // The pipeline holds at most dest-issue iterations between issue
      // and consume, bounding both the counter and the data FIFO.
      unsigned depth = ei.destStageOffset > ei.issueStageOffset
                           ? ei.destStageOffset - ei.issueStageOffset
                           : 1;
      auto cntTy =
          IntegerType::get(ctx, std::max(1u, llvm::Log2_64_Ceil(depth + 1)));
      Value cntZero = hw::ConstantOp::create(hwBuilder, loc, cntTy, 0);
      Value cntOne = hw::ConstantOp::create(hwBuilder, loc, cntTy, 1);
      Value cntMax = hw::ConstantOp::create(hwBuilder, loc, cntTy, depth);
      Backedge cntBE = bb.get(cntTy);
      Value count = Value(cntBE);
      Value isEmpty = comb::ICmpOp::create(
          hwBuilder, loc, comb::ICmpPredicate::eq, count, cntZero);
      Value notEmpty = comb::createOrFoldNot(hwBuilder, loc, isEmpty);
      Value consumeEff = comb::AndOp::create(
          hwBuilder, loc, stageCE[ei.destStageOffset], notStall, false);
      // A done arriving on the consume cycle of an empty FIFO is consumed
      // live (bypass): no push, zero stall overhead for on-time dones.
      Value bypass =
          comb::AndOp::create(hwBuilder, loc, consumeEff, isEmpty, false);
      Value pop =
          comb::AndOp::create(hwBuilder, loc, consumeEff, notEmpty, false);
      Value isFull = comb::ICmpOp::create(
          hwBuilder, loc, comb::ICmpPredicate::eq, count, cntMax);
      Value push = comb::AndOp::create(
          hwBuilder, loc, doneE,
          comb::AndOp::create(hwBuilder, loc,
                              comb::createOrFoldNot(hwBuilder, loc, bypass),
                              comb::createOrFoldNot(hwBuilder, loc, isFull),
                              false),
          false);
      Value inc = comb::MuxOp::create(hwBuilder, loc, push, cntOne, cntZero);
      Value dec = comb::MuxOp::create(hwBuilder, loc, pop, cntOne, cntZero);
      Value cntHeld = comb::SubOp::create(
          hwBuilder, loc,
          comb::AddOp::create(hwBuilder, loc, count, inc, false), dec, false);
      Value cntNext =
          comb::MuxOp::create(hwBuilder, loc, startSignal, cntZero, cntHeld);
      auto cntName = hwBuilder.getStringAttr(
          (namePrefix + "_expect_count_" + std::to_string(eIdx)).str());
      Value cntReg = seq::CompRegOp::create(hwBuilder, loc, cntNext, clk, rst,
                                            cntZero, cntName);
      cntBE.setValue(cntReg);
      expectValid[eIdx] =
          comb::OrOp::create(hwBuilder, loc, doneE, notEmpty, false);

      // Load-data FIFO: push the live rdData on each done pulse, pop on
      // consume; the consumer reads the head (or the live wire through
      // the empty bypass). Shift-register FIFO, head at entry[0].
      Value liveRdData =
          ei.isLoad ? memPorts.find(portValue)->second.rdData : Value();
      if (!liveRdData)
        continue;
      Type dataTy = liveRdData.getType();
      Value zeroData = hw::ConstantOp::create(hwBuilder, loc, dataTy, 0);
      SmallVector<Value> entries;
      SmallVector<Backedge> entryBEs;
      for (unsigned i = 0; i < depth; ++i) {
        entryBEs.push_back(bb.get(dataTy));
        auto eName = hwBuilder.getStringAttr(
            (namePrefix + "_expect_fifo_" + std::to_string(eIdx) + "_" +
             std::to_string(i))
                .str());
        entries.push_back(seq::CompRegOp::create(hwBuilder, loc,
                                                 Value(entryBEs.back()), clk,
                                                 rst, zeroData, eName));
      }
      // The shift consumes the head first, so a simultaneous push lands
      // at count-1; otherwise at count.
      Value writeIndex = comb::MuxOp::create(
          hwBuilder, loc, pop,
          comb::SubOp::create(hwBuilder, loc, cntReg, cntOne, false), cntReg);
      for (unsigned i = 0; i < depth; ++i) {
        Value upper = (i + 1 < depth) ? entries[i + 1] : zeroData;
        Value shifted =
            comb::MuxOp::create(hwBuilder, loc, pop, upper, entries[i]);
        Value wsel = comb::ICmpOp::create(
            hwBuilder, loc, comb::ICmpPredicate::eq, writeIndex,
            hw::ConstantOp::create(hwBuilder, loc, cntTy, i));
        Value writeI = comb::AndOp::create(hwBuilder, loc, push, wsel, false);
        entryBEs[i].setValue(comb::MuxOp::create(hwBuilder, loc, writeI,
                                                 liveRdData, shifted));
      }
      Value outData = comb::MuxOp::create(hwBuilder, loc, isEmpty, liveRdData,
                                          entries[0]);
      dataSplices.push_back({ei.issueStageOffset, portValue, outData});
    }
  }

  // Per-stage memory port mappings. Loads/stores in stage `i` accumulate in
  // `perStagePorts[i]` so that multiple accesses to the same memref across
  // stages don't clobber each other in a single shared MemPortMapping. After
  // all stages are lowered we mux them together into `memPorts` using the
  // stage clock-enables as selectors.
  SmallVector<DenseMap<Value, MemPortMapping>> perStagePorts(stages.size());
  for (unsigned s = 0; s < stages.size(); ++s)
    for (auto &entry : memPorts)
      perStagePorts[s][entry.first].rdData = entry.second.rdData;
  // Route each expect's FIFO output (head / live bypass) to the load that
  // issued it: handleHWLoad maps the load's result from the issue stage's
  // per-stage rdData slot.
  for (auto &ds : dataSplices)
    if (ds.issueStage < stages.size())
      perStagePorts[ds.issueStage][ds.portValue].rdData = ds.data;

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
  // overwriting stage-J register. The chain MUST clock on the
  // stall-gated CE: the stage value registers freeze during a stall, and
  // a raw-CE chain would keep shifting, sliding values across iterations
  // (each consumer would read a LATER iteration's value — caught by the
  // axi_copy end-to-end test as dst[i] = src[i+1]).
  CrossStageValueResolver delayResolver(
      hwBuilder, loc, hwBody, clk, rst, stages, gatedStageCE, mapping,
      operatorLibrary, namePrefix.str());

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
      // when effects reach memory). `rawGate` mirrors `gate` but is built
      // from the UN-stall-gated stageCE — stall terms derived from it
      // cannot form a combinational loop through `notStall`.
      std::function<LogicalResult(Operation *, Value, Value)> processOp =
          [&](Operation *inner, Value gate, Value rawGate) -> LogicalResult {
        if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp>(inner))
          return success();
        if (auto ifOp = dyn_cast<LoopScheduleIfOp>(inner)) {
          Value cond = mapping.lookup(ifOp.getCond());
          Value innerGate =
              comb::AndOp::create(hwBuilder, loc, gate, cond);
          Value innerRaw =
              comb::AndOp::create(hwBuilder, loc, rawGate, cond);
          for (auto &nested : ifOp.getBody().front()) {
            if (auto yieldOp = dyn_cast<LoopScheduleYieldOp>(&nested)) {
              for (auto [res, val] :
                   llvm::zip(ifOp.getResults(), yieldOp.getOperands()))
                mapping.map(res, mapping.lookup(val));
              continue;
            }
            if (failed(processOp(&nested, innerGate, innerRaw)))
              return failure();
          }
          return success();
        }
        if (auto storeOp = dyn_cast<HWStoreLoweringInterface>(inner)) {
          // Dynamic store issue is a single-cycle pulse on a cycle the
          // memory will accept it: gate wr_en with the port's acceptance
          // `ready` and stall the pipeline while the memory won't take the
          // request. This covers posted (no_wait) and blocking stores alike
          // — a blocking store's completion is tracked by its expect's done
          // counter at the dest stage, not by holding wr_en.
          Value storeGate = gate;
          if (auto si = dyn_cast<loopschedule::StoreInterface>(inner)) {
            if (si.isDynamic()) {
              auto pIt = memPorts.find(storeOp.getMemoryValue());
              if (pIt != memPorts.end() && pIt->second.ready) {
                Value ready = pIt->second.ready;
                storeGate =
                    comb::AndOp::create(hwBuilder, loc, storeGate, ready);
                postedReadyStalls.push_back({rawGate, ready});
              }
            }
          }
          return handleHWStore(storeOp, hwBuilder, mapping, storeGate,
                                  perStagePorts[stageIdx]);
        }
        if (auto loadOp = dyn_cast<HWLoadLoweringInterface>(inner)) {
          // Every memref-backed load is a latency-1 registered read:
          // local hlmems by construction, and external ports by the
          // BRAM-port contract (the external memory's output register
          // provides the 1-cycle delay, so no stage register is added).
          if (auto ls = dyn_cast<LoopScheduleLoadOp>(inner))
            localLoadResults.insert(ls.getResult());
          if (loadOp.requiresReadEnable() && loadOp.getReadLatency() > 0)
            localLoadResults.insert(loadOp.getResult());
          // Dynamic load issue is a single-cycle pulse like stores. Use the
          // stall-gated `gate`, NOT the raw stageCE: a request held across
          // stall cycles re-issues on a queue-style engine — an AXI master
          // accepts a NEW read every cycle rd_en stays high (duplicate
          // ARs, caught by the axi_copy end-to-end test).
          Value loadGate = gate;
          // Acceptance backpressure for pipelined dynamic loads: with the
          // expect at +latency, several requests are in flight at once, so
          // a full read queue (AXI) or a busy arbiter lane (on-chip) must
          // stall issue rather than drop the request.
          {
            auto pIt = memPorts.find(loadOp.getMemoryValue());
            if (pIt != memPorts.end() && pIt->second.ready) {
              Value ready = pIt->second.ready;
              loadGate =
                  comb::AndOp::create(hwBuilder, loc, loadGate, ready);
              postedReadyStalls.push_back({rawGate, ready});
            }
          }
          return handleHWLoad(loadOp, hwBuilder, mapping,
                                 perStagePorts[stageIdx], loadGate);
        }
        return emitComputeOp(
            inner, hwBuilder, mapping,
            hwBody->getParentOp()->getParentOfType<ModuleOp>(), clk, rst,
            /*opCE=*/notStall, /*shareGate=*/gate);
      };
      // Use the stall-gated stage CE as the write-/read-enable gate so
      // memory requests don't re-fire when the pipeline idles on !done.
      // The raw stageCE rides along for loop-free stall terms.
      opResult = processOp(&op, gatedStageCE[stageIdx], stageCE[stageIdx]);

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
      if (computeValueCycleLatency(val, operatorLibrary) > 0) {
        mapping.map(stageOp.getResult(regIdx), mappedVal);
        continue;
      }

      Value resetVal = createZeroConstant(hwBuilder, loc, mappedVal.getType());
      auto regName = hwBuilder.getStringAttr(
          (namePrefix + "_s" + std::to_string(stageIdx) + "_r" +
           std::to_string(regIdx))
              .str());

      // Stall-gated CE so iter_args / inter-stage values don't advance
      // during a multi-bank arbiter stall. The arbiter's address latch
      // (in `--amc-to-hw`'s arbiter lowering) keeps the in-flight
      // requests pointing at iter K's addresses across the stall;
      // gating the iter_arg register here keeps the live-address path
      // pointing at the same iter so a fresh issue at consume cycle
      // sees iter K+1's addresses instead of iter K+S.
      Value reg = seq::CompRegClockEnabledOp::create(
          hwBuilder, loc, mappedVal, clk, gatedStageCE[stageIdx], rst,
          resetVal, regName);
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

  // Iter-arg init delivery. Historically each iter_arg's value was a
  // combinational mux `first_iter_s<fs> ? init : feedback` driving the
  // backedge, so a 1-bit per-stage first_iter flop fanned out across
  // every consumer bit of every iter_arg (bitwidth x iter_args loads —
  // a fanout-97 net on doitgen's accumulator) and the mux logic sat in
  // front of the accumulator's carry chain on the critical path.
  //
  // Instead, PRELOAD the feedback register with the init value on a
  // delayed copy of the start pulse and resolve the backedge to the
  // register output directly: no datapath mux, no first_iter fanout.
  //
  // Preload phasing: for a start pulse in cycle T, stage u first fires
  // (consumers sample, captures land) during cycle T+1+u — `active`
  // rises at T+1 and stageCE[u] is activeCE delayed u cycles, all on
  // notStall-gated flops that freeze in lockstep with the preload delay
  // chain during stalls. Preloading on start delayed by d lands the
  // init at the END of cycle T+d, register-readable from T+1+d — one
  // full cycle before the earliest consumer fire at T+1+u, u >= d. The
  // depth d = firstIterArgUseOffset matches the early child_start
  // peephole's init-timing guard exactly: the peephole only guarantees
  // a same-frame-produced init is readable at cycle start + firstUse,
  // so sampling any earlier would latch garbage on peephole-started
  // pipelines (the doitgen accumulator case).
  //
  // Forced/fallback cases:
  //  * an iter_arg feeding the loop condition combinationally must
  //    present init from T+1 (the condition gates activeCE from the
  //    first post-start cycle, reading the backedge unregistered), so
  //    its depth is forced to 0 — the same timing the old mux gave,
  //    whose first_iter re-armed at the end of the start cycle;
  //  * when firstUse > feedbackStage (preload could sample an init the
  //    peephole never proved ready) or the feedback is not a plain
  //    stage register (local hlmem reads, latent operator outputs,
  //    passthroughs), fall back to the historical first_iter mux,
  //    built lazily below.
  //
  // The preload is edge-triggered by the start pulse rather than
  // derived from CE history, which structurally removes the run-N>0
  // stall hazard the old first_iter clear had (the clear had to use the
  // stall-gated CE or run N>0 of stall-heavy AXI loops read the
  // previous run's final accumulator instead of init).
  SmallVector<Value> firstIterPerStage(stages.size());
  auto getFirstIter = [&](unsigned s) -> Value {
    if (firstIterPerStage[s])
      return firstIterPerStage[s];
    // 1 on reset, re-arms to 1 on start, clears the first time
    // gatedStageCE[s] genuinely fires (the raw stageCE only means
    // "stage occupied" and freezes high across stalls).
    Backedge be = bb.get(hwBuilder.getI1Type());
    Value notCE = comb::createOrFoldNot(hwBuilder, loc, gatedStageCE[s]);
    Value sticky = comb::AndOp::create(hwBuilder, loc, Value(be), notCE);
    Value nxt = comb::OrOp::create(hwBuilder, loc, startSignal, sticky);
    auto reg = seq::CompRegOp::create(
        hwBuilder, loc, nxt, clk, rst, trueConst,
        hwBuilder.getStringAttr(
            (namePrefix + "_first_iter_s" + std::to_string(s)).str()));
    be.setValue(reg);
    firstIterPerStage[s] = reg;
    return reg;
  };

  // Lazily grown chain of notStall-gated start-pulse delays. Assumes the
  // 1-cycle start pulse contract the old first_iter re-arm also relied
  // on.
  SmallVector<Value> startDelayChain = {startSignal};
  auto getStartDelayed = [&](unsigned d) -> Value {
    while (startDelayChain.size() <= d)
      startDelayChain.push_back(seq::CompRegClockEnabledOp::create(
          hwBuilder, loc, startDelayChain.back(), clk, notStall, rst,
          falseConst,
          hwBuilder.getStringAttr((namePrefix + "_preload_start_d" +
                                   std::to_string(startDelayChain.size()))
                                      .str())));
    return startDelayChain[d];
  };

  // Conservative backward slice of the condition's combinational cone
  // over the source IR: stage-result crossings are registered and stop
  // the walk; every stages-block argument reached is an iter_arg the
  // condition reads combinationally at T+1.
  DenseSet<unsigned> condConeArgs;
  {
    SmallVector<Value> worklist;
    if (auto condResult = dyn_cast<OpResult>(condTermVal);
        condResult && isa<LoopScheduleAtOp>(condResult.getOwner()))
      worklist.push_back(stages[condStageIdx].getYieldOp().getOperands()
                             [condResult.getResultNumber()]);
    else
      worklist.push_back(condTermVal);
    DenseSet<Value> seen;
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!v || !seen.insert(v).second)
        continue;
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        if (barg.getOwner() == &pipOp.getStagesBlock())
          condConeArgs.insert(barg.getArgNumber());
        continue;
      }
      Operation *def = v.getDefiningOp();
      if (!def || isa<LoopScheduleAtOp>(def) || !pipOp->isAncestor(def))
        continue; // registered stage result / defined outside: stop.
      for (Value operand : def->getOperands())
        worklist.push_back(operand);
    }
  }

  auto pipIterArgUpdates = loopschedule::getIterArgUpdatesInOrder(pipOp);
  // Registers already preloaded: init value + pulse depth, so a second
  // iter_arg resolving to the same register can reuse the preload only
  // when it is provably compatible.
  DenseMap<Operation *, std::pair<Value, unsigned>> preloadedRegs;
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value init = mapping.lookup(pipOp.getInits()[i]);
    Value termVal = pipIterArgUpdates[i]
                        ? loopschedule::getIterArgPhaseResult(pipIterArgUpdates[i])
                        : Value{};
    Value feedback = mapping.lookup(termVal);
    // The feedback value is produced at some stage; the init must stay
    // visible until that stage has fired at least once. If the feedback
    // isn't a stage result (shouldn't happen for well-formed pipelines),
    // fall back to stage 0.
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

    unsigned firstUse = firstIterArgUseOffset(pipOp, i);
    unsigned d = condConeArgs.contains(i) ? 0u
                 : firstUse == UINT_MAX   ? feedbackStage
                                          : firstUse;

    auto feedbackReg =
        feedback ? feedback.getDefiningOp<seq::CompRegClockEnabledOp>()
                 : seq::CompRegClockEnabledOp();
    bool canPreload =
        feedbackReg && d <= feedbackStage && init != Value(iterArgBackedges[i]);
    if (canPreload) {
      if (auto it = preloadedRegs.find(feedbackReg);
          it != preloadedRegs.end())
        canPreload = it->second.first == init && it->second.second <= d;
      else
        canPreload =
            feedbackReg.getClockEnable() == gatedStageCE[feedbackStage];
    }

    if (canPreload) {
      if (!preloadedRegs.count(feedbackReg)) {
        Value pulse = getStartDelayed(d);
        Value newD = comb::MuxOp::create(hwBuilder, loc, pulse, init,
                                         feedbackReg.getInput());
        Value newCE = comb::OrOp::create(
            hwBuilder, loc, feedbackReg.getClockEnable(), pulse);
        feedbackReg.getInputMutable().assign(newD);
        feedbackReg.getClockEnableMutable().assign(newCE);
        preloadedRegs[feedbackReg] = {init, d};
      }
      iterArgBackedges[i].setValue(feedbackReg);
      continue;
    }
    Value muxed = comb::MuxOp::create(
        hwBuilder, loc, getFirstIter(feedbackStage), init, feedback);
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

  // The epilogue chain shifts in LOCKSTEP with the stage-CE chain
  // (CE = notStall): a stalled pipeline freezes its in-flight iterations,
  // and an epilogue that kept shifting would race ahead of them and fire
  // `done` while a mid-stage iteration is still frozen (the tail-CE guard
  // below only covers the LAST stage). II=1 pipelines masked this — dense
  // occupancy keeps the tail stage busy through the drain — but sparse
  // II>1 schedules with a mid-stage ready/expect stall drop accesses of
  // still-in-flight iterations when the enclosing frame consumes the
  // early done.
  Value notStart = comb::createOrFoldNot(hwBuilder, loc, startSignal);
  Backedge epilogueBE = bb.get(hwBuilder.getI1Type());
  Value epilogueHold =
      comb::AndOp::create(hwBuilder, loc, Value(epilogueBE), notStart);
  Value epilogueNext =
      comb::OrOp::create(hwBuilder, loc, epilogueTrigger, epilogueHold);
  auto epilogueReg = seq::CompRegClockEnabledOp::create(
      hwBuilder, loc, epilogueNext, clk, notStall, rst, falseConst,
      hwBuilder.getStringAttr(namePrefix + "_epilogue"));
  epilogueBE.setValue(epilogueReg);

  // Depth of the epilogue: prefer the schedule's declared iteration latency
  // (which covers result-less tails like static store commits even if no
  // stage spans them); fall back to the stage count for IR without the
  // attribute. Take the max so extra stages (e.g. expect destinations
  // appended after scheduling) still drain fully.
  unsigned pipelineDepth = stages.size();
  if (auto latency = pipOp.getLatency())
    pipelineDepth = std::max<unsigned>(pipelineDepth, *latency);

  Value delayedEpilogue = Value(epilogueReg);
  for (unsigned i = 0; i + 2 < pipelineDepth; ++i) {
    Value delayInput =
        comb::MuxOp::create(hwBuilder, loc, startSignal, falseConst,
                            delayedEpilogue);
    delayedEpilogue = seq::CompRegClockEnabledOp::create(
        hwBuilder, loc, delayInput, clk, notStall, rst, falseConst,
        hwBuilder.getStringAttr(
            (namePrefix + "_epilogue_delay_" + std::to_string(i)).str()));
  }

  Value notTailCE = comb::createOrFoldNot(hwBuilder, loc, stageCE.back());
  Value doneComb =
      comb::AndOp::create(hwBuilder, loc, delayedEpilogue, notTailCE);

  // A static-port store commits its write `latency` cycles after issue, but
  // the epilogue chain above only counts stages — stages exist at op START
  // times, so a tail store's commit window extends past the last stage and
  // `done` would fire before the write lands (inter-loop RAW hazard: the
  // next loop reads stale data). Count the worst store-commit overhang and
  // hold `done` that many extra cycles. The count starts from `doneComb`
  // (actual tail-stage drain, II-robust) rather than from the epilogue
  // chain. Dynamic-port stores are excluded: no static count is sufficient
  // for them (they are handled by launch/expect wrapping or, eventually,
  // explicit fences).
  unsigned pipeEnd =
      std::max(pipelineDepth, (unsigned)stages.back().getOffset() + 1);
  unsigned maxStoreEnd = 0;
  for (auto stageOp : stages)
    stageOp.walk([&](Operation *op) {
      if (auto st = dyn_cast<StoreInterface>(op))
        if (!st.isDynamic())
          maxStoreEnd = std::max(
              maxStoreEnd, (unsigned)stageOp.getOffset() + st.getLatency());
    });
  unsigned storeTailCycles = maxStoreEnd > pipeEnd ? maxStoreEnd - pipeEnd : 0;
  for (unsigned i = 0; i < storeTailCycles; ++i) {
    Value tailInput = comb::MuxOp::create(hwBuilder, loc, startSignal,
                                          falseConst, doneComb);
    doneComb = seq::CompRegClockEnabledOp::create(
        hwBuilder, loc, tailInput, clk, notStall, rst, falseConst,
        hwBuilder.getStringAttr(
            (namePrefix + "_store_tail_" + std::to_string(i)).str()));
  }

  // Done cuts through combinationally. doneComb is already exact (the
  // tail stage has drained) and level-held: the epilogue chain holds
  // until the next start clears it at the start edge, so like the old
  // registered done it reads 0 from one cycle after child_start — the
  // first cycle any WAIT state can sample it. Registering it here only
  // cost a flat +1 cycle on every launch.
  doneSignal = doneComb;

  // Phase 3B: resolve the stall backedge. Stall while an expect's dest
  // stage wants to consume but its done counter is empty and no done is
  // arriving this cycle (`expectValid = doneE | count != 0` is
  // combinational, so on-time dones pass with zero overhead). The terms
  // are built from registered signals only (stageCE, memory done, the
  // counter), so no combinational loop through `notStall`.
  SmallVector<Value> stallBits;
  for (auto [eIdx, ei] : llvm::enumerate(expectInfos)) {
    if (!expectValid[eIdx])
      continue;
    Value notValid =
        comb::createOrFoldNot(hwBuilder, loc, expectValid[eIdx]);
    Value stallI = comb::AndOp::create(hwBuilder, loc,
                                         stageCE[ei.destStageOffset],
                                         notValid);
    stallBits.push_back(stallI);
  }
  // Posted-issue backpressure: hold the pipeline while a posted (no_wait)
  // access wants to fire into a not-ready memory. The issue gate is the
  // raw (un-stall-gated) stageCE & predication, so the term cannot loop
  // through `notStall`; `ready` is memory/adapter state with no dependence
  // on the pipeline's stall.
  for (auto &[rawGate, ready] : postedReadyStalls) {
    Value notReady = comb::createOrFoldNot(hwBuilder, loc, ready);
    stallBits.push_back(
        comb::AndOp::create(hwBuilder, loc, rawGate, notReady));
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
    } else if (!hw::isHWValueType(arg.getType())) {
      // An external-memory reference (e.g. !amc.memory_ref) has no scalar HW
      // signal: its interface (BRAM/AXI) ports are appended to this module
      // after its expand_ref adapter is lowered (see bramBoundaries).
      continue;
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

  // The loopschedule func ops carry the block-control protocol as an
  // inherent attribute (their `control = <kind>` clause); republish it as a
  // discardable `amc.control_interface` marker on the hw.module for the
  // control-slave wrapper pass and the testbench generator.
  {
    StringAttr ctrl;
    Operation *funcOperation = funcOp.getOperation();
    if (auto seq = dyn_cast<loopschedule::LoopScheduleFuncSequentialOp>(
            funcOperation))
      ctrl = seq.getControlInterfaceAttr();
    else if (auto pipe = dyn_cast<loopschedule::LoopScheduleFuncPipelineOp>(
                 funcOperation))
      ctrl = pipe.getControlInterfaceAttr();
    if (ctrl)
      hwMod->setAttr("amc.control_interface", ctrl);
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
    } else if (!hw::isHWValueType(arg.getType())) {
      // External-memory reference arg: elided from the signature above, so it
      // consumes no block argument here.
      continue;
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

/// The name the boundary is ordered by: its first port. Port names are unique
/// within a module, so this is a total order over the boundaries of one kernel.
static StringRef bramBoundaryOrderKey(const loopschedule::BramBoundary &b) {
  if (!b.inputPorts.empty())
    return b.inputPorts.front().getName();
  if (!b.outputPorts.empty())
    return b.outputPorts.front().getName();
  return StringRef();
}

/// Splice the external-memory (BRAM/AXI) boundary ports an `expand_ref` adapter
/// registered onto the kernel `hw.module`: append the interface INPUT ports
/// (resolving each dout backedge to its new block arg) and the interface OUTPUT
/// ports (driven by the adapter's results). Runs after `buildHWOutput` so the
/// appended outputs extend the just-built `hw.output`.
///
/// `bramBoundaries` is keyed by SSA value, so iterating it directly would emit
/// the kernel's boundary ports (and the `amc.axi_bundles` register map derived
/// from them) in POINTER-HASH order — different from run to run for the same
/// input, which is how a pure refactor ended up wobbling downstream synthesis
/// results by ~1%. Order the boundaries by their first port name instead, with
/// a numeric-aware compare so `mem2` precedes `mem10`. Every producer in tree
/// names a boundary after the function argument it came from (`mem<argNo>` for
/// a BRAM arg, `gmem<k>`-style bundles assigned in argument order for m_axi),
/// so this IS argument order, and the per-bundle `<b>_base_addr` boundary sorts
/// just ahead of its own `<b>_m_axi_*` face.
static void appendBramBoundaries(
    hw::HWModuleOp hwMod,
    DenseMap<Value, loopschedule::BramBoundary> &bramBoundaries) {
  Block *body = hwMod.getBodyBlock();
  SmallVector<Attribute> axiBundles;
  SmallVector<loopschedule::BramBoundary *> ordered;
  for (auto &kv : bramBoundaries)
    ordered.push_back(&kv.second);
  llvm::sort(ordered, [](const loopschedule::BramBoundary *a,
                         const loopschedule::BramBoundary *b) {
    return bramBoundaryOrderKey(*a).compare_numeric(bramBoundaryOrderKey(*b)) <
           0;
  });
  for (auto *bp : ordered) {
    auto &b = *bp;
    if (!b.inputPorts.empty()) {
      unsigned base = hwMod.getModuleType().getNumInputs();
      // modifyPorts updates the module TYPE but not the body block args (and it
      // internally re-locs every input arg), so add the matching block args
      // FIRST — at the end, where we insert the ports — to keep type and body
      // consistent, then update the type.
      SmallVector<std::pair<unsigned, hw::PortInfo>> ins;
      for (auto [k, p] : llvm::enumerate(b.inputPorts)) {
        auto arg = body->addArgument(p.type, hwMod.getLoc());
        b.inputBackedges[k].setValue(arg);
        ins.push_back({base, p});
      }
      hwMod.modifyPorts(ins, {}, {}, {});
    }
    for (auto [p, v] : llvm::zip(b.outputPorts, b.outputValues))
      hwMod.appendOutput(p.name, v);
    if (b.axiMeta) {
      auto *ctx = hwMod.getContext();
      OpBuilder ab(ctx);
      SmallVector<int64_t> argElems(b.axiMeta->argElems.begin(),
                                    b.axiMeta->argElems.end());
      axiBundles.push_back(ab.getDictionaryAttr(
          {ab.getNamedAttr("name", ab.getStringAttr(b.axiMeta->bundle)),
           ab.getNamedAttr("depth", ab.getI64IntegerAttr(b.axiMeta->depth)),
           ab.getNamedAttr("elem_width",
                           ab.getI64IntegerAttr(b.axiMeta->elemWidth)),
           // One entry per ARGUMENT on the bundle, in base-register order:
           // each has its own runtime base, so a testbench places them
           // independently instead of assuming one packed allocation.
           ab.getNamedAttr("arg_elems", ab.getDenseI64ArrayAttr(argElems))}));
    }
  }
  // Publish AXI bundle metadata for the testbench generator (slave depth /
  // element width are elaboration constants it cannot recover from the port
  // face).
  if (!axiBundles.empty())
    hwMod->setAttr("amc.axi_bundles",
                   ArrayAttr::get(hwMod.getContext(), axiBundles));
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

  // The registered ports in a DEFINED order: instance ops in program order,
  // each op's results in result order. `portMap` is keyed by SSA value, so
  // walking it directly would order `memrefArgs` — and with it the FSM's
  // per-port naming and the order the per-step access drives are merged in —
  // by pointer hash, i.e. differently from run to run for the same input.
  // Every registered value is a result of the instance op that registered it,
  // so this traversal is exhaustive.
  SmallVector<Value> orderedPorts;
  for (auto instOp : instanceOps)
    for (Value res : instOp->getResults())
      if (memInstState.portMap.count(res))
        orderedPorts.push_back(res);
  assert(orderedPorts.size() == memInstState.portMap.size() &&
         "a registered port is not a result of any lowered instance op");

  for (Value portValue : orderedPorts) {
    const auto &signals = *memInstState.lookupPort(portValue);
    MemPortMapping mp;
    mp.rdData = signals.rdData;
    mp.addrs.assign(signals.addrs.size(), Value());
    mp.done = signals.done;
    mp.wrDone = signals.wrDone;
    mp.ready = signals.ready;
    mp.rdIdle = signals.rdIdle;
    mp.wrIdle = signals.wrIdle;
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
  for (Value portValue : orderedPorts) {
    const auto &signals = *memInstState.lookupPort(portValue);
    // A control channel registers `done` only (no addresses, no data, no
    // enables): nothing to drive or merge per step, so it must not become
    // a PortArgInfo — the FSM reads its done straight from memPortMap
    // (barrier entries, kind 3).
    if (signals.addrs.empty() && !signals.rdData && !signals.wrData &&
        !signals.wrEn)
      continue;
    PortArgInfo info;
    info.originalArg = portValue;
    info.isAmcPort = true;
    info.latency = signals.latency;
    info.addrWidths.reserve(signals.addrs.size());
    for (Value a : signals.addrs)
      info.addrWidths.push_back(cast<IntegerType>(a.getType()).getWidth());
    info.isRead = signals.rdData != Value();
    info.isWrite = signals.wrEn != Value();
    info.hasReady = signals.ready != Value();
    info.hasDone = signals.done != Value();
    info.hasWrDone = signals.wrDone != Value();
    info.hasRdIdle = signals.rdIdle != Value();
    info.hasWrIdle = signals.wrIdle != Value();
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
  sharedOperators.clear();

  // Resolve the enclosing builtin.module so the operator-library lowering
  // can look up `hw.module.extern` declarations by symbol.
  auto enclosingModule = funcOp->getParentOfType<ModuleOp>();

  builder.setInsertionPointToEnd(hwBody);

  auto i1 = builder.getI1Type();
  BackedgeBuilder funcBB(builder, loc);
  SmallVector<HLMemBackedges> hlmemBEs;
  SymbolTable modSymTab(enclosingModule);
  DenseMap<Value, loopschedule::BramBoundary> bramBoundaries;
  loopschedule::HWMemoryLoweringState memInstState(clk, rst, funcBB, modSymTab,
                                                    bramBoundaries);
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
    // -1 leaf, 0 sequential, 1 pipeline, 2 call, 3 barrier, 4 dynamic
    // port access (launch-wrapped m_axi load/store in a function frame)
    int kind;
    LoopScheduleSequentialOp seqOp;
    LoopSchedulePipelineOp pipOp;
    LoopScheduleCallOp callOp;
    loopschedule::HWStoreLoweringInterface barrierOp; // kind 3
    Operation *dynOp = nullptr;                       // kind 4
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
      } else if (auto barrier =
                     dyn_cast_or_null<loopschedule::HWStoreLoweringInterface>(
                         child);
                 barrier && isa<loopschedule::StoreInterface>(child) &&
                 cast<loopschedule::StoreInterface>(child).isBarrier()) {
        // A barrier store (amc.control): no child hardware; its "done" is
        // the memory control channel's completion level.
        fc.kind = 3;
        fc.barrierOp = barrier;
        entries.push_back(fc);
      } else if (child && isSeqDynAccess(child)) {
        // A launch-wrapped dynamic port access (m_axi load/store in a
        // function frame, wrapped by the scheduler's func strategy): no
        // child hardware; the entry drives the port with the issue
        // handshake and its WAIT state holds until the completion pulse.
        fc.kind = 4;
        fc.dynOp = child;
        entries.push_back(fc);
      }
    }
  }

  // Build a per-entry kind vector for createFunctionFSM. The FSM sees one
  // state per entry, not per step. Leaf entries whose frame contains no
  // ops at all (await-only frames, e.g. the trailing `frame await` that
  // orders the function terminator after the last launch) become
  // STATELESS (-2): the preceding WAIT already enforced the ordering, so
  // the state would burn a cycle doing nothing. Their awaits still
  // resolve in the entry-processing loop below.
  SmallVector<int> entryKinds;
  for (auto &e : entries) {
    int kind = e.kind;
    if (kind == -1 &&
        topFrames[e.frameIdx].getBodyBlock().getOps<LoopScheduleAtOp>()
            .empty())
      kind = -2;
    entryKinds.push_back(kind);
  }

  // Leaf entries with multi-cycle frames (e.g. two same-port stores the
  // scheduler serialized to `at 0` / `at 1`) hold their FSM state for the
  // frame's latency and get per-cycle gates.
  SmallVector<unsigned> entryLatencies;
  for (auto &e : entries)
    entryLatencies.push_back(
        e.kind < 0 ? computeFrameLatency(topFrames[e.frameIdx]) : 1u);

  // Function-level port contention (same rationale as the sequential-loop
  // body's bodyMultiAccess): count static accesses across ALL top frames so
  // same-port accesses in different frames/cycles get gate-muxed drives.
  llvm::DenseSet<std::pair<Value, unsigned>> funcMultiAccess;
  {
    llvm::DenseMap<std::pair<Value, unsigned>, unsigned> counts;
    // Walk EVERY region of each frame: await-frames keep their post-await
    // work in the `do` region, which getBodyBlock() (the await region)
    // doesn't cover.
    for (auto frameOp : topFrames)
      for (Region &region : frameOp->getRegions())
        for (Block &block : region)
          collectFrameAccessCounts(block, counts);
    for (auto &entry : counts)
      if (entry.second > 1)
        funcMultiAccess.insert(entry.first);
  }

  // Count non-leaf entries.
  unsigned numChildren = 0;
  SmallVector<int> childIndexForEntry(entries.size(), -1);
  for (unsigned i = 0; i < entries.size(); ++i) {
    if (entries[i].kind >= 0) {
      childIndexForEntry[i] = numChildren;
      numChildren++;
    }
  }

  // Per-entry group ids: entries of one frame are CONCURRENT siblings and
  // share one FSM stage (start together, wait together) — EXCEPT calls
  // and barriers. `construct-memory-dependencies` does not model a
  // call's memory effects (nor a control barrier's ordering), so the
  // scheduler may co-frame a call with launches it actually depends on;
  // the historical linear FSM masked that by executing entries in order.
  // Keep that contract: a call/barrier entry is always its own group,
  // sequenced against its frame siblings.
  SmallVector<unsigned> entryGroups;
  {
    unsigned g = 0;
    auto isolated = [&](unsigned i) {
      return entries[i].kind == 2 || entries[i].kind == 3;
    };
    for (unsigned i = 0; i < entries.size(); ++i) {
      if (i > 0 && (entries[i].frameIdx != entries[i - 1].frameIdx ||
                    isolated(i) || isolated(i - 1)))
        ++g;
      entryGroups.push_back(g);
    }
  }

  // Create function-level FSM.
  std::string funcName = funcOp.getName().str();
  std::string fsmName = funcName + "_fsm";
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  SmallVector<int> entryCycleOutBase;
  createFunctionFSM(builder, loc, fsmName, entryKinds, entryLatencies,
                    entryGroups, entryCycleOutBase);
  unsigned totalEntryCycleOuts = 0;
  for (unsigned i = 0; i < entries.size(); ++i)
    if (entryCycleOutBase[i] >= 0)
      totalEntryCycleOuts += entryLatencies[i];

  // Instantiate function FSM in HW module body.
  builder.setInsertionPointToEnd(hwBody);
  BackedgeBuilder bb(builder, loc);

  // Per-child done backedges (resolved by each child's lowering below).
  SmallVector<Backedge> childDoneBEs;
  for (unsigned i = 0; i < numChildren; ++i)
    childDoneBEs.push_back(bb.get(i1));

  // FSM inputs: start + one combined done per child-launching group. A
  // singleton group feeds its child's done straight through (identical to
  // the historical per-entry FSM). A multi-child group ANDs its members'
  // dones, with a sticky `seen` latch per member so dones that pulse at
  // different times still conjoin; the latches clear on the group's
  // child_start pulse (an FSM output — wired through a backedge resolved
  // after instantiation).
  SmallVector<Value> fsmInputs;
  fsmInputs.push_back(start);
  struct GroupDoneWiring {
    Backedge startPulse;         // resolved to the group's child_start
    unsigned firstChildIdx;      // whose start signal to use
  };
  SmallVector<GroupDoneWiring> groupDoneWirings;
  {
    Value falseC = hw::ConstantOp::create(builder, loc, i1, 0);
    for (unsigned i = 0; i < entries.size();) {
      unsigned groupId = entryGroups[i];
      unsigned frameIdx = entries[i].frameIdx;
      SmallVector<unsigned> cis;
      unsigned j = i;
      for (; j < entries.size() && entryGroups[j] == groupId; ++j)
        if (childIndexForEntry[j] >= 0)
          cis.push_back((unsigned)childIndexForEntry[j]);
      if (cis.size() == 1) {
        fsmInputs.push_back(Value(childDoneBEs[cis[0]]));
      } else if (cis.size() > 1) {
        auto startBE = bb.get(i1);
        Value notStart =
            comb::createOrFoldNot(builder, loc, Value(startBE));
        Value combined;
        for (unsigned ci : cis) {
          Value done = Value(childDoneBEs[ci]);
          Backedge seenBE = bb.get(i1);
          Value seenOrDone =
              comb::OrOp::create(builder, loc, Value(seenBE), done);
          Value seenNext =
              comb::AndOp::create(builder, loc, seenOrDone, notStart);
          auto seenReg = seq::CompRegOp::create(
              builder, loc, seenNext, clk, rst, falseC,
              builder.getStringAttr("frame" + std::to_string(frameIdx) +
                                    "_done_seen_" + std::to_string(ci)));
          seenBE.setValue(seenReg);
          Value term = comb::OrOp::create(builder, loc, seenReg, done);
          combined = combined
                         ? comb::AndOp::create(builder, loc, combined, term)
                               .getResult()
                         : term;
        }
        fsmInputs.push_back(combined);
        groupDoneWirings.push_back({startBE, cis[0]});
      }
      i = j;
    }
  }

  unsigned numEntries = entries.size();
  SmallVector<Type> fsmResultTypes(
      1 + numChildren + numEntries + totalEntryCycleOuts, i1);
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

  // Resolve multi-child groups' seen-latch clear pulses: a group's
  // members all start in the same FSM state, so member 0's child_start
  // is the group's pulse.
  for (auto &w : groupDoneWirings)
    w.startPulse.setValue(childStartSignals[w.firstChildIdx]);

  // Transaction-in-flight bit (also drives `ready` at module output):
  // set on start, cleared on the FINAL done (backedge — resolved after
  // the early-done OR below).
  Backedge funcDoneBE = bb.get(i1);
  Value falseConstIF = hw::ConstantOp::create(builder, loc, i1, 0);
  Backedge inFlightNextBE = bb.get(i1);
  auto inFlightReg = seq::CompRegOp::create(
      builder, loc, Value(inFlightNextBE), clk, rst, falseConstIF,
      builder.getStringAttr("tx_in_flight"));
  Value notFuncDone = comb::createOrFoldNot(builder, loc, Value(funcDoneBE));
  Value holdInFlight =
      comb::AndOp::create(builder, loc, inFlightReg, notFuncDone);
  Value inFlightNext = comb::OrOp::create(builder, loc, start, holdInFlight);
  inFlightNextBE.setValue(inFlightNext);

  // Per-entry issue gates: multi-cycle leaf entries get their dedicated
  // frame_cycle_<i>_<c> outputs; single-cycle entries use entry_running.
  SmallVector<SmallVector<Value>> entryCycleGates(numEntries);
  for (unsigned i = 0; i < numEntries; ++i) {
    if (entryCycleOutBase[i] >= 0) {
      for (unsigned c = 0; c < entryLatencies[i]; ++c)
        entryCycleGates[i].push_back(fsmInst.getResult(
            1 + numChildren + numEntries + entryCycleOutBase[i] + c));
    } else {
      entryCycleGates[i].push_back(entryRunningSignals[i]);
    }
  }

  // Early function done: when the LAST stateful entry is a PIPELINE
  // launch and the function returns nothing, done cuts through on the
  // child's completion instead of waiting for the Moore DONE state.
  // ~child_start excludes the launch cycle, where a stale done from the
  // previous invocation may still be high. Pipelines only: their done
  // carries the epilogue/store-tail margin (it asserts at least one
  // cycle after the final store issues), whereas a sequential child's
  // advance-edge done coincides with its final store cycle — cutting the
  // MODULE-boundary done through on that edge would let the outside
  // world observe done on the same posedge the last write commits.
  {
    int lastEmitted = -1;
    for (int i = (int)numEntries - 1; i >= 0; --i) {
      if (entryKinds[i] == -2)
        continue;
      lastEmitted = i;
      break;
    }
    bool funcReturnsNothing =
        funcOp.getBody().front().getTerminator()->getNumOperands() == 0;
    // Only for a SOLO pipeline launch: with concurrent siblings in the
    // last frame, one child's done must not cut the module done early.
    bool lastIsSoloChild =
        lastEmitted >= 0 &&
        llvm::count_if(entries, [&](const FrameChild &e) {
          return e.frameIdx == entries[lastEmitted].frameIdx && e.kind >= 0;
        }) == 1;
    if (lastEmitted >= 0 && entries[lastEmitted].kind == 1 &&
        lastIsSoloChild && funcReturnsNothing) {
      unsigned ci = (unsigned)childIndexForEntry[lastEmitted];
      Value notStart =
          comb::createOrFoldNot(builder, loc, childStartSignals[ci]);
      Value inWait = comb::AndOp::create(
          builder, loc, entryRunningSignals[(unsigned)lastEmitted], notStart);
      Value early = comb::AndOp::create(builder, loc, inWait,
                                        Value(childDoneBEs[ci]));
      doneSignal = comb::OrOp::create(builder, loc, doneSignal, early);
    }
  }

  // Per-entry memory port mappings. Each starts with shared rdData and
  // (when present) the per-port `done` so an inline-lowered pipeline
  // can build its handshake registers.
  SmallVector<DenseMap<Value, MemPortMapping>> perEntryPorts(numEntries);
  for (unsigned i = 0; i < numEntries; ++i) {
    for (auto &memInfo : memrefArgs) {
      perEntryPorts[i][memInfo.originalArg].rdData =
          memPortMap[memInfo.originalArg].rdData;
      perEntryPorts[i][memInfo.originalArg].done =
          memPortMap[memInfo.originalArg].done;
      perEntryPorts[i][memInfo.originalArg].wrDone =
          memPortMap[memInfo.originalArg].wrDone;
      perEntryPorts[i][memInfo.originalArg].ready =
          memPortMap[memInfo.originalArg].ready;
      perEntryPorts[i][memInfo.originalArg].rdIdle =
          memPortMap[memInfo.originalArg].rdIdle;
      perEntryPorts[i][memInfo.originalArg].wrIdle =
          memPortMap[memInfo.originalArg].wrIdle;
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
  // Collected static access drives from `cloneFrameAtBodies`. Each entry's
  // `wrEn` (stores) / `rdEn` (loads) is already gated by the owning
  // frame's 1-cycle header pulse (see caller). After `mergeStepMemPorts`
  // runs — which unconditionally rebuilds every memory's drives from the
  // per-entry maps, clobbering anything driven into `memPortMap` directly
  // — these get priority-composed back in, so a static access coexisting
  // with a launch in the same frame takes priority during its single
  // firing cycle (and leaves the child's drives untouched otherwise).
  SmallVector<std::pair<Value, MemPortMapping>> staticDrives;
  unsigned staticFrameCounter = 0;
  unsigned staticCapCounter = 0;

  auto cloneFrameAtBodies = [&](LoopScheduleFrameOp frame,
                                 Value preGate) -> LogicalResult {
    // Per-offset issue gates: `at k` bodies fire k cycles after the
    // frame's header pulse, matching their scheduled cycle. Two static
    // accesses on the SAME port at different offsets would otherwise
    // both fire on the header cycle and the later one's address mux
    // would shadow the earlier one's read entirely.
    unsigned frameId = staticFrameCounter++;
    SmallVector<Value> offsetGates{preGate};
    auto gateAt = [&](unsigned offset) -> Value {
      while (offsetGates.size() <= offset) {
        Value zero1 =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
        auto name = builder.getStringAttr(
            "frame" + std::to_string(frameId) + "_hdr_d" +
            std::to_string(offsetGates.size()));
        offsetGates.push_back(seq::CompRegOp::create(
            builder, loc, offsetGates.back(), clk, rst, zero1, name));
      }
      return offsetGates[offset];
    };

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
      Value atGate = gateAt((unsigned)atOp.getOffset());
      for (auto &op : atOp.getBodyBlock()) {
        if (isa<LoopScheduleYieldOp, LoopScheduleIterArgUpdateOp,
                LoopScheduleLaunchOp>(&op))
          continue;
        // Dynamic-latency accesses need the full issue/completion handshake
        // (SeqDynCtx), which the function-level frame path does not provide.
        // Lowering them as static single-cycle drives silently produces
        // wrong hardware in general (no ready gating, no completion wait,
        // same-cycle clobbering of a shared port). The ONE historically
        // supported shape stays: a posted (`no_wait`) store that is its
        // port's only static access in the function frames — one issue
        // pulse, nothing to wait for, nothing to clobber. Everything else
        // is rejected, matching the explicit error lowerSeqDynAccess raises
        // without a context.
        if (isSeqDynAccess(&op)) {
          auto asStore = dyn_cast<loopschedule::HWStoreLoweringInterface>(&op);
          bool posted = op.hasAttr("no_wait");
          bool multi =
              asStore && funcMultiAccess.contains(
                             {asStore.getMemoryValue(), getBindingPort(&op)});
          if (!asStore || !posted || multi)
            return op.emitError(
                "dynamic memory accesses in function-level frames are not "
                "yet supported by the FSM lowering (only a posted `no_wait` "
                "store that is its port's sole access is allowed); keep the "
                "access inside a loop");
        }
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
          if (failed(handleHWStore(storeOp, builder, mapping, atGate,
                                      localPorts)))
            return failure();
          Value target = storeOp.getMemoryValue();
          staticDrives.emplace_back(target, std::move(localPorts[target]));
          continue;
        }
        if (auto loadOp =
                dyn_cast<loopschedule::HWLoadLoweringInterface>(&op)) {
          // Like the store path above: drive into a temporary map and
          // stash, so the addr/rd_en survive the `mergeStepMemPorts`
          // overwrite. The result mapping (load -> rdData wire) is
          // unaffected by the merge and needs the real rdData here.
          DenseMap<Value, MemPortMapping> localPorts;
          unsigned latency = 1;
          for (auto &memInfo : memrefArgs) {
            localPorts[memInfo.originalArg].rdData =
                memPortMap[memInfo.originalArg].rdData;
            if (memInfo.originalArg == loadOp.getMemoryValue())
              latency = std::max(1u, memInfo.latency);
          }
          if (failed(handleHWLoad(loadOp, builder, mapping, localPorts,
                                     atGate)))
            return failure();
          Value target = loadOp.getMemoryValue();
          // The port's rd_data wire only holds this load's data until the
          // port's NEXT read — and another hoisted load may share the
          // port (e.g. two coefficient reads at different frame offsets).
          // Capture the data into a register on its valid cycle and hand
          // consumers a live-cycle bypass, mirroring handleHWLoad's
          // contended-port path inside loops.
          Value rdData = memPortMap[target].rdData;
          Value dataGate = gateAt((unsigned)atOp.getOffset() + latency);
          Value zeroD = createZeroConstant(builder, loc, rdData.getType());
          auto capName = builder.getStringAttr(
              "frame" + std::to_string(frameId) + "_ldcap_" +
              std::to_string(staticCapCounter++));
          Value captured = seq::CompRegClockEnabledOp::create(
              builder, loc, rdData, clk, dataGate, rst, zeroD, capName);
          Value bypass =
              comb::MuxOp::create(builder, loc, dataGate, rdData, captured);
          mapping.map(loadOp.getResult(), bypass);
          staticDrives.emplace_back(target, std::move(localPorts[target]));
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
      // Mirror the loop module's `mem*_done` input order.
      for (auto &memInfo : memrefArgs) {
        if (!memInfo.isAmcPort || !memInfo.hasDone)
          continue;
        auto &callerMp = memPortMap[memInfo.originalArg];
        Value d = callerMp.done
                      ? callerMp.done
                      : hw::ConstantOp::create(builder, loc,
                                                  builder.getI1Type(), 0);
        childInputs.push_back(d);
      }
      // Mirror the loop module's `mem*_wr_done` input order.
      for (auto &memInfo : memrefArgs) {
        if (!memInfo.isAmcPort || !memInfo.hasWrDone)
          continue;
        auto &callerMp = memPortMap[memInfo.originalArg];
        Value d = callerMp.wrDone
                      ? callerMp.wrDone
                      : hw::ConstantOp::create(builder, loc,
                                                  builder.getI1Type(), 0);
        childInputs.push_back(d);
      }
      // Mirror the loop module's `mem*_ready` input order. A missing
      // caller signal ties acceptance high (no backpressure).
      for (auto &memInfo : memrefArgs) {
        if (!memInfo.isAmcPort || !memInfo.hasReady)
          continue;
        auto &callerMp = memPortMap[memInfo.originalArg];
        Value rdy = callerMp.ready
                        ? callerMp.ready
                        : hw::ConstantOp::create(builder, loc,
                                                    builder.getI1Type(), 1);
        childInputs.push_back(rdy);
      }
      // Mirror the loop module's `mem*_rd_idle` / `mem*_wr_idle` input order. A
      // missing caller signal ties the drain high (fence is a no-op).
      for (auto &memInfo : memrefArgs) {
        if (!memInfo.isAmcPort || !memInfo.hasRdIdle)
          continue;
        auto &callerMp = memPortMap[memInfo.originalArg];
        Value v = callerMp.rdIdle
                      ? callerMp.rdIdle
                      : hw::ConstantOp::create(builder, loc,
                                                  builder.getI1Type(), 1);
        childInputs.push_back(v);
      }
      for (auto &memInfo : memrefArgs) {
        if (!memInfo.isAmcPort || !memInfo.hasWrIdle)
          continue;
        auto &callerMp = memPortMap[memInfo.originalArg];
        Value v = callerMp.wrIdle
                      ? callerMp.wrIdle
                      : hw::ConstantOp::create(builder, loc,
                                                  builder.getI1Type(), 1);
        childInputs.push_back(v);
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

    } else if (entry.kind == 3) {
      // Barrier (e.g. amc.control): no child hardware to instantiate. The
      // FSM's WAIT state samples the control channel's `done` LEVEL live
      // (no latching), so the frame holds exactly until the memory reports
      // completion — for an AXI flush, until no write is outstanding.
      // child_start is unused (the channel's request side is reserved).
      Value ctrlVal = entry.barrierOp.getMemoryValue();
      Value done;
      auto it = memPortMap.find(ctrlVal);
      if (it != memPortMap.end())
        done = it->second.done;
      if (!done)
        done = hw::ConstantOp::create(builder, loc, i1, 1);
      childDoneBEs[childIndexForEntry[ei]].setValue(done);

      // The barrier yields no values; resolve its handle to nothing so a
      // downstream await maps cleanly.
      if (entry.launchOp) {
        if (auto launchAt =
                entry.launchOp->getParentOfType<LoopScheduleAtOp>()) {
          auto atYield = launchAt.getYieldOp();
          for (auto [atRes, yOperand] :
               llvm::zip(launchAt.getResults(), atYield.getOperands())) {
            if (yOperand == entry.launchOp.getHandle())
              funcHandleValueMap[atRes] = SmallVector<Value>{};
          }
        }
        funcHandleValueMap[entry.launchOp.getHandle()] = SmallVector<Value>{};
      }

    } else if (entry.kind == 4) {
      // Dynamic port access entry (launch-wrapped m_axi load/store in a
      // function frame). Mirrors lowerSeqDynAccess, with the FSM's WAIT
      // state standing in for the sequential stall machinery: child_start
      // pulses in FRAME_i, the request is held until the port accepts it,
      // and WAIT_i holds the frame until the port's completion pulse is
      // attributed to this access. Drives land in perEntryPorts[ei], so
      // mergeStepMemPorts gates them on this entry's running signal.
      Operation *dop = entry.dynOp;
      auto dLoad = dyn_cast<loopschedule::HWLoadLoweringInterface>(dop);
      auto dStore = dyn_cast<loopschedule::HWStoreLoweringInterface>(dop);
      Value memVal = dLoad ? dLoad.getMemoryValue() : dStore.getMemoryValue();
      unsigned dynPort = getBindingPort(dop);
      auto pit = perEntryPorts[ei].find(memVal);
      if (pit == perEntryPorts[ei].end())
        return dop->emitError("unmapped memory in function-frame dynamic "
                              "access");
      MemPortMapping &mp = pit->second;
      PortDrivesRef pref = portRef(mp, dynPort);

      Value startPulse = childStartSignals[childIndexForEntry[ei]];
      Value active = entryRunningSignals[ei];
      Value zero1 = hw::ConstantOp::create(builder, loc, i1, 0);
      std::string base =
          funcOp.getName().str() + "_fdyn" + std::to_string(ei);

      // Request-held latch: the start pulse is one cycle, but the port may
      // not be ready that cycle. Clears when the entry deactivates, so a
      // later kernel invocation starts fresh.
      Backedge wantNextBE = bb.get(i1);
      auto wantReg =
          seq::CompRegOp::create(builder, loc, Value(wantNextBE), clk, rst,
                                 zero1, builder.getStringAttr(base + "_want"));
      Value want =
          comb::OrOp::create(builder, loc, startPulse, Value(wantReg), false);
      wantNextBE.setValue(
          comb::AndOp::create(builder, loc, want, active, false));

      // Accepted latch: one-shot issue.
      Backedge accNextBE = bb.get(i1);
      auto accReg =
          seq::CompRegOp::create(builder, loc, Value(accNextBE), clk, rst,
                                 zero1, builder.getStringAttr(base + "_acc"));
      Value notAcc = comb::createOrFoldNot(builder, loc, accReg);
      Value issue = comb::AndOp::create(builder, loc, want, notAcc, false);
      if (mp.ready)
        issue = comb::AndOp::create(builder, loc, issue, mp.ready, false);
      accNextBE.setValue(comb::AndOp::create(
          builder, loc,
          comb::OrOp::create(builder, loc, accReg, issue, false), active,
          false));

      // Operand resolution: awaited frame results arrive via `mapping`;
      // constants defined at function scope are cloned on demand.
      auto resolve = [&](Value v) -> Value {
        if (auto m = mapping.lookupOrNull(v))
          return m;
        if (auto *def = v.getDefiningOp();
            def && def->hasTrait<OpTrait::ConstantLike>())
          return builder.clone(*def, mapping)->getResult(0);
        return Value();
      };

      SmallVector<unsigned> widths =
          dLoad ? dLoad.getAddrWidths() : dStore.getAddrWidths();
      auto indices = dLoad ? dLoad.getIndices() : dStore.getIndices();
      pref.addrs->resize(widths.size());
      for (auto [d, idx] : llvm::enumerate(indices)) {
        Value a = resolve(idx);
        if (!a)
          return dop->emitError("unresolved address operand in "
                                "function-frame dynamic access");
        (*pref.addrs)[d] = resizeIntTo(builder, loc, a, widths[d]);
      }

      Value done;
      if (dStore) {
        // Data-less dynamic stores (amc.burst_copy: the data plane lives
        // inside the copy engine) drive only the enable.
        if (Value toStore = dStore.getValueToStore()) {
          Value wrData = resolve(toStore);
          if (!wrData)
            return dop->emitError("unresolved store operand in "
                                  "function-frame dynamic access");
          *pref.wrData = wrData;
        }
        *pref.wrEn = issue;
        // rw faces split write completion out; write-only ports report it
        // on `done`.
        done = mp.wrDone ? mp.wrDone : mp.done;
      } else {
        if (dLoad.requiresReadEnable())
          *pref.rdEn = issue;
        done = mp.done;
      }

      // Completion attribution: the FIRST post-issue pulse belongs to this
      // access; the sticky `seen` latch keeps later pulses on the shared
      // port from re-triggering, and clears when the entry deactivates.
      Value childDone;
      Value doneMine;
      if (done) {
        Backedge seenNextBE = bb.get(i1);
        auto seenReg = seq::CompRegOp::create(
            builder, loc, Value(seenNextBE), clk, rst, zero1,
            builder.getStringAttr(base + "_seen"));
        Value notSeen = comb::createOrFoldNot(builder, loc, seenReg);
        doneMine = comb::AndOp::create(
            builder, loc,
            comb::AndOp::create(builder, loc, done, Value(accReg), false),
            notSeen, false);
        Value seenFed =
            comb::OrOp::create(builder, loc, Value(seenReg), doneMine, false);
        seenNextBE.setValue(
            comb::AndOp::create(builder, loc, seenFed, active, false));
        childDone = seenFed;
      } else {
        // No completion signal on this port: treat acceptance as done.
        childDone = accReg;
      }
      childDoneBEs[childIndexForEntry[ei]].setValue(childDone);

      // Loads deliver their data via the launch-handle map; capture on the
      // completion pulse since consumers run in later FSM states.
      SmallVector<Value> resultVals;
      if (dLoad) {
        Value cap = mp.rdData;
        if (mp.rdData && dLoad.requiresReadEnable() &&
            dLoad.getReadLatency() == 0) {
          // FWFT stream pop: the beat is valid ON the pop (issue) cycle and
          // the FIFO head advances right after — capture at issue, which is
          // gated on the port's beat-valid `ready`. The doneMine-timed
          // capture below would latch the NEXT beat (accReg registers one
          // cycle behind the pop); acceptance is completion here.
          cap = seq::CompRegClockEnabledOp::create(
              builder, loc, mp.rdData, clk, issue, rst,
              createZeroConstant(builder, loc, mp.rdData.getType()),
              builder.getStringAttr(base + "_cap"));
        } else if (done && mp.rdData) {
          cap = seq::CompRegClockEnabledOp::create(
              builder, loc, mp.rdData, clk, doneMine, rst,
              createZeroConstant(builder, loc, mp.rdData.getType()),
              builder.getStringAttr(base + "_cap"));
        }
        mapping.map(dLoad->getResult(0), cap);
        resultVals.push_back(cap);
      }
      if (entry.launchOp) {
        if (auto launchAt =
                entry.launchOp->getParentOfType<LoopScheduleAtOp>()) {
          auto atYield = launchAt.getYieldOp();
          for (auto [atRes, yOperand] :
               llvm::zip(launchAt.getResults(), atYield.getOperands())) {
            if (yOperand == entry.launchOp.getHandle())
              funcHandleValueMap[atRes] = resultVals;
          }
        }
        funcHandleValueMap[entry.launchOp.getHandle()] = resultVals;
      }

    } else {
      // Leaf frame: lower body with the entry's per-cycle gates (a
      // single-cycle frame's gate vector is just entry_running; a
      // multi-cycle frame gets the FSM's frame_cycle_<i>_<c> outputs so
      // `at K` bodies issue at their scheduled cycle). Contended ports
      // (multiple static accesses across the function's frames, e.g. two
      // stores to one memory serialized to `at 0`/`at 1`) need the
      // gate-muxed drives — without a SeqPortMuxCtx the later access's
      // addr/data/enable would simply overwrite the earlier one's.
      SeqPortMuxCtx seqMux;
      seqMux.multi = funcMultiAccess;
      for (auto &memInfo : memrefArgs)
        seqMux.registeredMems.insert(memInfo.originalArg);
      seqMux.cycleGates = entryCycleGates[ei];
      seqMux.clk = clk;
      seqMux.rst = rst;
      seqMux.regPrefix = funcOp.getName().str() + "_e" + std::to_string(ei);
      SeqPortMuxCtx *seqMuxPtr = seqMux.multi.empty() ? nullptr : &seqMux;
      if (failed(lowerFrameBody(&topFrames[frameIdx].getBodyBlock(), builder,
                                mapping, entryCycleGates[ei],
                                perEntryPorts[ei], enclosingModule, clk,
                                rst, /*seqDyn=*/nullptr, seqMuxPtr)))
        return failure();
    }

    // Register frame results only after the last entry for this frame
    // completes.
    if (ei == lastEntryForFrame[frameIdx]) {
      // Pre-scan memref loads in this frame. Local hlmem reads hold their
      // output (the read enable drops after the frame), so their results
      // map raw. External BRAM ports are always-enabled latency-1 reads:
      // the data for this frame's address is on the wire only during the
      // NEXT FSM state, so their results are captured one state later.
      DenseSet<Value> localLoadResults;
      DenseSet<Value> externalLoadResults;
      topFrames[frameIdx]->walk([&](LoopScheduleLoadOp loadOp) {
        bool isLocal = false;
        for (auto &memInfo : memrefArgs)
          if (memInfo.originalArg == loadOp.getMemRef() &&
              memInfo.isLocalMem)
            isLocal = true;
        (isLocal ? localLoadResults : externalLoadResults)
            .insert(loadOp.getResult());
      });

      // Forward each at's external results from its yield operands so the
      // frame body yield (whose operands reference at results) resolves.
      // Propagate the load classification through the at-yield forwarding
      // so the frame-result loop below sees it on the yield operands.
      for (auto atOp :
           topFrames[frameIdx].getBodyBlock().getOps<LoopScheduleAtOp>()) {
        auto atYield = atOp.getYieldOp();
        for (auto [res, val] :
             llvm::zip(atOp.getResults(), atYield.getOperands())) {
          if (localLoadResults.count(val))
            localLoadResults.insert(res);
          if (externalLoadResults.count(val))
            externalLoadResults.insert(res);
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
      Value delayedEntryGate;
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
        // Multi-cycle leaf frames capture on their LAST cycle gate so
        // results computed in later `at` offsets are latched, not the
        // cycle-0 values.
        Value captureGate = entryCycleGates[ei].back();
        if (externalLoadResults.count(yOperand)) {
          // External BRAM read: the data arrives the state after the
          // address was presented, so delay the capture gate one cycle.
          if (!delayedEntryGate) {
            Value falseVal =
                hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
            delayedEntryGate = seq::CompRegOp::create(
                builder, loc, entryRunningSignals[ei], clk, rst, falseVal,
                builder.getStringAttr(funcName + "_frame" +
                                      std::to_string(frameIdx) +
                                      "_capture_gate_delayed"));
          }
          captureGate = delayedEntryGate;
        }
        auto regName = builder.getStringAttr(
            funcName + "_frame" + std::to_string(frameIdx) + "_result_" +
            std::to_string(frameResult.getResultNumber()));
        Value resetVal = createZeroConstant(builder, loc, val.getType());
        auto reg = seq::CompRegClockEnabledOp::create(
            builder, loc, val, clk, captureGate, rst, resetVal,
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
                    memrefArgs, mergedMemPorts, entryGroups);

  // Priority-compose the static at-body access drives (collected by
  // `cloneFrameAtBodies`) over the merged entry drives. Each static
  // drive's wrEn (stores) / rdEn (loads) is already gated by its owning
  // frame's header pulse, so the static access fires during that one
  // cycle and the child's drives take over for the rest of the frame.
  for (auto &[memVal, staticMp] : staticDrives) {
    auto &merged = mergedMemPorts[memVal];
    Value sGate = staticMp.wrEn ? staticMp.wrEn : staticMp.rdEn;
    if (!sGate)
      continue;
    if (staticMp.wrEn) {
      if (merged.wrEn)
        merged.wrEn = comb::OrOp::create(builder, loc, sGate, merged.wrEn);
      else
        merged.wrEn = sGate;
    }
    if (staticMp.rdEn) {
      if (merged.rdEn)
        merged.rdEn =
            comb::OrOp::create(builder, loc, staticMp.rdEn, merged.rdEn);
      else
        merged.rdEn = staticMp.rdEn;
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

  // Copy merged ports to function-level memPortMap. `rdEn` propagates for
  // every port that declares one: amc ports need it so the stall-aware
  // gate built in `lowerPipelineChild` from the per-port `done` handshake
  // reaches the amc.instance's rd_en backedge, and external memref ports
  // need it for the BRAM-port hold contract (without it the module output
  // ties rd_en low and the external memory never updates its read
  // register).
  for (auto &memInfo : memrefArgs) {
    memPortMap[memInfo.originalArg].addrs =
        mergedMemPorts[memInfo.originalArg].addrs;
    memPortMap[memInfo.originalArg].wrData =
        mergedMemPorts[memInfo.originalArg].wrData;
    memPortMap[memInfo.originalArg].wrEn =
        mergedMemPorts[memInfo.originalArg].wrEn;
    if (memInfo.requiresRdEn)
      memPortMap[memInfo.originalArg].rdEn =
          mergedMemPorts[memInfo.originalArg].rdEn;
  }

  // Resolve hlmem + amc.instance backedges left dangling by the prelude.
  resolveFunctionMemoryBackedges(builder, loc, hwBody, hlmemBEs, memInstState,
                                  memPortMap);

  // Build hw.output. `ready` = "no transaction in flight" — the sticky
  // bit built next to the FSM instance (set on start, cleared on done);
  // its done backedge resolves against the final (early-done-OR'd)
  // signal here.
  builder.setInsertionPointToEnd(hwBody);
  funcDoneBE.setValue(doneSignal);
  Value readySignal = comb::createOrFoldNot(builder, loc, inFlightReg);

  buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, readySignal,
                doneSignal);
  appendBramBoundaries(hwMod, bramBoundaries);

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
  sharedOperators.clear();
  auto enclosingModule = funcOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToEnd(hwBody);

  // --- Prelude: clone non-stage ops, set up local hlmems + amc instances ---
  auto i1 = builder.getI1Type();
  BackedgeBuilder funcBB(builder, loc);
  SmallVector<HLMemBackedges> hlmemBEs;
  SymbolTable modSymTab(enclosingModule);
  DenseMap<Value, loopschedule::BramBoundary> bramBoundaries;
  loopschedule::HWMemoryLoweringState memInstState(clk, rst, funcBB, modSymTab,
                                                    bramBoundaries);
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
          // Every memref-backed load is a latency-1 registered read:
          // local hlmems by construction, and external ports by the
          // BRAM-port contract (the external memory's output register
          // provides the 1-cycle delay, so no stage register is added).
          if (auto ls = dyn_cast<LoopScheduleLoadOp>(inner))
            localLoadResults.insert(ls.getResult());
          if (loadOp.requiresReadEnable() && loadOp.getReadLatency() > 0)
            localLoadResults.insert(loadOp.getResult());
          return handleHWLoad(loadOp, builder, mapping,
                              perStagePorts[stageIdx], gate);
        }
        return emitComputeOp(inner, builder, mapping, enclosingModule, clk,
                             rst, /*opCE=*/{}, /*shareGate=*/gate);
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
      if (computeValueCycleLatency(val, operatorLibrary) > 0) {
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
  appendBramBoundaries(hwMod, bramBoundaries);

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
