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

#include "circt/Conversion/LoopScheduleToFSM.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"
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

/// Recursively compute the latency contribution of a region of LoopSchedule
/// ops. Direct children contribute their own cycle_latency at offset 0; each
/// `loopschedule.delay` child contributes `delay.latency + region body
/// latency`. The result is `max(over body ops) issueCycle + opCycleLatency`.
static unsigned computeRegionCycleLatency(Block &block) {
  unsigned maxLat = 0;
  for (Operation &op : block) {
    if (isa<LoopScheduleRegisterOp>(op))
      continue;
    if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(op))
      continue;
    if (auto delayOp = dyn_cast<LoopScheduleDelayOp>(&op)) {
      unsigned childLat = computeRegionCycleLatency(delayOp.getBodyBlock());
      maxLat = std::max(maxLat,
                        (unsigned)delayOp.getLatency() + std::max(childLat, 1u));
      continue;
    }
    maxLat = std::max(maxLat, getOpCycleLatency(&op));
  }
  return std::max(maxLat, 1u);
}

/// Top-level latency of a step.
static unsigned computeStepLatency(LoopScheduleStepOp step) {
  return computeRegionCycleLatency(step.getBodyBlock());
}

/// Tracks the hw.module ports for a memref function argument.
/// `addrs` carries one address Value per memref dim (empty for 0-rank).
struct MemPortMapping {
  Value rdData;             // input: read data from memory
  SmallVector<Value> addrs; // will be set: per-dim address outputs
  Value wrData;             // will be set: write data output
  Value wrEn;               // will be set: write enable output
};

/// Information about a memref argument for threading through loop modules.
struct MemrefArgInfo {
  Value originalArg; // original func::FuncOp argument
  MemRefType memType;
  bool isLocalMem = false; // true for memref.alloc → seq.hlmem
};

/// Append hw.module output port declarations for a memref: per-dim address,
/// write data, and write enable.
static void appendMemrefOutputPorts(OpBuilder &builder, StringRef baseName,
                                    MemRefType memType,
                                    SmallVectorImpl<hw::PortInfo> &ports) {
  auto *ctx = builder.getContext();
  auto widths = getDimAddrWidths(memType);
  Type dataType = memType.getElementType();
  bool isOneDim = widths.size() == 1;
  for (auto [d, w] : llvm::enumerate(widths)) {
    std::string addrName =
        isOneDim ? (baseName + "_addr").str()
                 : (baseName + "_addr_" + std::to_string(d)).str();
    ports.push_back({{builder.getStringAttr(addrName),
                       IntegerType::get(ctx, w),
                       hw::ModulePort::Direction::Output}});
  }
  ports.push_back({{builder.getStringAttr((baseName + "_wr_data").str()),
                     dataType, hw::ModulePort::Direction::Output}});
  ports.push_back({{builder.getStringAttr((baseName + "_wr_en").str()),
                     builder.getI1Type(), hw::ModulePort::Direction::Output}});
}

/// Append hw.output values for a memref's output ports (addr, wr_data, wr_en),
/// falling back to zero constants when the mapping is absent or incomplete.
static void appendMemrefOutputValues(OpBuilder &builder, Location loc,
                                     MemRefType memType, Value memKey,
                                     DenseMap<Value, MemPortMapping> &memPortMap,
                                     SmallVectorImpl<Value> &outputs) {
  auto *ctx = builder.getContext();
  auto widths = getDimAddrWidths(memType);
  Type dataType = memType.getElementType();
  auto it = memPortMap.find(memKey);
  bool has = it != memPortMap.end();
  for (auto [d, w] : llvm::enumerate(widths)) {
    Type addrType = IntegerType::get(ctx, w);
    if (has && d < it->second.addrs.size() && it->second.addrs[d])
      outputs.push_back(it->second.addrs[d]);
    else
      outputs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
  }
  if (has && it->second.wrData)
    outputs.push_back(it->second.wrData);
  else
    outputs.push_back(hw::ConstantOp::create(builder, loc, dataType, 0));
  if (has && it->second.wrEn)
    outputs.push_back(it->second.wrEn);
  else
    outputs.push_back(
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0));
}

/// Represents one sequential loop in the nesting tree.
struct LoopNode {
  LoopScheduleSequentialOp seqOp;
  std::string prefix;              // e.g., "loop0", "loop0_loop1"
  std::vector<LoopNode> children;
  /// For step index i, stepChildIdx[i] = index into children, or -1.
  SmallVector<int> stepChildIdx;
  /// For step index i, stepPipelineIdx[i] = index into pipelineChildren, or -1.
  SmallVector<int> stepPipelineIdx;
  SmallVector<LoopSchedulePipelineOp> pipelineChildren;
  bool isLeaf() const {
    return children.empty() && pipelineChildren.empty();
  }
  bool hasChild(unsigned stepIdx) const {
    return stepChildIdx[stepIdx] >= 0 || stepPipelineIdx[stepIdx] >= 0;
  }
};

/// Main pass converting LoopSchedule ops to FSM + HW.
class LoopScheduleToFSMPass
    : public circt::impl::LoopScheduleToFSMBase<LoopScheduleToFSMPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult lowerFunction(func::FuncOp funcOp);

  /// Build the nesting tree from a sequential op.
  void buildLoopTree(LoopScheduleSequentialOp seqOp, LoopNode &node,
                     const std::string &prefix, unsigned &loopCounter);

  /// Create FSM machine for a sequential loop.
  ///
  /// `waitStepIndices` lists the step indices whose body launches an external
  /// computation that must be waited on (a child sequential loop, a pipeline
  /// child, or any future variable-latency op). It must be sorted ascending.
  /// Each such step gets dedicated child_start/child_done/post_active signals
  /// and inserts WAIT_<i>/POST_<i> states around STEP_<i>. Steps not in the
  /// list are "regular" steps that drive step_active_<i> in their STEP_<i>
  /// state.
  ///
  /// Inputs (fixed order):
  ///   start, cond, child_done_0..C-1            (C = waitStepIndices.size())
  ///
  /// Outputs (fixed order):
  ///   done, first_iter, iter_advance,
  ///   step_active_0..N-1,                       (N = numSteps)
  ///   child_start_0..C-1,
  ///   post_active_0..C-1
  fsm::MachineOp createSequentialFSM(OpBuilder &builder, Location loc,
                                      StringRef fsmName, unsigned numSteps,
                                      ArrayRef<unsigned> waitStepIndices,
                                      ArrayRef<unsigned> stepLatencies = {});

  /// Create linear run-once FSM for sequencing top-level function steps.
  /// stepChildKind[i]: -1 = leaf, 0 = sequential child, 1 = pipeline child.
  fsm::MachineOp createFunctionFSM(OpBuilder &builder, Location loc,
                                    StringRef fsmName,
                                    ArrayRef<int> stepChildKind);

  /// Recursively lower a loop node as its own hw.module.
  /// Creates the module and populates outModule. capturedVals are the
  /// external values the caller must wire as inputs when instantiating.
  LogicalResult lowerLoopNodeAsModule(
      const LoopNode &node, OpBuilder &builder, Location loc,
      func::FuncOp funcOp, ArrayRef<MemrefArgInfo> memrefArgs,
      IRMapping &parentMapping,
      hw::HWModuleOp &outModule,
      SmallVectorImpl<Value> &capturedVals);

  /// Lower step body ops, gating store wrEn with the per-issue-cycle active
  /// signal. `cycleGates[c]` is the i1 high in cycle `c` of the enclosing
  /// step. For single-cycle steps the array has one entry equal to the
  /// step's overall active signal.
  LogicalResult lowerStepBody(Block *stepBody, OpBuilder &builder,
                              IRMapping &mapping,
                              ArrayRef<Value> cycleGates,
                              DenseMap<Value, MemPortMapping> &memPorts);

  /// Lower a pipeline as a child of a sequential loop (no FSM needed).
  LogicalResult lowerPipelineChild(LoopSchedulePipelineOp pipOp,
                                   OpBuilder &builder, Location loc,
                                   Block *hwBody, IRMapping &mapping,
                                   Value clk, Value rst, Value startSignal,
                                   StringRef namePrefix, Value &doneSignal,
                                   DenseMap<Value, MemPortMapping> &memPorts,
                                   ArrayRef<MemrefArgInfo> memrefArgs);

  /// Map from original func memref args to their hw.module port values.
  DenseMap<Value, MemPortMapping> memPortMap;

  /// Backedge info for a local hlmem (memref.alloc).
  struct HLMemBackedges {
    Value allocResult;          // original memref.alloc result
    SmallVector<Backedge> addrBEs; // one per memref dim (in hlmem's widths)
    Backedge wrDataBE;
    Backedge wrEnBE;
  };
};

//===----------------------------------------------------------------------===//
// Load/store helpers
//===----------------------------------------------------------------------===//

/// Resolve the per-dim address values for a memref access. Looks up the
/// memref's port mapping, walks the access's indices, and width-fixes each
/// to the corresponding entry in `getDimAddrWidths(memType)`. Stores the
/// resulting addresses into `portsOut->addrs`.
static LogicalResult
prepareMemAccess(Operation *op, Value memref, ValueRange indexVals,
                 OpBuilder &builder, IRMapping &mapping,
                 DenseMap<Value, MemPortMapping> &memPorts,
                 MemPortMapping *&portsOut) {
  auto it = memPorts.find(memref);
  if (it == memPorts.end())
    return op->emitError("unmapped memref");
  portsOut = &it->second;
  auto memType = cast<MemRefType>(memref.getType());
  auto widths = getDimAddrWidths(memType);
  if (indexVals.size() != widths.size())
    return op->emitError("loopschedule access index count (")
           << indexVals.size() << ") does not match memref rank ("
           << widths.size() << ")";
  portsOut->addrs.resize(widths.size());
  for (auto [d, idx] : llvm::enumerate(indexVals)) {
    Value addr = mapping.lookup(idx);
    portsOut->addrs[d] = resizeIntTo(builder, op->getLoc(), addr, widths[d]);
  }
  return success();
}

/// Lower a loopschedule.load: drive the read addresses on the memory port
/// mapping and map the load result to the port's read data.
static LogicalResult
handleLoad(LoopScheduleLoadOp loadOp, OpBuilder &builder, IRMapping &mapping,
           DenseMap<Value, MemPortMapping> &memPorts) {
  MemPortMapping *ports = nullptr;
  if (failed(prepareMemAccess(loadOp, loadOp.getMemRef(), loadOp.getIndices(),
                              builder, mapping, memPorts, ports)))
    return failure();
  mapping.map(loadOp.getResult(), ports->rdData);
  return success();
}

/// Lower a loopschedule.store: drive the addresses, write data, and write
/// enable (gated by wrEnGate) on the memory port mapping.
static LogicalResult
handleStore(LoopScheduleStoreOp storeOp, OpBuilder &builder,
            IRMapping &mapping, Value wrEnGate,
            DenseMap<Value, MemPortMapping> &memPorts) {
  assert(wrEnGate && "handleStore requires a wrEnGate");
  MemPortMapping *ports = nullptr;
  if (failed(prepareMemAccess(storeOp, storeOp.getMemRef(), storeOp.getIndices(),
                              builder, mapping, memPorts, ports)))
    return failure();
  ports->wrData = mapping.lookup(storeOp.getValue());
  ports->wrEn = wrEnGate;
  return success();
}

//===----------------------------------------------------------------------===//
// Step body lowering
//===----------------------------------------------------------------------===//

/// Recursive worker that lowers ops in a step or delay body. `baseCycle` is
/// the issue cycle of the immediately enclosing region (0 for the step body,
/// `delay.latency + parent base` for nested delays).
static LogicalResult
lowerRegionBody(Block *body, OpBuilder &builder, IRMapping &mapping,
                ArrayRef<Value> cycleGates, unsigned baseCycle,
                DenseMap<Value, MemPortMapping> &memPorts) {
  auto pickGate = [&](unsigned c) -> Value {
    assert(c < cycleGates.size() &&
           "issue cycle exceeds enclosing step latency");
    return cycleGates[c];
  };
  for (auto &op : *body) {
    if (isa<LoopScheduleRegisterOp>(&op))
      continue;
    if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(&op))
      continue;

    if (auto delayOp = dyn_cast<LoopScheduleDelayOp>(&op)) {
      unsigned childBase = baseCycle + (unsigned)delayOp.getLatency();
      if (failed(lowerRegionBody(&delayOp.getBodyBlock(), builder, mapping,
                                 cycleGates, childBase, memPorts)))
        return failure();
      // Map the delay op's external results to the cloned register operands.
      auto regOp = delayOp.getRegisterOp();
      for (auto [res, regVal] :
           llvm::zip(delayOp.getResults(), regOp.getOperands()))
        mapping.map(res, mapping.lookup(regVal));
      continue;
    }

    if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
      if (failed(handleStore(storeOp, builder, mapping, pickGate(baseCycle),
                              memPorts)))
        return failure();
      continue;
    }

    if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
      if (failed(handleLoad(loadOp, builder, mapping, memPorts)))
        return failure();
      continue;
    }

    builder.clone(op, mapping);
  }
  return success();
}

LogicalResult LoopScheduleToFSMPass::lowerStepBody(
    Block *stepBody, OpBuilder &builder, IRMapping &mapping,
    ArrayRef<Value> cycleGates,
    DenseMap<Value, MemPortMapping> &memPorts) {
  return lowerRegionBody(stepBody, builder, mapping, cycleGates, 0, memPorts);
}

//===----------------------------------------------------------------------===//
// Loop tree construction
//===----------------------------------------------------------------------===//

void LoopScheduleToFSMPass::buildLoopTree(
    LoopScheduleSequentialOp seqOp, LoopNode &node,
    const std::string &prefix, unsigned &loopCounter) {
  node.seqOp = seqOp;
  node.prefix = prefix;

  SmallVector<LoopScheduleStepOp> steps;
  for (auto &op : seqOp.getScheduleBlock().getOperations())
    if (auto stepOp = dyn_cast<LoopScheduleStepOp>(&op))
      steps.push_back(stepOp);

  node.stepChildIdx.resize(steps.size(), -1);
  node.stepPipelineIdx.resize(steps.size(), -1);

  for (auto [stepIdx, stepOp] : llvm::enumerate(steps)) {
    for (auto &op : stepOp.getBodyBlock().getOperations()) {
      if (auto childSeq = dyn_cast<LoopScheduleSequentialOp>(&op)) {
        unsigned childIdx = node.children.size();
        node.stepChildIdx[stepIdx] = childIdx;
        node.children.emplace_back();
        std::string childPrefix =
            prefix + "_loop" + std::to_string(loopCounter++);
        buildLoopTree(childSeq, node.children.back(), childPrefix,
                      loopCounter);
        break; // At most one child per step.
      }
      if (auto childPip = dyn_cast<LoopSchedulePipelineOp>(&op)) {
        unsigned pipIdx = node.pipelineChildren.size();
        node.stepPipelineIdx[stepIdx] = pipIdx;
        node.pipelineChildren.push_back(childPip);
        break; // At most one child per step.
      }
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
    OpBuilder &builder, Location loc, StringRef fsmName, unsigned numSteps,
    ArrayRef<unsigned> waitStepIndices, ArrayRef<unsigned> stepLatencies) {
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();

  // For each step, the index into waitStepIndices if it's a wait step, or -1.
  SmallVector<int> stepWaitIdx(numSteps, -1);
  for (auto [j, i] : llvm::enumerate(waitStepIndices)) {
    assert(i < numSteps && "wait step index out of range");
    assert(stepWaitIdx[i] == -1 && "duplicate wait step index");
    stepWaitIdx[i] = (int)j;
  }
  // Verify ascending order (caller contract).
  for (unsigned k = 1; k < waitStepIndices.size(); ++k)
    assert(waitStepIndices[k - 1] < waitStepIndices[k] &&
           "waitStepIndices must be sorted ascending");

  unsigned numWaits = waitStepIndices.size();

  // Normalize stepLatencies — default 1 per step.
  SmallVector<unsigned> stepLats(numSteps, 1);
  for (unsigned i = 0; i < numSteps && i < stepLatencies.size(); ++i)
    stepLats[i] = std::max(stepLatencies[i], 1u);
  // Wait steps must be single-cycle: their step body launches a child whose
  // latency is variable, so the bucket merger is forbidden from coalescing
  // additional ops into them. Enforce as a contract.
  for (unsigned i = 0; i < numSteps; ++i)
    assert((stepWaitIdx[i] < 0 || stepLats[i] == 1) &&
           "wait step must have latency 1");

  // Compute the per-step base index in the "step_cycle" output region. Steps
  // with latency > 1 contribute L_i outputs; single-cycle steps contribute 0
  // (their per-cycle gate is the step_active_i output).
  SmallVector<int> cycleOutBase(numSteps, -1);
  unsigned totalCycleOuts = 0;
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepLats[i] > 1) {
      cycleOutBase[i] = (int)totalCycleOuts;
      totalCycleOuts += stepLats[i];
    }
  }

  // Inputs: start, cond, child_done_0..C-1.
  SmallVector<Type> inputTypes;
  inputTypes.push_back(i1); // start
  inputTypes.push_back(i1); // cond
  for (unsigned j = 0; j < numWaits; ++j)
    inputTypes.push_back(i1); // child_done_j

  // Outputs: done, first_iter, iter_advance,
  //          step_active_0..N-1,
  //          child_start_0..C-1, child_active_0..C-1, post_active_0..C-1,
  //          step_cycle_<i>_<c>... (one per multi-cycle step's sub-cycle).
  // child_start_j  : 1-cycle pulse in STEP_<wait_step_j>; drives child start.
  // child_active_j : high while child j is producing on the wires
  //                  (in STEP_<wait_step_j> launch AND throughout WAIT_<...>).
  //                  Used to mux child j's mem ports out of the parent module.
  // post_active_j  : high in POST_<wait_step_j> (parent's post-child phase).
  // step_cycle_<i>_<c> : high in the c-th sub-state of STEP_<i>; only emitted
  //                      for steps with latency > 1.
  unsigned numOutputs = 3 + numSteps + 3 * numWaits + totalCycleOuts;
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
  for (unsigned i = 0; i < numSteps; ++i)
    resNames.push_back(
        builder.getStringAttr("step_active_" + std::to_string(i)));
  for (unsigned j = 0; j < numWaits; ++j)
    resNames.push_back(
        builder.getStringAttr("child_start_" + std::to_string(j)));
  for (unsigned j = 0; j < numWaits; ++j)
    resNames.push_back(
        builder.getStringAttr("child_active_" + std::to_string(j)));
  for (unsigned j = 0; j < numWaits; ++j)
    resNames.push_back(
        builder.getStringAttr("post_active_" + std::to_string(j)));
  for (unsigned i = 0; i < numSteps; ++i) {
    if (cycleOutBase[i] < 0)
      continue;
    for (unsigned c = 0; c < stepLats[i]; ++c)
      resNames.push_back(builder.getStringAttr(
          "step_cycle_" + std::to_string(i) + "_" + std::to_string(c)));
  }
  machine.setResNamesAttr(builder.getArrayAttr(resNames));

  OpBuilder fb(ctx);
  fb.setInsertionPointToEnd(&machine.getBody().front());

  Value trueVal = hw::ConstantOp::create(fb, loc, i1, 1);
  Value falseVal = hw::ConstantOp::create(fb, loc, i1, 0);
  auto fiVar = fsm::VariableOp::create(
      fb, loc, i1, fb.getBoolAttr(true), "first_iter");

  // Helper: build output vector.
  //   activeStep    : index of step_active to drive high (-1 = none)
  //   activeChild   : index into waitStepIndices for child_start (-1 = none)
  //   activeLive    : index into waitStepIndices for child_active (-1 = none)
  //   activePost    : index into waitStepIndices for post_active (-1 = none)
  //   cycleStep, cycleIdx : if cycleStep >= 0 drives step_cycle_<cycleStep>_<cycleIdx>
  auto buildOut = [&](Value done, bool iterAdv, int activeStep,
                      int activeChild, int activeLive,
                      int activePost, int cycleStep = -1,
                      int cycleIdx = -1) -> SmallVector<Value> {
    SmallVector<Value> v;
    v.push_back(done);
    v.push_back(fiVar);
    v.push_back(iterAdv ? trueVal : falseVal);
    for (unsigned i = 0; i < numSteps; ++i)
      v.push_back((int)i == activeStep ? trueVal : falseVal);
    for (unsigned j = 0; j < numWaits; ++j)
      v.push_back((int)j == activeChild ? trueVal : falseVal);
    for (unsigned j = 0; j < numWaits; ++j)
      v.push_back((int)j == activeLive ? trueVal : falseVal);
    for (unsigned j = 0; j < numWaits; ++j)
      v.push_back((int)j == activePost ? trueVal : falseVal);
    // Per-cycle outputs (multi-cycle steps only).
    for (unsigned i = 0; i < numSteps; ++i) {
      if (cycleOutBase[i] < 0)
        continue;
      for (unsigned c = 0; c < stepLats[i]; ++c) {
        bool on = ((int)i == cycleStep) && ((int)c == cycleIdx);
        v.push_back(on ? trueVal : falseVal);
      }
    }
    return v;
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
        fb, loc, StringRef("STEP_0"),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(1)); },
        [&]() { fsm::UpdateOp::create(fb, loc, fiVar, falseVal); });
    fsm::TransitionOp::create(fb, loc, StringRef("DONE"));
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // Helper: emit the last step's exit transition. We go back to COND so the
  // condition is evaluated with the UPDATED iter_args (the iter_arg register
  // latches on the clock edge leaving the last step, which asserts
  // iter_advance). Checking `cond` directly at the last step instead would
  // race: the register hasn't updated yet, so cond still reflects the OLD
  // iter_args — producing one extra spurious iteration.
  auto emitLastStepTransition = [&](Block *tb) {
    fb.setInsertionPointToEnd(tb);
    fsm::TransitionOp::create(fb, loc, StringRef("COND"));
  };

  // --- STEP_i (and WAIT_i / POST_i for wait steps) ---
  for (unsigned i = 0; i < numSteps; ++i) {
    bool isWait = stepWaitIdx[i] >= 0;
    bool isLast = (i + 1 == numSteps);
    std::string stepName = "STEP_" + std::to_string(i);
    std::string nextStepName =
        isLast ? "STEP_0" : "STEP_" + std::to_string(i + 1);
    unsigned L = stepLats[i];

    if (isWait || L == 1) {
      auto st = fsm::StateOp::create(fb, loc, stepName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      if (isWait) {
        fsm::OutputOp::create(
            fb, loc,
            buildOut(falseVal, /*iterAdv=*/false, /*activeStep=*/-1,
                     /*activeChild=*/stepWaitIdx[i],
                     /*activeLive=*/stepWaitIdx[i], /*activePost=*/-1));
      } else {
        bool iterAdv = isLast;
        fsm::OutputOp::create(
            fb, loc,
            buildOut(falseVal, iterAdv, /*activeStep=*/(int)i,
                     /*activeChild=*/-1, /*activeLive=*/-1,
                     /*activePost=*/-1));
      }
      Block *tb = &st.getTransitions().front();
      if (isWait) {
        fb.setInsertionPointToEnd(tb);
        std::string waitName = "WAIT_" + std::to_string(i);
        fsm::TransitionOp::create(fb, loc, StringRef(waitName));
      } else if (isLast) {
        emitLastStepTransition(tb);
      } else {
        fb.setInsertionPointToEnd(tb);
        fsm::TransitionOp::create(fb, loc, StringRef(nextStepName));
      }
      fb.setInsertionPointToEnd(&machine.getBody().front());
    } else {
      for (unsigned c = 0; c < L; ++c) {
        std::string subName =
            (c == 0) ? stepName
                     : (stepName + "_c" + std::to_string(c));
        bool isLastSub = (c + 1 == L);
        std::string nextSub = isLastSub
                                  ? nextStepName
                                  : (stepName + "_c" + std::to_string(c + 1));
        bool iterAdv = isLast && isLastSub;
        auto st = fsm::StateOp::create(fb, loc, subName);
        Block *ob = st.ensureOutput(fb);
        ob->getTerminator()->erase();
        fb.setInsertionPointToEnd(ob);
        fsm::OutputOp::create(
            fb, loc,
            buildOut(falseVal, iterAdv, /*activeStep=*/(int)i,
                     /*activeChild=*/-1, /*activeLive=*/-1,
                     /*activePost=*/-1, /*cycleStep=*/(int)i,
                     /*cycleIdx=*/(int)c));
        Block *tb = &st.getTransitions().front();
        if (isLast && isLastSub) {
          emitLastStepTransition(tb);
        } else {
          fb.setInsertionPointToEnd(tb);
          fsm::TransitionOp::create(fb, loc, StringRef(nextSub));
        }
        fb.setInsertionPointToEnd(&machine.getBody().front());
      }
    }

    if (!isWait)
      continue;

    // WAIT_i: wait for child_done_j, then go to POST_i.
    std::string waitName = "WAIT_" + std::to_string(i);
    std::string postName = "POST_" + std::to_string(i);
    unsigned childDoneArgIdx = 2 + (unsigned)stepWaitIdx[i];
    {
      auto st = fsm::StateOp::create(fb, loc, waitName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      fsm::OutputOp::create(
          fb, loc,
          buildOut(falseVal, /*iterAdv=*/false, /*activeStep=*/-1,
                   /*activeChild=*/-1, /*activeLive=*/stepWaitIdx[i],
                   /*activePost=*/-1));
      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      fsm::TransitionOp::create(
          fb, loc, StringRef(postName),
          [&]() {
            fsm::ReturnOp::create(fb, loc,
                                  machine.getArgument(childDoneArgIdx));
          },
          []() {});
    }
    fb.setInsertionPointToEnd(&machine.getBody().front());

    // POST_i: drive post_active_j; iter_advance if this is the last step.
    {
      auto st = fsm::StateOp::create(fb, loc, postName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      bool iterAdv = isLast;
      fsm::OutputOp::create(
          fb, loc,
          buildOut(falseVal, iterAdv, /*activeStep=*/-1,
                   /*activeChild=*/-1, /*activeLive=*/-1,
                   /*activePost=*/stepWaitIdx[i]));
      Block *tb = &st.getTransitions().front();
      if (isLast) {
        emitLastStepTransition(tb);
      } else {
        fb.setInsertionPointToEnd(tb);
        fsm::TransitionOp::create(fb, loc, StringRef(nextStepName));
      }
    }
    fb.setInsertionPointToEnd(&machine.getBody().front());
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
    ArrayRef<int> stepChildKind) {
  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();
  unsigned numSteps = stepChildKind.size();

  // Count non-leaf steps to determine child_done inputs and child_start outputs.
  unsigned numChildren = 0;
  SmallVector<int> childIndexForStep(numSteps, -1);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepChildKind[i] >= 0) {
      childIndexForStep[i] = numChildren;
      numChildren++;
    }
  }

  // Inputs: start, child_done_0, ..., child_done_{numChildren-1}
  SmallVector<Type> inputTypes(1 + numChildren, i1);
  // Outputs: done, child_start_0..N, step_running_0..M
  SmallVector<Type> outputTypes(1 + numChildren + numSteps, i1);

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
  for (unsigned i = 0; i < numSteps; ++i)
    resNameAttrs.push_back(
        builder.getStringAttr("step_running_" + std::to_string(i)));
  machine.setResNamesAttr(builder.getArrayAttr(resNameAttrs));

  OpBuilder fb(ctx);
  fb.setInsertionPointToEnd(&machine.getBody().front());

  Value trueVal = hw::ConstantOp::create(fb, loc, i1, 1);
  Value falseVal = hw::ConstantOp::create(fb, loc, i1, 0);

  // Helper to build output vector.
  // out[0] = done, out[1..numChildren] = child_start, out[1+numChildren..] = step_running
  auto makeOutput = [&](bool done, int activeChildStart,
                        int activeStepRunning) -> SmallVector<Value> {
    SmallVector<Value> vals;
    vals.push_back(done ? trueVal : falseVal);
    for (unsigned i = 0; i < numChildren; ++i)
      vals.push_back((int)i == activeChildStart ? trueVal : falseVal);
    for (unsigned i = 0; i < numSteps; ++i)
      vals.push_back((int)i == activeStepRunning ? trueVal : falseVal);
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
        fb, loc, StringRef("STEP_0"),
        [&]() { fsm::ReturnOp::create(fb, loc, machine.getArgument(0)); },
        []() {});
  }
  fb.setInsertionPointToEnd(&machine.getBody().front());

  // --- STEP_i and WAIT_i states ---
  for (unsigned i = 0; i < numSteps; ++i) {
    std::string stepName = "STEP_" + std::to_string(i);
    std::string nextState =
        (i + 1 < numSteps) ? "STEP_" + std::to_string(i + 1) : "DONE";
    bool isLeaf = (stepChildKind[i] < 0);

    // STEP_i
    {
      auto st = fsm::StateOp::create(fb, loc, stepName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);

      if (isLeaf) {
        // Leaf: step_running_i = 1, no child_start
        fsm::OutputOp::create(fb, loc, makeOutput(false, -1, i));
      } else {
        // Non-leaf: child_start_j = 1, step_running_i = 1
        fsm::OutputOp::create(fb, loc,
                               makeOutput(false, childIndexForStep[i], i));
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
      // step_running_i stays high during WAIT
      fsm::OutputOp::create(fb, loc, makeOutput(false, -1, i));
      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      // Transition to next on child_done
      unsigned childDoneArgIdx = 1 + childIndexForStep[i];
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
    ArrayRef<MemrefArgInfo> memrefArgs,
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
    Type dataType = memInfo.memType.getElementType();
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_rd_data"), dataType,
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
    appendMemrefOutputPorts(builder, "mem" + std::to_string(i),
                            memInfo.memType, ports);

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
    mp.rdData = hwBody->getArgument(argIdx++);
    mp.addrs.assign(getNumAddrPorts(memInfo.memType), Value());
    mp.wrData = Value();
    mp.wrEn = Value();
    localMemPortMap[memInfo.originalArg] = mp;
  }

  return hwMod;
}

/// Build hw.output for a loop module.
static void buildLoopModuleOutput(
    OpBuilder &builder, Location loc,
    DenseMap<Value, MemPortMapping> &localMemPortMap,
    ArrayRef<MemrefArgInfo> memrefArgs,
    Value doneSignal,
    ArrayRef<Value> resultValues) {

  SmallVector<Value> outputs;

  outputs.push_back(doneSignal);

  for (Value v : resultValues)
    outputs.push_back(v);

  for (auto &memInfo : memrefArgs)
    appendMemrefOutputValues(builder, loc, memInfo.memType,
                             memInfo.originalArg, localMemPortMap, outputs);

  hw::OutputOp::create(builder, loc, outputs);
}

/// Merge per-step memory port mappings into a single mapping using
/// step_running signals. Since steps are mutually exclusive, a priority mux
/// chain selects the active step's ports.
static void mergeStepMemPorts(
    OpBuilder &builder, Location loc,
    ArrayRef<DenseMap<Value, MemPortMapping>> perStepPorts,
    ArrayRef<Value> stepRunningSignals,
    ArrayRef<MemrefArgInfo> memrefArgs,
    DenseMap<Value, MemPortMapping> &mergedPorts) {

  auto *ctx = builder.getContext();
  auto i1 = builder.getI1Type();

  for (auto &memInfo : memrefArgs) {
    auto widths = getDimAddrWidths(memInfo.memType);
    Type dataType = memInfo.memType.getElementType();

    // Start with zero defaults: one per dim, plus wrData/wrEn.
    SmallVector<Value> addrs;
    for (unsigned w : widths) {
      Type addrType = IntegerType::get(ctx, w);
      addrs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
    }
    Value wrData = hw::ConstantOp::create(builder, loc, dataType, 0);
    Value wrEn = hw::ConstantOp::create(builder, loc, i1, 0);

    // Build priority mux chain (last step has lowest priority).
    for (int i = perStepPorts.size() - 1; i >= 0; --i) {
      auto it = perStepPorts[i].find(memInfo.originalArg);
      if (it == perStepPorts[i].end())
        continue;
      auto &ports = it->second;
      for (auto [d, w] : llvm::enumerate(widths)) {
        Type addrType = IntegerType::get(ctx, w);
        Value stepAddr = (d < ports.addrs.size() && ports.addrs[d])
            ? ports.addrs[d]
            : hw::ConstantOp::create(builder, loc, addrType, 0);
        addrs[d] = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                        stepAddr, addrs[d]);
      }
      Value stepWrData = ports.wrData ? ports.wrData
          : hw::ConstantOp::create(builder, loc, dataType, 0);
      Value stepWrEn = ports.wrEn ? ports.wrEn
          : hw::ConstantOp::create(builder, loc, i1, 0);
      wrData = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                    stepWrData, wrData);
      wrEn = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                  stepWrEn, wrEn);
    }

    auto &merged = mergedPorts[memInfo.originalArg];
    merged.addrs = std::move(addrs);
    merged.wrData = wrData;
    merged.wrEn = wrEn;
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
    auto memType = cast<MemRefType>(memref.getType());
    auto widths = getDimAddrWidths(memType);
    Type dataType = memType.getElementType();

    SmallVector<Value> addrs;
    for (unsigned w : widths) {
      Type addrType = IntegerType::get(ctx, w);
      addrs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
    }
    Value wrData = hw::ConstantOp::create(builder, loc, dataType, 0);
    Value wrEn = hw::ConstantOp::create(builder, loc, i1, 0);

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
      if (!hasAddr && !ports.wrData && !ports.wrEn)
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
    }

    auto &out = outPorts[memref];
    out.addrs = std::move(addrs);
    out.wrData = wrData;
    out.wrEn = wrEn;
    // rdData is preserved (it is set by the caller from the module's input
    // port and shared across all stages).
  }
}

//===----------------------------------------------------------------------===//
// Modular loop lowering (one hw.module per sequential loop)
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleToFSMPass::lowerLoopNodeAsModule(
    const LoopNode &node, OpBuilder &builder, Location loc,
    func::FuncOp funcOp, ArrayRef<MemrefArgInfo> memrefArgs,
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

  // Clone referenced constants into the module body.
  OpBuilder hw(ctx);
  hw.setInsertionPointToEnd(hwBody);
  for (auto *constOp : referencedConsts) {
    hw.clone(*constOp, localMapping);
  }

  // --- Collect steps ---
  SmallVector<LoopScheduleStepOp> steps;
  for (auto &op : seqOp.getScheduleBlock().getOperations())
    if (auto stepOp = dyn_cast<LoopScheduleStepOp>(&op))
      steps.push_back(stepOp);

  if (steps.empty())
    return seqOp.emitError("sequential loop has no steps");

  auto terminatorOp =
      cast<LoopScheduleTerminatorOp>(seqOp.getScheduleBlock().getTerminator());

  unsigned numSteps = steps.size();

  // Collect the step indices that need to wait on an external done signal
  // (today: child sequential loops and pipeline children; tomorrow: any
  // variable-latency op). Sorted ascending by construction.
  SmallVector<unsigned> waitStepIndices;
  // For each waitStepIndices entry j, stepWaitIdx maps the loop step index
  // back to j; -1 for "regular" steps.
  SmallVector<int> stepWaitIdx(numSteps, -1);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (node.stepChildIdx[i] >= 0 || node.stepPipelineIdx[i] >= 0) {
      stepWaitIdx[i] = (int)waitStepIndices.size();
      waitStepIndices.push_back(i);
    }
  }
  unsigned numWaits = waitStepIndices.size();

  // --- Compute per-step latencies ---
  // Multi-cycle steps arise when SCFToLoopSchedule's bucket merger coalesces
  // overlapping start times into a single step containing one or more
  // loopschedule.delay regions and/or stamps multi-cycle operator latencies.
  SmallVector<unsigned> stepLatencies(numSteps, 1);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepWaitIdx[i] >= 0) {
      // Wait steps must be 1 (the bucket merger refuses to merge into them).
      stepLatencies[i] = 1;
      continue;
    }
    stepLatencies[i] = computeStepLatency(steps[i]);
  }
  // Per-step base index in the FSM's appended cycle-output region.
  SmallVector<int> stepCycleOutBase(numSteps, -1);
  unsigned totalCycleOuts = 0;
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepLatencies[i] > 1) {
      stepCycleOutBase[i] = (int)totalCycleOuts;
      totalCycleOuts += stepLatencies[i];
    }
  }

  // --- Create FSM machine ---
  std::string fsmName = node.prefix + "_fsm";
  builder.setInsertionPointToEnd(moduleOp.getBody());
  (void)createSequentialFSM(builder, loc, fsmName, numSteps, waitStepIndices,
                            stepLatencies);

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
  //               step_active_0..N-1, child_start_0..C-1,
  //               child_active_0..C-1, post_active_0..C-1,
  //               step_cycle_<i>_<c>...
  unsigned numFsmResults = 3 + numSteps + 3 * numWaits + totalCycleOuts;
  SmallVector<Type> resTys(numFsmResults, i1);
  auto inst = fsm::HWInstanceOp::create(
      hw, loc, resTys, hw.getStringAttr(fsmName + "_inst"),
      hw.getAttr<FlatSymbolRefAttr>(fsmName), instInputs, clk, rst);

  Value fsmDone = inst.getResult(0);
  Value fsmFirstIter = inst.getResult(1);
  Value fsmIterAdvance = inst.getResult(2);
  SmallVector<Value> fsmStepActives(numSteps);
  for (unsigned i = 0; i < numSteps; ++i)
    fsmStepActives[i] = inst.getResult(3 + i);
  SmallVector<Value> fsmChildStarts(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    fsmChildStarts[j] = inst.getResult(3 + numSteps + j);
  SmallVector<Value> fsmChildActives(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    fsmChildActives[j] = inst.getResult(3 + numSteps + numWaits + j);
  SmallVector<Value> fsmPostActives(numWaits);
  for (unsigned j = 0; j < numWaits; ++j)
    fsmPostActives[j] = inst.getResult(3 + numSteps + 2 * numWaits + j);
  // Per-step per-cycle gates. For single-cycle steps the gate vector contains
  // just the step's overall step_active signal; for multi-cycle steps it
  // contains the L_i dedicated step_cycle_<i>_<c> outputs.
  unsigned cycleOutOffset = 3 + numSteps + 3 * numWaits;
  SmallVector<SmallVector<Value>> fsmStepCycleGates(numSteps);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepCycleOutBase[i] < 0) {
      fsmStepCycleGates[i].push_back(fsmStepActives[i]);
    } else {
      for (unsigned c = 0; c < stepLatencies[i]; ++c)
        fsmStepCycleGates[i].push_back(
            inst.getResult(cycleOutOffset + (unsigned)stepCycleOutBase[i] + c));
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

  // --- Per-step capture gates ---
  // Regular (non-wait) steps latch their register-op operands into hardware
  // registers at the end of the step so that later steps read the correctly-
  // held value rather than the combinational expression driven off
  // now-deallocated addresses. For single-cycle regular steps, capture on
  // step_active_<i>. For multi-cycle regular steps, capture on the last
  // sub-cycle gate. Wait steps (child sequential / pipeline) capture nothing:
  // their results come from child modules whose outputs are already
  // registered.
  SmallVector<Value> stepCaptureGate(numSteps);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepWaitIdx[i] >= 0)
      continue;
    stepCaptureGate[i] = fsmStepCycleGates[i].back();
  }

  // Combinational aliases for step-result values, captured BEFORE each step's
  // capture registers rewrite the mapping. Used for wiring the FSM cond input
  // and sequential iter_arg register feedback — both of which must see the
  // current-iteration combinational value, not a one-cycle-lagged register.
  DenseMap<Value, Value> stepResultComb;

  // Per-step memory port mappings. Each starts with the shared `rdData` from
  // the module's input port so that loads always read the correct port. Per-
  // step addrs/wrData/wrEn are merged at the end via a priority mux gated on
  // each step's "alive" signal — the wait-step memory ops (child launch, child
  // active, post-child stores) must all see their step as alive, so wait steps
  // use `child_active_j | post_active_j` as their merge selector instead of
  // `step_active_i` (which is -1 for wait steps).
  SmallVector<DenseMap<Value, MemPortMapping>> perStepPorts(numSteps);
  for (unsigned i = 0; i < numSteps; ++i) {
    for (auto &memInfo : memrefArgs) {
      perStepPorts[i][memInfo.originalArg].rdData =
          localMemPorts[memInfo.originalArg].rdData;
    }
  }

  // Determine which step produces the loop condition so we can resolve the
  // condition backedge after lowering that step.
  Value condTermVal = terminatorOp.getCondition();
  unsigned condStepIdx = 0;
  for (auto [idx, s] : llvm::enumerate(steps)) {
    if (condTermVal.getDefiningOp() == s.getOperation()) {
      condStepIdx = idx;
      break;
    }
  }

  // --- Lower step bodies ---
  for (auto [stepIdx, stepOp] : llvm::enumerate(steps)) {
    Block *body = &stepOp.getBodyBlock();
    hw.setInsertionPointToEnd(hwBody);

    int childIdx = node.stepChildIdx[stepIdx];
    int waitIdx = stepWaitIdx[stepIdx];
    if (childIdx >= 0) {
      // This step contains a child sequential loop.
      auto &childNode = node.children[childIdx];
      auto childSeqOp = childNode.seqOp;
      Value stepChildStart = fsmChildStarts[waitIdx];
      Value stepPostActive = fsmPostActives[waitIdx];

      // Lower pre-child ops.
      for (auto &op : *body) {
        if (&op == childSeqOp.getOperation())
          break;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        if (isa<LoopScheduleSequentialOp>(&op))
          break;

        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          if (failed(handleLoad(loadOp, hw, localMapping,
                                perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
          // Pre-child stores fire only during the child-launch state.
          if (failed(handleStore(storeOp, hw, localMapping, stepChildStart,
                                  perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        hw.clone(op, localMapping);
      }

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
      childInputs.push_back(stepChildStart);
      for (Value cap : childCaptured)
        childInputs.push_back(localMapping.lookup(cap));
      for (auto &memInfo : memrefArgs)
        childInputs.push_back(
            perStepPorts[stepIdx][memInfo.originalArg].rdData);

      auto childInst = hw::InstanceOp::create(
          hw, loc, childModule,
          hw.getStringAttr(childNode.prefix + "_inst"),
          childInputs, nullptr);

      // Extract child outputs: done, results..., (addr, wrData, wrEn) per mem...
      unsigned outIdx = 0;
      Value childDone = childInst.getResult(outIdx++);
      childDoneBEs[waitIdx].setValue(childDone);

      // Map child sequential op results.
      for (auto result : childSeqOp.getResults())
        localMapping.map(result, childInst.getResult(outIdx++));

      // Mux child memory ports into this step's per-step port mapping. Any
      // pre-child op may have already written a pre-child address into
      // perStepPorts[stepIdx]; the child's outputs take priority when
      // child_active_j is high so only the launch cycle's pre-child writes
      // actually fire while the child is running.
      Value childActive = fsmChildActives[waitIdx];
      for (auto &memInfo : memrefArgs) {
        auto widths = getDimAddrWidths(memInfo.memType);
        SmallVector<Value> childAddrs;
        for (unsigned d = 0; d < widths.size(); ++d)
          childAddrs.push_back(childInst.getResult(outIdx++));
        Value childWrData = childInst.getResult(outIdx++);
        Value childWrEn = childInst.getResult(outIdx++);

        auto &ports = perStepPorts[stepIdx][memInfo.originalArg];
        if (ports.addrs.size() != widths.size())
          ports.addrs.assign(widths.size(), Value());
        Type dataType = memInfo.memType.getElementType();
        for (auto [d, w] : llvm::enumerate(widths)) {
          Type addrType = IntegerType::get(ctx, w);
          Value myAddr = ports.addrs[d] ? ports.addrs[d]
              : hw::ConstantOp::create(hw, loc, addrType, 0);
          ports.addrs[d] = comb::MuxOp::create(hw, loc, childActive,
                                                childAddrs[d], myAddr);
        }
        Value myWrData = ports.wrData ? ports.wrData
            : hw::ConstantOp::create(hw, loc, dataType, 0);
        Value myWrEn = ports.wrEn ? ports.wrEn
            : hw::ConstantOp::create(hw, loc, i1, 0);
        ports.wrData =
            comb::MuxOp::create(hw, loc, childActive, childWrData, myWrData);
        ports.wrEn =
            comb::MuxOp::create(hw, loc, childActive, childWrEn, myWrEn);
      }

      // Lower post-child ops with wrEnGate = postActive.
      hw.setInsertionPointToEnd(hwBody);
      bool pastChild = false;
      for (auto &op : *body) {
        if (&op == childSeqOp.getOperation()) {
          pastChild = true;
          continue;
        }
        if (!pastChild)
          continue;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;

        if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
          if (failed(handleStore(storeOp, hw, localMapping, stepPostActive,
                                  perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          if (failed(handleLoad(loadOp, hw, localMapping,
                                perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        hw.clone(op, localMapping);
      }
    } else if (node.stepPipelineIdx[stepIdx] >= 0) {
      // This step contains a pipeline child.
      int pipIdx = node.stepPipelineIdx[stepIdx];
      auto pipOp = node.pipelineChildren[pipIdx];
      Value stepChildStart = fsmChildStarts[waitIdx];
      Value stepPostActive = fsmPostActives[waitIdx];

      // Lower pre-pipeline ops.
      for (auto &op : *body) {
        if (&op == pipOp.getOperation())
          break;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        if (isa<LoopSchedulePipelineOp>(&op))
          break;

        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          if (failed(handleLoad(loadOp, hw, localMapping,
                                perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
          // Pre-pipeline stores fire only during the pipeline-launch state.
          if (failed(handleStore(storeOp, hw, localMapping, stepChildStart,
                                  perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        hw.clone(op, localMapping);
      }

      // Lower pipeline child (inline in this module).
      hw.setInsertionPointToEnd(hwBody);
      Value pipDone;
      std::string pipPrefix = node.prefix + "_pip" + std::to_string(pipIdx);
      if (failed(lowerPipelineChild(pipOp, hw, loc, hwBody, localMapping, clk,
                                    rst, stepChildStart, pipPrefix, pipDone,
                                    perStepPorts[stepIdx], memrefArgs)))
        return failure();
      childDoneBEs[waitIdx].setValue(pipDone);

      // Lower post-pipeline ops with wrEnGate = postActive.
      hw.setInsertionPointToEnd(hwBody);
      bool pastChild = false;
      for (auto &op : *body) {
        if (&op == pipOp.getOperation()) {
          pastChild = true;
          continue;
        }
        if (!pastChild)
          continue;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;

        if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
          if (failed(handleStore(storeOp, hw, localMapping, stepPostActive,
                                  perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          if (failed(handleLoad(loadOp, hw, localMapping,
                                perStepPorts[stepIdx])))
            return failure();
          continue;
        }

        hw.clone(op, localMapping);
      }
    } else {
      // Regular step (no child). Gate stores per issue cycle: ops directly
      // in the step body get cycleGates[0]; ops nested inside delay regions
      // get cycleGates[delay.latency].
      if (failed(lowerStepBody(body, hw, localMapping,
                               fsmStepCycleGates[stepIdx],
                               perStepPorts[stepIdx])))
        return failure();
    }

    // Detect whether this step contains any loads from local hlmem memories.
    // Local hlmem read ports have latency=1: `rd_data` during cycle N reflects
    // `rd_addr` from cycle N-1. So during the cycle where step_active_<i> is
    // high, rd_data is stale (reflecting the prior state's addr). We must
    // capture rd_data ONE CYCLE LATER, when it correctly reflects the addr
    // driven during the step. For steps without local loads, the normal gate
    // works fine (register/combinational ops settle within their cycle).
    bool stepHasLocalLoad = false;
    for (auto &op : *body) {
      if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
        for (auto &memInfo : memrefArgs) {
          if (memInfo.originalArg == loadOp.getMemRef() && memInfo.isLocalMem) {
            stepHasLocalLoad = true;
            break;
          }
        }
        if (stepHasLocalLoad)
          break;
      }
    }

    // Map step results. First save the combinational aliases (keyed on the
    // stepOp result value), then — if this step has a capture gate — create
    // per-operand hardware registers and point the mapping at those so
    // subsequent step bodies read the held value instead of a combinational
    // expression driven off now-deallocated memory addresses. Wait steps
    // (childIdx/pipelineIdx >= 0) skip the capture: their register op operands
    // come from the child instance and are already registered.
    auto regOp = cast<LoopScheduleRegisterOp>(body->getTerminator());
    for (auto [result, regVal] :
         llvm::zip(stepOp.getResults(), regOp.getOperands())) {
      Value combVal = localMapping.lookup(regVal);
      stepResultComb[result] = combVal;
      localMapping.map(result, combVal);
    }

    if (stepCaptureGate[stepIdx]) {
      hw.setInsertionPointToEnd(hwBody);
      // If this step contains a local hlmem load, delay the capture gate by
      // one cycle so we latch rd_data when it's valid (the cycle after the
      // address is applied), not when it's still stale.
      Value captureGate = stepCaptureGate[stepIdx];
      if (stepHasLocalLoad) {
        Value falseConstCap =
            hw::ConstantOp::create(hw, loc, hw.getI1Type(), 0);
        captureGate = seq::CompRegOp::create(
            hw, loc, stepCaptureGate[stepIdx], clk, rst, falseConstCap,
            hw.getStringAttr(node.prefix + "_step" + std::to_string(stepIdx) +
                             "_capture_gate_delayed"));
      }
      for (auto [idx, it] : llvm::enumerate(llvm::zip(
               stepOp.getResults(), regOp.getOperands()))) {
        auto [result, regVal] = it;
        Value combVal = stepResultComb[result];
        Value resetVal = createZeroConstant(hw, loc, combVal.getType());
        auto regName = hw.getStringAttr(node.prefix + "_step" +
                                        std::to_string(stepIdx) + "_r" +
                                        std::to_string(idx));
        Value reg = seq::CompRegClockEnabledOp::create(
            hw, loc, combVal, clk, captureGate, rst, resetVal, regName);
        localMapping.map(result, reg);
      }
    }

    // After lowering the step that produces the loop condition, wire it into
    // the FSM instance via the condition backedge. Prefer the combinational
    // alias (stepResultComb) over the mapping, which may now point at the
    // post-step capture register.
    if (stepIdx == condStepIdx) {
      auto it = stepResultComb.find(condTermVal);
      condBE.setValue(it != stepResultComb.end()
                          ? it->second
                          : localMapping.lookup(condTermVal));
    }
  }

  // --- Wire up iter_arg feedback ---
  // iter_arg registers clock on iter_advance (the last step's last cycle)
  // which is the SAME posedge the step-capture registers update on. The
  // captured register sees its own pre-edge value on that posedge, which is
  // stale by one iteration; always prefer the combinational alias for the
  // iter_arg D input.
  hw.setInsertionPointToEnd(hwBody);
  for (unsigned i = 0; i < iterArgRegs.size(); ++i) {
    Value termArg = terminatorOp.getIterArgs()[i];
    auto combIt = stepResultComb.find(termArg);
    Value feedback = combIt != stepResultComb.end()
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
    for (unsigned j = 0; j < terminatorOp.getIterArgs().size(); ++j) {
      if (termResult == terminatorOp.getIterArgs()[j]) {
        resultValues.push_back(iterArgRegs[j]);
        mapped = true;
        break;
      }
    }
    if (!mapped)
      resultValues.push_back(localMapping.lookup(termResult));
  }

  // --- Merge per-step memory ports ---
  // Build per-step "alive" signals used by the merge priority mux. Regular
  // steps use their step_active_<i> (high in every sub-state of STEP_<i>).
  // Wait steps must cover STEP+WAIT+POST, and since `step_active_<i>` is
  // never driven high for wait steps, we OR together child_active_<j> (which
  // spans STEP and WAIT for that child) and post_active_<j> (POST state).
  hw.setInsertionPointToEnd(hwBody);
  SmallVector<Value> stepMergeSignals(numSteps);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepWaitIdx[i] >= 0) {
      unsigned j = (unsigned)stepWaitIdx[i];
      stepMergeSignals[i] = comb::OrOp::create(
          hw, loc, fsmChildActives[j], fsmPostActives[j]);
    } else {
      stepMergeSignals[i] = fsmStepActives[i];
    }
  }

  DenseMap<Value, MemPortMapping> mergedMemPorts;
  for (auto &memInfo : memrefArgs)
    mergedMemPorts[memInfo.originalArg].rdData =
        localMemPorts[memInfo.originalArg].rdData;
  mergeStepMemPorts(hw, loc, perStepPorts, stepMergeSignals, memrefArgs,
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

LogicalResult LoopScheduleToFSMPass::lowerPipelineChild(
    LoopSchedulePipelineOp pipOp, OpBuilder &builder, Location loc,
    Block *hwBody, IRMapping &mapping, Value clk, Value rst,
    Value startSignal, StringRef namePrefix, Value &doneSignal,
    DenseMap<Value, MemPortMapping> &memPorts,
    ArrayRef<MemrefArgInfo> memrefArgs) {

  auto *ctx = builder.getContext();

  SmallVector<LoopSchedulePipelineStageOp> stages;
  for (auto &op : pipOp.getStagesBlock().getOperations())
    if (auto stageOp = dyn_cast<LoopSchedulePipelineStageOp>(&op))
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

  Backedge activeNextBE = bb.get(hwBuilder.getI1Type());
  auto activeReg = seq::CompRegOp::create(
      hwBuilder, loc, Value(activeNextBE), clk, rst, falseConst,
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
    auto counterReg = seq::CompRegOp::create(
        hwBuilder, loc, Value(counterBackedge), clk, rst, cZero,
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

  SmallVector<Value> stageCE(stages.size());
  stageCE[0] = activeCE;
  for (unsigned i = 1; i < stages.size(); ++i) {
    auto ceName = hwBuilder.getStringAttr(
        (namePrefix + "_ce_stage_" + std::to_string(i)).str());
    stageCE[i] = seq::CompRegOp::create(hwBuilder, loc, stageCE[i - 1], clk,
                                        rst, falseConst, ceName);
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

  // Delay chains for cross-stage values. If a value V is produced at stage J
  // and consumed at stage K > J+1, we must insert (K - J - 1) delay registers
  // so the consumer sees the correct pipeline iteration rather than whatever
  // value currently occupies stage J's register (which is overwritten each
  // cycle in an II=1 pipeline). chain[0] is the direct stage J register;
  // chain[k] (k >= 1) is a delay register clocked on stageCE[J + k]. Consumer
  // stage K reads chain[K - J - 1].
  DenseMap<Value, SmallVector<Value>> delayChain;

  auto resolveForStage = [&](Value origVal,
                             unsigned consumerStage) -> Value {
    auto result = dyn_cast<OpResult>(origVal);
    if (!result)
      return Value();
    auto defStage =
        dyn_cast<LoopSchedulePipelineStageOp>(result.getOwner());
    if (!defStage)
      return Value();
    unsigned J = stages.size();
    for (unsigned s = 0; s < stages.size(); ++s)
      if (stages[s] == defStage) {
        J = s;
        break;
      }
    // Only delay values whose consumer is strictly later than the
    // stage immediately after the producer. Adjacent stages (K == J+1)
    // already read the directly-mapped stage J register correctly.
    if (J == stages.size() || consumerStage <= J + 1)
      return Value();
    auto &chain = delayChain[origVal];
    if (chain.empty())
      chain.push_back(mapping.lookup(origVal));
    while (chain.size() < consumerStage - J) {
      unsigned ceStage = J + chain.size();
      Value prev = chain.back();
      Value resetVal = createZeroConstant(hwBuilder, loc, prev.getType());
      auto regName = hwBuilder.getStringAttr(
          (namePrefix + "_s" + std::to_string(ceStage) + "_dly_r" +
           std::to_string(result.getResultNumber()) + "_from_s" +
           std::to_string(J))
              .str());
      OpBuilder::InsertionGuard g(hwBuilder);
      hwBuilder.setInsertionPointToEnd(hwBody);
      Value reg = seq::CompRegClockEnabledOp::create(
          hwBuilder, loc, prev, clk, stageCE[ceStage], rst, resetVal,
          regName);
      chain.push_back(reg);
    }
    return chain[consumerStage - J - 1];
  };

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
      if (isa<LoopScheduleRegisterOp>(&op))
        continue;

      // Temporarily override the mapping entries for any operands that come
      // from earlier stages so the clone sees the correctly-delayed version.
      // Restore the original mappings after lowering this op so later uses
      // aren't affected.
      SmallVector<std::pair<Value, Value>> savedMappings;
      for (Value operand : op.getOperands()) {
        Value delayed = resolveForStage(operand, stageIdx);
        if (!delayed)
          continue;
        Value current = mapping.lookup(operand);
        if (delayed != current) {
          savedMappings.emplace_back(operand, current);
          mapping.map(operand, delayed);
        }
      }

      LogicalResult opResult = success();
      if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
        opResult = handleStore(storeOp, hwBuilder, mapping,
                               stageCE[stageIdx], perStagePorts[stageIdx]);
      } else if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
        if (isLocalMemref(loadOp.getMemRef()))
          localLoadResults.insert(loadOp.getResult());
        opResult =
            handleLoad(loadOp, hwBuilder, mapping, perStagePorts[stageIdx]);
      } else {
        hwBuilder.clone(op, mapping);
      }

      for (auto &sm : savedMappings)
        mapping.map(sm.first, sm.second);

      if (failed(opResult))
        return failure();
    }

    auto regOp = cast<LoopScheduleRegisterOp>(body.getTerminator());

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
      Value delayed = resolveForStage(val, stageIdx);
      Value mappedVal = delayed ? delayed : mapping.lookup(val);

      if (localLoadResults.count(val)) {
        // Local hlmem read (latency=1) already provides the 1-cycle delay.
        // Map the stage result directly to rdData — no extra register.
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

  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value init = mapping.lookup(pipOp.getInits()[i]);
    Value feedback = mapping.lookup(terminatorOp.getIterArgs()[i]);
    // The feedback value is produced at some stage; first_iter must stay
    // high until that stage has fired at least once. If the feedback isn't
    // a stage result (shouldn't happen for well-formed pipelines), fall
    // back to stage 0.
    unsigned feedbackStage = 0;
    Value termVal = terminatorOp.getIterArgs()[i];
    if (auto opResult = dyn_cast<OpResult>(termVal)) {
      if (auto stage = dyn_cast<LoopSchedulePipelineStageOp>(
              opResult.getOwner())) {
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

  return success();
}

//===----------------------------------------------------------------------===//
// Helper: build hw.module ports and create hw.module
//===----------------------------------------------------------------------===//

/// Shared hw.module creation logic used by both sequential and pipeline paths.
static hw::HWModuleOp createHWModule(
    func::FuncOp funcOp, OpBuilder &builder, IRMapping &mapping,
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
      inputIdx++;
      ports.push_back({{builder.getStringAttr(baseName + "_rd_data"), dataType,
                         hw::ModulePort::Direction::Input}});
      appendMemrefOutputPorts(builder, baseName, memType, ports);
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

  ports.push_back({{builder.getStringAttr("done"), builder.getI1Type(),
                     hw::ModulePort::Direction::Output}});
  
  for (auto [idx, retType] : llvm::enumerate(funcOp.getResultTypes())) {
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
      mp.rdData = hwBody->getArgument(hwArgIdx++);
      mp.addrs.assign(getNumAddrPorts(memTy), Value());
      mp.wrData = Value();
      mp.wrEn = Value();
      memPortMap[arg] = mp;
    } else {
      mapping.map(arg, hwBody->getArgument(hwArgIdx));
      hwArgIdx++;
    }
  }

  return hwMod;
}

/// Build hw.output with memory ports, done signal, and return values.
static void buildHWOutput(func::FuncOp funcOp, OpBuilder &builder,
                          Location loc, Block *hwBody, IRMapping &mapping,
                          DenseMap<Value, MemPortMapping> &memPortMap,
                          Value doneSignal) {
  SmallVector<Value> outputs;

  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    auto memType = dyn_cast<MemRefType>(arg.getType());
    if (!memType)
      continue;
    appendMemrefOutputValues(builder, loc, memType, arg, memPortMap, outputs);
  }

  outputs.push_back(doneSignal);

  auto returnOp =
      cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
  for (auto retVal : returnOp.getOperands()) {
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

LogicalResult LoopScheduleToFSMPass::lowerFunction(func::FuncOp funcOp) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();
  OpBuilder builder(ctx);

  // --- Sequential path (supports nesting) ---

  IRMapping mapping;
  unsigned clkIdx, rstIdx, startIdx;
  auto hwMod = createHWModule(funcOp, builder, mapping, memPortMap, clkIdx,
                              rstIdx, startIdx);
  Block *hwBody = hwMod.getBodyBlock();
  Value clk = hwBody->getArgument(clkIdx);
  Value rst = hwBody->getArgument(rstIdx);
  Value start = hwBody->getArgument(startIdx);

  builder.setInsertionPointToEnd(hwBody);

  // Clone non-loopschedule ops (constants, etc.), skipping allocs.
  SmallVector<memref::AllocOp> localAllocs;
  for (auto &op : funcOp.getBody().front()) {
    if (isa<func::ReturnOp>(&op))
      continue;
    if (isa<LoopScheduleStepOp, LoopScheduleSequentialOp,
            LoopSchedulePipelineOp>(&op))
      continue;
    if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
      localAllocs.push_back(allocOp);
      continue;
    }
    builder.clone(op, mapping);
  }

  // Create seq.hlmem for each memref.alloc.
  auto i1 = builder.getI1Type();
  BackedgeBuilder funcBB(builder, loc);
  SmallVector<HLMemBackedges> hlmemBEs;
  for (auto [i, allocOp] : llvm::enumerate(localAllocs)) {
    auto memType = allocOp.getType();
    auto dataType = memType.getElementType();

    std::string name = "local_mem" + std::to_string(i);
    auto hlmem = seq::HLMemOp::create(builder, loc, clk, rst, name,
                                        memType.getShape(), dataType);

    // Per-dim address backedges using hlmem's exact address types.
    SmallVector<Backedge> addrBEs;
    SmallVector<Value> addrVals;
    for (Type addrTy : hlmem.getHandle().getType().getAddressTypes()) {
      Backedge be = funcBB.get(addrTy);
      addrBEs.push_back(be);
      addrVals.push_back(Value(be));
    }
    Backedge wrDataBE = funcBB.get(dataType);
    Backedge wrEnBE = funcBB.get(i1);

    // Read port: rdEn = NOT wrEn (read when not writing).
    Value notWrEn = comb::XorOp::create(
        builder, loc, Value(wrEnBE),
        hw::ConstantOp::create(builder, loc, i1, 1));
    auto readPort = seq::ReadPortOp::create(
        builder, loc, hlmem.getHandle(),
        ValueRange(addrVals), notWrEn, /*latency=*/1);

    // Write port.
    seq::WritePortOp::create(
        builder, loc, hlmem.getHandle(),
        ValueRange(addrVals), Value(wrDataBE), Value(wrEnBE),
        /*latency=*/1);

    // Add to memPortMap.
    MemPortMapping mp;
    mp.rdData = readPort.getReadData();
    mp.addrs.assign(getNumAddrPorts(memType), Value());
    mp.wrData = Value();
    mp.wrEn = Value();
    memPortMap[allocOp.getResult()] = mp;

    hlmemBEs.push_back({allocOp.getResult(), std::move(addrBEs), wrDataBE,
                        wrEnBE});
  }

  // Collect top-level steps.
  SmallVector<LoopScheduleStepOp> topSteps;
  for (auto &op : funcOp.getBody().front())
    if (auto stepOp = dyn_cast<LoopScheduleStepOp>(&op))
      topSteps.push_back(stepOp);

  // Collect memref argument info for threading through modules.
  SmallVector<MemrefArgInfo> memrefArgs;
  for (auto arg : funcOp.getArguments()) {
    if (auto memType = dyn_cast<MemRefType>(arg.getType()))
      memrefArgs.push_back({arg, memType});
  }
  // Add local allocs to memrefArgs so they're threaded through child modules.
  for (auto allocOp : localAllocs) {
    memrefArgs.push_back({allocOp.getResult(), allocOp.getType(),
                          /*isLocalMem=*/true});
  }

  // --- Multi-step path ---
  if (topSteps.empty())
    return funcOp.emitError("no top-level steps found");

  unsigned numSteps = topSteps.size();

  // Build a flat list of children: one entry per child op across all steps.
  // A step with two pipelines produces two entries sharing the same stepIdx.
  // A leaf step (no seq/pipeline children) produces one entry with kind=-1.
  struct StepChild {
    unsigned stepIdx;
    int kind; // -1 leaf, 0 sequential, 1 pipeline
    LoopScheduleSequentialOp seqOp;
    LoopSchedulePipelineOp pipOp;
  };
  SmallVector<StepChild> entries;
  for (unsigned i = 0; i < numSteps; ++i) {
    bool any = false;
    for (auto &innerOp : topSteps[i].getBodyBlock()) {
      if (auto seqOp = dyn_cast<LoopScheduleSequentialOp>(&innerOp)) {
        entries.push_back({i, 0, seqOp, {}});
        any = true;
      } else if (auto pipOp = dyn_cast<LoopSchedulePipelineOp>(&innerOp)) {
        entries.push_back({i, 1, {}, pipOp});
        any = true;
      }
    }
    if (!any)
      entries.push_back({i, -1, {}, {}});
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

  // Track which entries belong to each step for pre/post-child op cloning.
  // firstEntryForStep[s] = index of first entry belonging to step s.
  // lastEntryForStep[s]  = index of last entry belonging to step s.
  SmallVector<unsigned> firstEntryForStep(numSteps, 0);
  SmallVector<unsigned> lastEntryForStep(numSteps, 0);
  for (unsigned i = 0; i < numEntries; ++i) {
    unsigned s = entries[i].stepIdx;
    if (i == 0 || entries[i - 1].stepIdx != s)
      firstEntryForStep[s] = i;
    lastEntryForStep[s] = i;
  }

  // Lower each entry.
  unsigned loopCounter = 0;
  for (unsigned ei = 0; ei < numEntries; ++ei) {
    auto &entry = entries[ei];
    unsigned stepIdx = entry.stepIdx;
    builder.setInsertionPointToEnd(hwBody);

    // Clone pre-child ops only for the first entry of this step.
    if (entry.kind >= 0 && ei == firstEntryForStep[stepIdx]) {
      Operation *firstChildOp = entry.seqOp ? entry.seqOp.getOperation()
                                            : entry.pipOp.getOperation();
      for (auto &op : topSteps[stepIdx].getBodyBlock().getOperations()) {
        if (&op == firstChildOp)
          break;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(&op))
          break;
        builder.clone(op, mapping);
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
      for (auto &memInfo : memrefArgs)
        childInputs.push_back(memPortMap[memInfo.originalArg].rdData);

      auto childInst = hw::InstanceOp::create(
          builder, loc, childModule,
          builder.getStringAttr(prefix + "_inst"),
          childInputs, nullptr);

      // Extract child outputs.
      unsigned outIdx = 0;
      Value childDone = childInst.getResult(outIdx++);
      childDoneBEs[childIndexForEntry[ei]].setValue(childDone);

      // Map seqOp results.
      for (auto result : seqOp.getResults())
        mapping.map(result, childInst.getResult(outIdx++));

      // Extract child memory outputs into per-entry ports.
      for (auto &memInfo : memrefArgs) {
        auto &mp = perEntryPorts[ei][memInfo.originalArg];
        auto widths = getDimAddrWidths(memInfo.memType);
        mp.addrs.assign(widths.size(), Value());
        for (unsigned d = 0; d < widths.size(); ++d)
          mp.addrs[d] = childInst.getResult(outIdx++);
        mp.wrData = childInst.getResult(outIdx++);
        mp.wrEn = childInst.getResult(outIdx++);
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

    } else {
      // Leaf step: lower body with entry_running as wrEn gate.
      Value oneGate = entryRunningSignals[ei];
      if (failed(lowerStepBody(&topSteps[stepIdx].getBodyBlock(), builder,
                               mapping, ArrayRef<Value>(oneGate),
                               perEntryPorts[ei])))
        return failure();
    }

    // Clone post-child ops only for the last entry of this step.
    if (entry.kind >= 0 && ei == lastEntryForStep[stepIdx]) {
      Operation *lastChildOp = entry.seqOp ? entry.seqOp.getOperation()
                                           : entry.pipOp.getOperation();
      builder.setInsertionPointToEnd(hwBody);
      bool pastChild = false;
      for (auto &op : topSteps[stepIdx].getBodyBlock().getOperations()) {
        if (&op == lastChildOp) {
          pastChild = true;
          continue;
        }
        if (!pastChild)
          continue;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(&op))
          continue;
        builder.clone(op, mapping);
      }
    }

    // Register step results only after the last entry for this step completes.
    if (ei == lastEntryForStep[stepIdx]) {
      // Pre-scan for loads from local hlmem — skip capture for those.
      DenseSet<Value> localLoadResults;
      for (auto &op : topSteps[stepIdx].getBodyBlock()) {
        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          for (auto &memInfo : memrefArgs)
            if (memInfo.originalArg == loadOp.getMemRef() &&
                memInfo.isLocalMem)
              localLoadResults.insert(loadOp.getResult());
        }
      }

      auto stepRegOp = cast<LoopScheduleRegisterOp>(
          topSteps[stepIdx].getBodyBlock().getTerminator());
      for (auto [stepResult, regOperand] :
           llvm::zip(topSteps[stepIdx].getResults(),
                     stepRegOp.getOperands())) {
        Value val = mapping.lookup(regOperand);
        if (localLoadResults.count(regOperand)) {
          // Local hlmem read (latency=1) provides the delay; skip register.
          mapping.map(stepResult, val);
          continue;
        }
        auto regName = builder.getStringAttr(
            funcName + "_step" + std::to_string(stepIdx) + "_result_" +
            std::to_string(stepResult.getResultNumber()));
        Value resetVal = createZeroConstant(builder, loc, val.getType());
        auto reg = seq::CompRegClockEnabledOp::create(
            builder, loc, val, clk, entryRunningSignals[ei], rst, resetVal,
            regName);
        mapping.map(stepResult, reg);
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

  // Copy merged ports to function-level memPortMap.
  for (auto &memInfo : memrefArgs) {
    memPortMap[memInfo.originalArg].addrs =
        mergedMemPorts[memInfo.originalArg].addrs;
    memPortMap[memInfo.originalArg].wrData =
        mergedMemPorts[memInfo.originalArg].wrData;
    memPortMap[memInfo.originalArg].wrEn =
        mergedMemPorts[memInfo.originalArg].wrEn;
  }

  // Resolve local hlmem backedges.
  builder.setInsertionPointToEnd(hwBody);
  for (auto &be : hlmemBEs) {
    auto &ports = memPortMap[be.allocResult];
    auto memTy = cast<MemRefType>(be.allocResult.getType());
    auto dataType = memTy.getElementType();
    for (auto [d, beAddr] : llvm::enumerate(be.addrBEs)) {
      auto hlmemAddrTy = cast<IntegerType>(Value(beAddr).getType());
      Value src;
      if (d < ports.addrs.size() && ports.addrs[d])
        src = resizeIntTo(builder, loc, ports.addrs[d],
                          hlmemAddrTy.getWidth());
      else
        src = hw::ConstantOp::create(builder, loc, hlmemAddrTy, 0);
      beAddr.setValue(src);
    }
    be.wrDataBE.setValue(ports.wrData ? ports.wrData
        : hw::ConstantOp::create(builder, loc, dataType, 0));
    be.wrEnBE.setValue(ports.wrEn ? ports.wrEn
        : hw::ConstantOp::create(builder, loc, i1, 0));
  }

  // Build hw.output.
  builder.setInsertionPointToEnd(hwBody);
  buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, doneSignal);

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
static void hoistReturnedAllocsToArgs(func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return;
  auto *block = &funcOp.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(block->getTerminator());
  if (!returnOp)
    return;
  // Collect returned memref allocs and their operand indices.
  SmallVector<memref::AllocOp> hoisted;
  SmallVector<unsigned> droppedReturnIdxs;
  for (auto [idx, retVal] : llvm::enumerate(returnOp.getOperands())) {
    if (!isa<MemRefType>(retVal.getType()))
      continue;
    auto allocOp = retVal.getDefiningOp<memref::AllocOp>();
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
  auto newReturn = func::ReturnOp::create(b, returnOp.getLoc(),
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
static LogicalResult flattenMultiDimMemrefs(func::FuncOp funcOp) {
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

  SmallVector<func::FuncOp> funcs;
  moduleOp.walk([&](func::FuncOp f) { funcs.push_back(f); });

  for (auto funcOp : funcs)
    hoistReturnedAllocsToArgs(funcOp);

  for (auto funcOp : funcs) {
    if (failed(flattenMultiDimMemrefs(funcOp))) {
      signalPassFailure();
      return;
    }
  }

  for (auto funcOp : funcs) {
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
