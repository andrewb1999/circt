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

/// Compute the number of bits needed to address a memref.
static unsigned getAddrWidth(MemRefType memType) {
  int64_t numElements = 1;
  for (int64_t dim : memType.getShape())
    numElements *= dim;
  if (numElements <= 1)
    return 1;
  return llvm::Log2_64_Ceil(numElements);
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
struct MemPortMapping {
  Value rdData;  // input: read data from memory
  Value addr;    // will be set: address output
  Value wrData;  // will be set: write data output
  Value wrEn;    // will be set: write enable output
};

/// Information about a memref argument for threading through loop modules.
struct MemrefArgInfo {
  Value originalArg; // original func::FuncOp argument
  MemRefType memType;
};

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
                                   DenseMap<Value, MemPortMapping> &memPorts);

  /// Pipeline support for legacy (top-level pipeline) path.
  LogicalResult lowerPipeline(LoopSchedulePipelineOp pipOp,
                              OpBuilder &builder, Location loc,
                              fsm::MachineOp machine, Block *hwBody,
                              IRMapping &mapping, Value clk, Value rst,
                              Value start, StringRef namePrefix,
                              Value &pipelineDone);

  /// Lower a function containing pipeline ops (old path).
  LogicalResult lowerPipelineFunction(func::FuncOp funcOp);

  /// Map from original func memref args to their hw.module port values.
  DenseMap<Value, MemPortMapping> memPortMap;

  /// Backedge info for a local hlmem (memref.alloc).
  struct HLMemBackedges {
    Value allocResult; // original memref.alloc result
    Backedge addrBE;   // shared address for read and write
    Backedge wrDataBE;
    Backedge wrEnBE;
  };
};

//===----------------------------------------------------------------------===//
// Load/store helpers
//===----------------------------------------------------------------------===//

/// Look up a memref's port mapping and compute the address (extending or
/// extracting to the memref's address width). Returns the resolved port
/// mapping by reference.
static LogicalResult
prepareMemAccess(Operation *op, Value memref, Value indexVal,
                 OpBuilder &builder, IRMapping &mapping,
                 DenseMap<Value, MemPortMapping> &memPorts,
                 MemPortMapping *&portsOut, Value &addrOut) {
  auto it = memPorts.find(memref);
  if (it == memPorts.end())
    return op->emitError("unmapped memref");
  portsOut = &it->second;
  auto memType = cast<MemRefType>(memref.getType());
  unsigned addrWidth = getAddrWidth(memType);
  Value addr = mapping.lookup(indexVal);
  if (cast<IntegerType>(addr.getType()).getWidth() != addrWidth) {
    auto addrType = IntegerType::get(builder.getContext(), addrWidth);
    addr = comb::ExtractOp::create(builder, op->getLoc(), addrType, addr, 0);
  }
  addrOut = addr;
  return success();
}

/// Lower a loopschedule.load: set the read address on the memory port
/// mapping and map the load result to the port's read data.
static LogicalResult
handleLoad(LoopScheduleLoadOp loadOp, OpBuilder &builder, IRMapping &mapping,
           DenseMap<Value, MemPortMapping> &memPorts) {
  if (loadOp.getIndices().size() != 1)
    return loadOp.emitError("multi-dimensional memref not yet supported");
  MemPortMapping *ports = nullptr;
  Value addr;
  if (failed(prepareMemAccess(loadOp, loadOp.getMemRef(),
                              loadOp.getIndices()[0], builder, mapping,
                              memPorts, ports, addr)))
    return failure();
  ports->addr = addr;
  mapping.map(loadOp.getResult(), ports->rdData);
  return success();
}

/// Lower a loopschedule.store: set address, write data, and write enable
/// (gated by wrEnGate) on the memory port mapping.
static LogicalResult
handleStore(LoopScheduleStoreOp storeOp, OpBuilder &builder,
            IRMapping &mapping, Value wrEnGate,
            DenseMap<Value, MemPortMapping> &memPorts) {
  if (storeOp.getIndices().size() != 1)
    return storeOp.emitError("multi-dimensional memref not yet supported");
  assert(wrEnGate && "handleStore requires a wrEnGate");
  MemPortMapping *ports = nullptr;
  Value addr;
  if (failed(prepareMemAccess(storeOp, storeOp.getMemRef(),
                              storeOp.getIndices()[0], builder, mapping,
                              memPorts, ports, addr)))
    return failure();
  ports->addr = addr;
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

  // Determine which state's exit advances iter_arg: the FSM's last cycle
  // before transitioning back to COND. That's POST_<last> if the last step
  // is a wait, otherwise STEP_<last>.
  bool lastIsWait = numSteps > 0 && stepWaitIdx[numSteps - 1] >= 0;

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

  // --- STEP_i (and WAIT_i / POST_i for wait steps) ---
  for (unsigned i = 0; i < numSteps; ++i) {
    bool isWait = stepWaitIdx[i] >= 0;
    bool isLast = (i + 1 == numSteps);
    std::string stepName = "STEP_" + std::to_string(i);
    std::string nextStepName = isLast ? "COND" : "STEP_" + std::to_string(i + 1);
    unsigned L = stepLats[i];

    if (isWait || L == 1) {
      // Single-cycle path (legacy shape — keep STEP_<i> name).
      auto st = fsm::StateOp::create(fb, loc, stepName);
      Block *ob = st.ensureOutput(fb);
      ob->getTerminator()->erase();
      fb.setInsertionPointToEnd(ob);
      if (isWait) {
        // Launch state: drive child_start_j AND child_active_j (the latter
        // also stays high through WAIT_i so external mem-port muxes can
        // forward this child's outputs). iter_advance is handled in POST_i.
        fsm::OutputOp::create(
            fb, loc,
            buildOut(falseVal, /*iterAdv=*/false, /*activeStep=*/-1,
                     /*activeChild=*/stepWaitIdx[i],
                     /*activeLive=*/stepWaitIdx[i], /*activePost=*/-1));
      } else {
        // Regular single-cycle step. iter_advance only when this is the
        // last state of the trip.
        bool iterAdv = isLast;
        fsm::OutputOp::create(
            fb, loc,
            buildOut(falseVal, iterAdv, /*activeStep=*/(int)i,
                     /*activeChild=*/-1, /*activeLive=*/-1,
                     /*activePost=*/-1));
      }
      Block *tb = &st.getTransitions().front();
      fb.setInsertionPointToEnd(tb);
      if (isWait) {
        std::string waitName = "WAIT_" + std::to_string(i);
        fsm::TransitionOp::create(fb, loc, StringRef(waitName));
      } else {
        fsm::TransitionOp::create(fb, loc, StringRef(nextStepName));
      }
      fb.setInsertionPointToEnd(&machine.getBody().front());
    } else {
      // Multi-cycle expansion: emit L sequential sub-states. The first is
      // named STEP_<i> (the entry — preserves backward-compat for tests
      // that match this name). The rest are STEP_<i>_c<c> for c = 1..L-1.
      for (unsigned c = 0; c < L; ++c) {
        std::string subName =
            (c == 0) ? stepName
                     : (stepName + "_c" + std::to_string(c));
        std::string nextSub = (c + 1 < L)
                                  ? (stepName + "_c" + std::to_string(c + 1))
                                  : nextStepName;
        bool isLastSub = (c + 1 == L);
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
        fb.setInsertionPointToEnd(tb);
        fsm::TransitionOp::create(fb, loc, StringRef(nextSub));
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
      // WAIT_i: child j is still producing on the wires. Keep
      // child_active_j high so external mem-port muxes route the child
      // until it deasserts done.
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
      fb.setInsertionPointToEnd(tb);
      fsm::TransitionOp::create(fb, loc, StringRef(nextStepName));
    }
    fb.setInsertionPointToEnd(&machine.getBody().front());
  }

  (void)lastIsWait; // documented above for clarity; not otherwise needed.

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

  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    unsigned addrWidth = getAddrWidth(memInfo.memType);
    Type addrType = IntegerType::get(ctx, addrWidth);
    Type dataType = memInfo.memType.getElementType();
    std::string baseName = "mem" + std::to_string(i);
    ports.push_back({{builder.getStringAttr(baseName + "_addr"), addrType,
                       hw::ModulePort::Direction::Output}});
    ports.push_back({{builder.getStringAttr(baseName + "_wr_data"), dataType,
                       hw::ModulePort::Direction::Output}});
    ports.push_back({{builder.getStringAttr(baseName + "_wr_en"),
                       builder.getI1Type(),
                       hw::ModulePort::Direction::Output}});
  }

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
    mp.addr = Value();
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

  auto *ctx = builder.getContext();
  SmallVector<Value> outputs;

  outputs.push_back(doneSignal);

  for (Value v : resultValues)
    outputs.push_back(v);

  for (auto [i, memInfo] : llvm::enumerate(memrefArgs)) {
    unsigned addrWidth = getAddrWidth(memInfo.memType);
    Type addrType = IntegerType::get(ctx, addrWidth);
    Type dataType = memInfo.memType.getElementType();
    auto it = localMemPortMap.find(memInfo.originalArg);
    if (it != localMemPortMap.end() && it->second.addr)
      outputs.push_back(it->second.addr);
    else
      outputs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
    if (it != localMemPortMap.end() && it->second.wrData)
      outputs.push_back(it->second.wrData);
    else
      outputs.push_back(hw::ConstantOp::create(builder, loc, dataType, 0));
    if (it != localMemPortMap.end() && it->second.wrEn)
      outputs.push_back(it->second.wrEn);
    else
      outputs.push_back(
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0));
  }

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
    unsigned addrWidth = getAddrWidth(memInfo.memType);
    Type addrType = IntegerType::get(ctx, addrWidth);
    Type dataType = memInfo.memType.getElementType();

    // Start with zero defaults.
    Value addr = hw::ConstantOp::create(builder, loc, addrType, 0);
    Value wrData = hw::ConstantOp::create(builder, loc, dataType, 0);
    Value wrEn = hw::ConstantOp::create(builder, loc, i1, 0);

    // Build priority mux chain (last step has lowest priority).
    for (int i = perStepPorts.size() - 1; i >= 0; --i) {
      auto it = perStepPorts[i].find(memInfo.originalArg);
      if (it == perStepPorts[i].end())
        continue;
      auto &ports = it->second;
      Value stepAddr = ports.addr ? ports.addr
          : hw::ConstantOp::create(builder, loc, addrType, 0);
      Value stepWrData = ports.wrData ? ports.wrData
          : hw::ConstantOp::create(builder, loc, dataType, 0);
      Value stepWrEn = ports.wrEn ? ports.wrEn
          : hw::ConstantOp::create(builder, loc, i1, 0);
      addr = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                  stepAddr, addr);
      wrData = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                    stepWrData, wrData);
      wrEn = comb::MuxOp::create(builder, loc, stepRunningSignals[i],
                                  stepWrEn, wrEn);
    }

    mergedPorts[memInfo.originalArg].addr = addr;
    mergedPorts[memInfo.originalArg].wrData = wrData;
    mergedPorts[memInfo.originalArg].wrEn = wrEn;
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
          if (failed(handleLoad(loadOp, hw, localMapping, localMemPorts)))
            return failure();
          continue;
        }

        if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
          // Pre-child stores fire only during the child-launch state.
          if (failed(handleStore(storeOp, hw, localMapping, stepChildStart,
                                  localMemPorts)))
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
        childInputs.push_back(localMemPorts[memInfo.originalArg].rdData);

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

      // Mux child memory ports against the parent's. Use this child's own
      // child_active_j predicate so multiple sibling children can priority-
      // chain on the same memory port without clobbering each other: each
      // child contributes a `mux(child_active_j, childPort, prev)` layer
      // over what was already there.
      Value childActive = fsmChildActives[waitIdx];
      for (auto &memInfo : memrefArgs) {
        Value childAddr = childInst.getResult(outIdx++);
        Value childWrData = childInst.getResult(outIdx++);
        Value childWrEn = childInst.getResult(outIdx++);

        auto &ports = localMemPorts[memInfo.originalArg];
        unsigned addrWidth = getAddrWidth(memInfo.memType);
        Type addrType = IntegerType::get(ctx, addrWidth);
        Type dataType = memInfo.memType.getElementType();
        Value myAddr = ports.addr ? ports.addr
            : hw::ConstantOp::create(hw, loc, addrType, 0);
        Value myWrData = ports.wrData ? ports.wrData
            : hw::ConstantOp::create(hw, loc, dataType, 0);
        Value myWrEn = ports.wrEn ? ports.wrEn
            : hw::ConstantOp::create(hw, loc, i1, 0);

        ports.addr =
            comb::MuxOp::create(hw, loc, childActive, childAddr, myAddr);
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
                                  localMemPorts)))
            return failure();
          continue;
        }

        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          if (failed(handleLoad(loadOp, hw, localMapping, localMemPorts)))
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
          if (failed(handleLoad(loadOp, hw, localMapping, localMemPorts)))
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
                                    localMemPorts)))
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
                                  localMemPorts)))
            return failure();
          continue;
        }

        if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
          if (failed(handleLoad(loadOp, hw, localMapping, localMemPorts)))
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
                               fsmStepCycleGates[stepIdx], localMemPorts)))
        return failure();
    }

    // Map step results.
    auto regOp = cast<LoopScheduleRegisterOp>(body->getTerminator());
    for (auto [result, regVal] :
         llvm::zip(stepOp.getResults(), regOp.getOperands()))
      localMapping.map(result, localMapping.lookup(regVal));

    // After lowering the first step, the loop's condition value (which the
    // verifier guarantees is produced by the first phase) is now available in
    // the mapping. Wire it into the FSM instance via the condition backedge.
    if (stepIdx == 0)
      condBE.setValue(localMapping.lookup(terminatorOp.getCondition()));
  }

  // --- Wire up iter_arg feedback ---
  hw.setInsertionPointToEnd(hwBody);
  for (unsigned i = 0; i < iterArgRegs.size(); ++i) {
    Value feedback = localMapping.lookup(terminatorOp.getIterArgs()[i]);
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
    DenseMap<Value, MemPortMapping> &memPorts) {

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

  // The loop condition is produced by the first stage (verifier-enforced),
  // so it isn't available until that stage's body has been lowered. Stand it
  // up as a backedge here so the active/CE chain can be built first; resolve
  // it after lowering stage 0 below.
  Backedge condValueBE = bb.get(hwBuilder.getI1Type());
  Value condValue = Value(condValueBE);

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

  for (auto [stageIdx, stageOp] : llvm::enumerate(stages)) {
    Block &body = stageOp.getBodyBlock();
    hwBuilder.setInsertionPointToEnd(hwBody);

    for (auto &op : body.getOperations()) {
      if (isa<LoopScheduleRegisterOp>(&op))
        continue;

      if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
        // Gate the write with the stage's clock enable so the store only
        // fires on cycles when this stage actually advances.
        if (failed(handleStore(storeOp, hwBuilder, mapping,
                                stageCE[stageIdx], memPorts)))
          return failure();
        continue;
      }

      if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
        if (failed(handleLoad(loadOp, hwBuilder, mapping, memPorts)))
          return failure();
        continue;
      }

      hwBuilder.clone(op, mapping);
    }

    auto regOp = cast<LoopScheduleRegisterOp>(body.getTerminator());

    // Resolve the condition backedge from stage 0's *unregistered* condition
    // value: the loop's combinational gate should not be delayed by the
    // pipeline registers. The verifier guarantees the terminator's condition
    // is one of stage 0's results, which corresponds to the register operand
    // at the same index.
    if (stageIdx == 0) {
      auto condResult = cast<OpResult>(terminatorOp.getCondition());
      condValueBE.setValue(
          mapping.lookup(regOp.getOperand(condResult.getResultNumber())));
      // Refresh the local condValue handle: BackedgeBuilder's RAUW updates
      // the underlying placeholder's uses, but the local Value snapshot still
      // points at the (now-orphaned) cast op.
      condValue = Value(condValueBE);
    }

    for (auto [regIdx, val] : llvm::enumerate(regOp.getOperands())) {
      Value mappedVal = mapping.lookup(val);
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

  Value notActive = comb::createOrFoldNot(hwBuilder, loc, active);
  auto firstIterReg = seq::CompRegOp::create(
      hwBuilder, loc, notActive, clk, rst, trueConst,
      hwBuilder.getStringAttr(namePrefix + "_first_iter"));

  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value init = mapping.lookup(pipOp.getInits()[i]);
    Value feedback = mapping.lookup(terminatorOp.getIterArgs()[i]);
    Value muxed =
        comb::MuxOp::create(hwBuilder, loc, firstIterReg, init, feedback);
    iterArgBackedges[i].setValue(muxed);
  }

  for (auto [result, termResult] :
       llvm::zip(pipOp.getResults(), terminatorOp.getResults()))
    mapping.map(result, mapping.lookup(termResult));

  Value notCond = comb::createOrFoldNot(hwBuilder, loc, condValue);
  Value notLastCE = comb::createOrFoldNot(hwBuilder, loc, stageCE.back());
  doneSignal = comb::AndOp::create(hwBuilder, loc, notCond, notLastCE);

  return success();
}

//===----------------------------------------------------------------------===//
// Pipeline lowering (legacy path for top-level pipelines)
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleToFSMPass::lowerPipeline(
    LoopSchedulePipelineOp pipOp, OpBuilder &builder, Location loc,
    fsm::MachineOp machine, Block *hwBody, IRMapping &mapping, Value clk,
    Value rst, Value start, StringRef namePrefix, Value &pipelineDone) {

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

  // The loop condition is produced by the first stage (verifier-enforced),
  // so it isn't available until that stage's body has been lowered. Use a
  // backedge for now and resolve it after lowering stage 0 below.
  Backedge condValueBE = bb.get(hwBuilder.getI1Type());
  Value condValue = Value(condValueBE);

  Backedge activeNextBE = bb.get(hwBuilder.getI1Type());
  auto activeReg = seq::CompRegOp::create(
      hwBuilder, loc, Value(activeNextBE), clk, rst, falseConst,
      hwBuilder.getStringAttr(namePrefix + "_active"));
  Value active = activeReg;

  Value holdActive = comb::AndOp::create(hwBuilder, loc, active, condValue);
  Value activeNext = comb::OrOp::create(hwBuilder, loc, start, holdActive);
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
    Value atMax = comb::ICmpOp::create(hwBuilder, loc,
                                       comb::ICmpPredicate::eq,
                                       counterReg, cIIMinusOne);
    Value wrapped =
        comb::MuxOp::create(hwBuilder, loc, atMax, cZero, counterPlusOne);
    Value counterNext =
        comb::MuxOp::create(hwBuilder, loc, active, wrapped, cZero);
    counterBackedge.setValue(counterNext);

    Value isZero = comb::ICmpOp::create(hwBuilder, loc,
                                        comb::ICmpPredicate::eq,
                                        counterReg, cZero);
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

  for (auto [stageIdx, stageOp] : llvm::enumerate(stages)) {
    Block &body = stageOp.getBodyBlock();
    hwBuilder.setInsertionPointToEnd(hwBody);

    for (auto &op : body.getOperations()) {
      if (isa<LoopScheduleRegisterOp>(&op))
        continue;

      if (auto storeOp = dyn_cast<LoopScheduleStoreOp>(&op)) {
        // Gate the write with the stage's clock enable so the store only
        // fires on cycles when this stage actually advances.
        if (failed(handleStore(storeOp, hwBuilder, mapping,
                                stageCE[stageIdx], memPortMap)))
          return failure();
        continue;
      }

      if (auto loadOp = dyn_cast<LoopScheduleLoadOp>(&op)) {
        if (failed(handleLoad(loadOp, hwBuilder, mapping, memPortMap)))
          return failure();
        continue;
      }

      hwBuilder.clone(op, mapping);
    }

    auto regOp = cast<LoopScheduleRegisterOp>(body.getTerminator());

    // Resolve the condition backedge from stage 0's *unregistered* condition
    // value: the loop's combinational gate should not be delayed by the
    // pipeline registers. The verifier guarantees the terminator's condition
    // is one of stage 0's results, which corresponds to the register operand
    // at the same index.
    if (stageIdx == 0) {
      auto condResult = cast<OpResult>(terminatorOp.getCondition());
      condValueBE.setValue(
          mapping.lookup(regOp.getOperand(condResult.getResultNumber())));
      // Refresh the local condValue handle: BackedgeBuilder's RAUW updates
      // the underlying placeholder's uses, but the local Value snapshot still
      // points at the (now-orphaned) cast op.
      condValue = Value(condValueBE);
    }

    for (auto [regIdx, val] : llvm::enumerate(regOp.getOperands())) {
      Value mappedVal = mapping.lookup(val);
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

  Value notActive = comb::createOrFoldNot(hwBuilder, loc, active);
  auto firstIterReg = seq::CompRegOp::create(
      hwBuilder, loc, notActive, clk, rst, trueConst,
      hwBuilder.getStringAttr(namePrefix + "_first_iter"));

  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value init = mapping.lookup(pipOp.getInits()[i]);
    Value feedback = mapping.lookup(terminatorOp.getIterArgs()[i]);
    Value muxed =
        comb::MuxOp::create(hwBuilder, loc, firstIterReg, init, feedback);
    iterArgBackedges[i].setValue(muxed);
  }

  for (auto [result, termResult] :
       llvm::zip(pipOp.getResults(), terminatorOp.getResults()))
    mapping.map(result, mapping.lookup(termResult));

  Value notCond = comb::createOrFoldNot(hwBuilder, loc, condValue);
  Value notLastCE = comb::createOrFoldNot(hwBuilder, loc, stageCE.back());
  pipelineDone = comb::AndOp::create(hwBuilder, loc, notCond, notLastCE);

  OpBuilder fsmBuilder(ctx);
  fsmBuilder.setInsertionPointToEnd(&machine.getBody().front());

  std::string executeStateName = (namePrefix + "_EXECUTE").str();
  std::string doneStateName = (namePrefix + "_DONE").str();

  auto createI1Const = [&](bool val) -> Value {
    return hw::ConstantOp::create(fsmBuilder, loc, fsmBuilder.getI1Type(),
                                  val ? 1 : 0);
  };

  auto executeState = fsm::StateOp::create(fsmBuilder, loc, executeStateName);
  {
    Block *outputBlock = executeState.ensureOutput(fsmBuilder);
    outputBlock->getTerminator()->erase();
    fsmBuilder.setInsertionPointToEnd(outputBlock);
    fsm::OutputOp::create(fsmBuilder, loc,
                          ValueRange{createI1Const(false)});
    Block *transBlock = &executeState.getTransitions().front();
    fsmBuilder.setInsertionPointToEnd(transBlock);
    fsm::TransitionOp::create(
        fsmBuilder, loc, doneStateName,
        [&]() {
          Value condInput = machine.getArgument(1);
          fsm::ReturnOp::create(fsmBuilder, loc, condInput);
        },
        []() {});
  }
  fsmBuilder.setInsertionPointToEnd(&machine.getBody().front());

  auto doneState = fsm::StateOp::create(fsmBuilder, loc, doneStateName);
  {
    Block *outputBlock = doneState.ensureOutput(fsmBuilder);
    outputBlock->getTerminator()->erase();
    fsmBuilder.setInsertionPointToEnd(outputBlock);
    fsm::OutputOp::create(fsmBuilder, loc,
                          ValueRange{createI1Const(true)});
    Block *transBlock = &doneState.getTransitions().front();
    fsmBuilder.setInsertionPointToEnd(transBlock);
    fsm::TransitionOp::create(fsmBuilder, loc, StringRef("IDLE"));
  }
  fsmBuilder.setInsertionPointToEnd(&machine.getBody().front());

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
      unsigned addrWidth = getAddrWidth(memType);
      Type dataType = memType.getElementType();
      Type addrType = IntegerType::get(ctx, addrWidth);

      inputIdx++;
      ports.push_back({{builder.getStringAttr(baseName + "_rd_data"), dataType,
                         hw::ModulePort::Direction::Input}});
            ports.push_back({{builder.getStringAttr(baseName + "_addr"), addrType,
                         hw::ModulePort::Direction::Output}});
            ports.push_back({{builder.getStringAttr(baseName + "_wr_data"), dataType,
                         hw::ModulePort::Direction::Output}});
            ports.push_back({{builder.getStringAttr(baseName + "_wr_en"),
                         builder.getI1Type(),
                         hw::ModulePort::Direction::Output}});
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
    if (isa<MemRefType>(arg.getType())) {
      MemPortMapping mp;
      mp.rdData = hwBody->getArgument(hwArgIdx++);
      mp.addr = Value();
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
  auto *ctx = builder.getContext();
  SmallVector<Value> outputs;

  for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
    auto memType = dyn_cast<MemRefType>(arg.getType());
    if (!memType)
      continue;
    Type addrType = IntegerType::get(ctx, getAddrWidth(memType));
    Type dataType = memType.getElementType();
    auto it = memPortMap.find(arg);
    if (it != memPortMap.end() && it->second.addr)
      outputs.push_back(it->second.addr);
    else
      outputs.push_back(hw::ConstantOp::create(builder, loc, addrType, 0));
    if (it != memPortMap.end() && it->second.wrData)
      outputs.push_back(it->second.wrData);
    else
      outputs.push_back(hw::ConstantOp::create(builder, loc, dataType, 0));
    if (it != memPortMap.end() && it->second.wrEn)
      outputs.push_back(it->second.wrEn);
    else
      outputs.push_back(
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0));
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
// Loop module creation (one hw.module per sequential loop)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pipeline function lowering (old path)
//===----------------------------------------------------------------------===//

LogicalResult
LoopScheduleToFSMPass::lowerPipelineFunction(func::FuncOp funcOp) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();
  OpBuilder builder(ctx);
  IRMapping mapping;

  unsigned clkIdx, rstIdx, startIdx;
  auto hwMod = createHWModule(funcOp, builder, mapping, memPortMap, clkIdx,
                              rstIdx, startIdx);
  Block *hwBody = hwMod.getBodyBlock();
  Value clk = hwBody->getArgument(clkIdx);
  Value rst = hwBody->getArgument(rstIdx);
  Value start = hwBody->getArgument(startIdx);

  builder.setInsertionPointToEnd(hwBody);

  // Clone non-loopschedule ops.
  for (auto &op : funcOp.getBody().front()) {
    if (isa<func::ReturnOp>(&op))
      continue;
    if (isa<LoopScheduleStepOp, LoopScheduleSequentialOp,
            LoopSchedulePipelineOp>(&op))
      continue;
    builder.clone(op, mapping);
  }

  // Find pipeline ops.
  SmallVector<Operation *> loopOps;
  funcOp.walk([&](Operation *op) {
    if (isa<LoopSchedulePipelineOp>(op))
      loopOps.push_back(op);
  });

  if (loopOps.empty())
    return funcOp.emitError("no pipeline ops found");

  // Create single FSM for pipeline(s).
  auto fsmFuncType = FunctionType::get(
      ctx, {builder.getI1Type(), builder.getI1Type()},
      {builder.getI1Type()});
  std::string fsmName = (funcOp.getName() + "_fsm").str();
  builder.setInsertionPointAfter(hwMod);
  auto machine = fsm::MachineOp::create(builder, loc, fsmName, "IDLE",
                                         fsmFuncType);
  machine.setArgNamesAttr(builder.getStrArrayAttr({"start", "cond"}));
  machine.setResNamesAttr(builder.getStrArrayAttr({"done"}));

  std::string firstStateName = "loop0_EXECUTE";
  OpBuilder fsmBuilder(ctx);
  fsmBuilder.setInsertionPointToEnd(&machine.getBody().front());

  auto idleState = fsm::StateOp::create(fsmBuilder, loc, "IDLE");
  {
    Block *outputBlock = idleState.ensureOutput(fsmBuilder);
    outputBlock->getTerminator()->erase();
    fsmBuilder.setInsertionPointToEnd(outputBlock);
    Value falseVal =
        hw::ConstantOp::create(fsmBuilder, loc, fsmBuilder.getI1Type(), 0);
    fsm::OutputOp::create(fsmBuilder, loc, ValueRange{falseVal});
    Block *transBlock = &idleState.getTransitions().front();
    fsmBuilder.setInsertionPointToEnd(transBlock);
    fsm::TransitionOp::create(
        fsmBuilder, loc, firstStateName,
        [&]() {
          fsm::ReturnOp::create(fsmBuilder, loc, machine.getArgument(0));
        },
        []() {});
  }
  fsmBuilder.setInsertionPointToEnd(&machine.getBody().front());

  builder.setInsertionPointToEnd(hwBody);
  BackedgeBuilder bbFunc(builder, loc);
  Backedge condBackedge = bbFunc.get(builder.getI1Type());

  auto fsmInst = fsm::HWInstanceOp::create(
      builder, loc, TypeRange{builder.getI1Type()},
      builder.getStringAttr(fsmName + "_inst"),
      builder.getAttr<FlatSymbolRefAttr>(fsmName),
      ValueRange{start, Value(condBackedge)}, clk, rst);
  Value doneSignal = fsmInst.getResult(0);

  Value condForFSM =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);

  for (auto [idx, loopOp] : llvm::enumerate(loopOps)) {
    std::string prefix = "loop" + std::to_string(idx);

    auto pipOp = cast<LoopSchedulePipelineOp>(loopOp);
    if (auto parentStep = pipOp->getParentOfType<LoopScheduleStepOp>()) {
      for (auto &op : parentStep.getBodyBlock().getOperations()) {
        if (&op == pipOp.getOperation())
          break;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        builder.clone(op, mapping);
      }
    }

    Value pipelineDone;
    if (failed(lowerPipeline(pipOp, builder, loc, machine, hwBody, mapping,
                             clk, rst, start, prefix, pipelineDone)))
      return failure();
    condForFSM = pipelineDone;

    if (auto parentStep = pipOp->getParentOfType<LoopScheduleStepOp>()) {
      auto stepRegOp = cast<LoopScheduleRegisterOp>(
          parentStep.getBodyBlock().getTerminator());
      for (auto [stepResult, regOperand] :
           llvm::zip(parentStep.getResults(), stepRegOp.getOperands()))
        mapping.map(stepResult, mapping.lookup(regOperand));
    }
  }

  condBackedge.setValue(condForFSM);

  builder.setInsertionPointToEnd(hwBody);
  buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, doneSignal);
  funcOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Function lowering (main entry)
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleToFSMPass::lowerFunction(func::FuncOp funcOp) {
  auto *ctx = funcOp.getContext();
  auto loc = funcOp.getLoc();
  OpBuilder builder(ctx);

  // Check if the top-level loop is a pipeline (not nested in sequential).
  // If so, use the legacy pipeline path. If there's a top-level sequential
  // (which may contain pipeline children), use the sequential path.
  bool hasTopLevelSequential = false;
  bool hasTopLevelPipeline = false;
  for (auto &op : funcOp.getBody().front()) {
    if (isa<LoopScheduleSequentialOp>(&op))
      hasTopLevelSequential = true;
    if (isa<LoopSchedulePipelineOp>(&op))
      hasTopLevelPipeline = true;
    if (auto step = dyn_cast<LoopScheduleStepOp>(&op)) {
      for (auto &innerOp : step.getBodyBlock()) {
        if (isa<LoopScheduleSequentialOp>(&innerOp))
          hasTopLevelSequential = true;
        if (isa<LoopSchedulePipelineOp>(&innerOp))
          hasTopLevelPipeline = true;
      }
    }
  }
  if (!hasTopLevelSequential && hasTopLevelPipeline)
    return lowerPipelineFunction(funcOp);

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
    unsigned addrWidth = getAddrWidth(memType);
    auto addrType = IntegerType::get(ctx, addrWidth);
    auto dataType = memType.getElementType();

    std::string name = "local_mem" + std::to_string(i);
    auto hlmem = seq::HLMemOp::create(builder, loc, clk, rst, name,
                                        memType.getShape(), dataType);

    // Shared address backedge for both read and write.
    Backedge addrBE = funcBB.get(addrType);
    Backedge wrDataBE = funcBB.get(dataType);
    Backedge wrEnBE = funcBB.get(i1);

    // Read port: rdEn = NOT wrEn (read when not writing).
    Value notWrEn = comb::XorOp::create(
        builder, loc, Value(wrEnBE),
        hw::ConstantOp::create(builder, loc, i1, 1));
    auto readPort = seq::ReadPortOp::create(
        builder, loc, hlmem.getHandle(),
        ValueRange{Value(addrBE)}, notWrEn, /*latency=*/0);

    // Write port.
    seq::WritePortOp::create(
        builder, loc, hlmem.getHandle(),
        ValueRange{Value(addrBE)}, Value(wrDataBE), Value(wrEnBE),
        /*latency=*/1);

    // Add to memPortMap.
    MemPortMapping mp;
    mp.rdData = readPort.getReadData();
    mp.addr = Value();
    mp.wrData = Value();
    mp.wrEn = Value();
    memPortMap[allocOp.getResult()] = mp;

    hlmemBEs.push_back({allocOp.getResult(), addrBE, wrDataBE, wrEnBE});
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
    memrefArgs.push_back({allocOp.getResult(), allocOp.getType()});
  }

  // --- Single-step fast path (preserves existing behavior) ---
  if (topSteps.size() == 1) {
    LoopScheduleSequentialOp topSeqOp;
    for (auto &innerOp : topSteps[0].getBodyBlock()) {
      if (auto seqOp = dyn_cast<LoopScheduleSequentialOp>(&innerOp)) {
        topSeqOp = seqOp;
        break;
      }
    }
    if (!topSeqOp)
      return funcOp.emitError("single step has no sequential loop");

    // Clone ops before the sequential in its parent step.
    for (auto &op : topSteps[0].getBodyBlock().getOperations()) {
      if (&op == topSeqOp.getOperation())
        break;
      if (isa<LoopScheduleRegisterOp>(&op))
        continue;
      builder.clone(op, mapping);
    }

    LoopNode rootNode;
    unsigned loopCounter = 1;
    buildLoopTree(topSeqOp, rootNode, "loop0", loopCounter);

    hw::HWModuleOp rootLoopModule;
    SmallVector<Value> rootCaptured;
    if (failed(lowerLoopNodeAsModule(rootNode, builder, loc, funcOp, memrefArgs,
                                     mapping, rootLoopModule, rootCaptured)))
      return failure();

    builder.setInsertionPointToEnd(hwBody);
    SmallVector<Value> rootInputs;
    rootInputs.push_back(clk);
    rootInputs.push_back(rst);
    rootInputs.push_back(start);
    for (Value cap : rootCaptured)
      rootInputs.push_back(mapping.lookup(cap));
    for (auto &memInfo : memrefArgs)
      rootInputs.push_back(memPortMap[memInfo.originalArg].rdData);

    auto rootInst = hw::InstanceOp::create(
        builder, loc, rootLoopModule,
        builder.getStringAttr("loop0_inst"), rootInputs, nullptr);

    unsigned outIdx = 0;
    Value doneSignal = rootInst.getResult(outIdx++);
    for (auto result : topSeqOp.getResults())
      mapping.map(result, rootInst.getResult(outIdx++));
    for (auto &memInfo : memrefArgs) {
      memPortMap[memInfo.originalArg].addr = rootInst.getResult(outIdx++);
      memPortMap[memInfo.originalArg].wrData = rootInst.getResult(outIdx++);
      memPortMap[memInfo.originalArg].wrEn = rootInst.getResult(outIdx++);
    }

    // Resolve local hlmem backedges.
    builder.setInsertionPointToEnd(hwBody);
    for (auto &be : hlmemBEs) {
      auto &ports = memPortMap[be.allocResult];
      unsigned addrWidth = getAddrWidth(
          cast<MemRefType>(be.allocResult.getType()));
      auto addrType = IntegerType::get(ctx, addrWidth);
      auto dataType =
          cast<MemRefType>(be.allocResult.getType()).getElementType();
      be.addrBE.setValue(ports.addr ? ports.addr
          : hw::ConstantOp::create(builder, loc, addrType, 0));
      be.wrDataBE.setValue(ports.wrData ? ports.wrData
          : hw::ConstantOp::create(builder, loc, dataType, 0));
      be.wrEnBE.setValue(ports.wrEn ? ports.wrEn
          : hw::ConstantOp::create(builder, loc, i1, 0));
    }

    builder.setInsertionPointToEnd(hwBody);
    buildHWOutput(funcOp, builder, loc, hwBody, mapping, memPortMap, doneSignal);
    funcOp.erase();
    return success();
  }

  // --- Multi-step path ---
  if (topSteps.empty())
    return funcOp.emitError("no top-level steps found");

  unsigned numSteps = topSteps.size();

  // Classify each step: -1 = leaf, 0 = sequential child, 1 = pipeline child.
  SmallVector<int> stepChildKind(numSteps, -1);
  SmallVector<LoopScheduleSequentialOp> stepSeqOps(numSteps);
  SmallVector<LoopSchedulePipelineOp> stepPipOps(numSteps);
  for (unsigned i = 0; i < numSteps; ++i) {
    for (auto &innerOp : topSteps[i].getBodyBlock()) {
      if (auto seqOp = dyn_cast<LoopScheduleSequentialOp>(&innerOp)) {
        stepChildKind[i] = 0;
        stepSeqOps[i] = seqOp;
        break;
      }
      if (auto pipOp = dyn_cast<LoopSchedulePipelineOp>(&innerOp)) {
        stepChildKind[i] = 1;
        stepPipOps[i] = pipOp;
        break;
      }
    }
  }

  // Count non-leaf steps.
  unsigned numChildren = 0;
  SmallVector<int> childIndexForStep(numSteps, -1);
  for (unsigned i = 0; i < numSteps; ++i) {
    if (stepChildKind[i] >= 0) {
      childIndexForStep[i] = numChildren;
      numChildren++;
    }
  }

  // Create function-level FSM.
  std::string funcName = funcOp.getName().str();
  std::string fsmName = funcName + "_fsm";
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  createFunctionFSM(builder, loc, fsmName, stepChildKind);

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

  SmallVector<Type> fsmResultTypes(1 + numChildren + numSteps, i1);
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
  SmallVector<Value> stepRunningSignals(numSteps);
  for (unsigned i = 0; i < numSteps; ++i)
    stepRunningSignals[i] = fsmInst.getResult(fsmOutIdx++);

  // Per-step memory port mappings. Each starts with shared rdData.
  SmallVector<DenseMap<Value, MemPortMapping>> perStepPorts(numSteps);
  for (unsigned i = 0; i < numSteps; ++i) {
    for (auto &memInfo : memrefArgs) {
      perStepPorts[i][memInfo.originalArg].rdData =
          memPortMap[memInfo.originalArg].rdData;
    }
  }

  // Lower each step.
  unsigned loopCounter = 0;
  for (unsigned i = 0; i < numSteps; ++i) {
    builder.setInsertionPointToEnd(hwBody);

    // Clone pre-child ops in this step.
    Operation *childOp = nullptr;
    if (stepSeqOps[i])
      childOp = stepSeqOps[i].getOperation();
    else if (stepPipOps[i])
      childOp = stepPipOps[i].getOperation();

    for (auto &op : topSteps[i].getBodyBlock().getOperations()) {
      if (childOp && &op == childOp)
        break;
      if (isa<LoopScheduleRegisterOp>(&op))
        continue;
      if (isa<LoopScheduleSequentialOp, LoopSchedulePipelineOp>(&op))
        break;
      builder.clone(op, mapping);
    }

    if (stepChildKind[i] == 0) {
      // Sequential child: build loop tree and create child module.
      auto seqOp = stepSeqOps[i];
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
      childInputs.push_back(childStartSignals[childIndexForStep[i]]);
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
      childDoneBEs[childIndexForStep[i]].setValue(childDone);

      // Map seqOp results.
      for (auto result : seqOp.getResults())
        mapping.map(result, childInst.getResult(outIdx++));

      // Extract child memory outputs into per-step ports.
      for (auto &memInfo : memrefArgs) {
        perStepPorts[i][memInfo.originalArg].addr = childInst.getResult(outIdx++);
        perStepPorts[i][memInfo.originalArg].wrData = childInst.getResult(outIdx++);
        perStepPorts[i][memInfo.originalArg].wrEn = childInst.getResult(outIdx++);
      }

      // Clone post-child ops.
      builder.setInsertionPointToEnd(hwBody);
      bool pastChild = false;
      for (auto &op : topSteps[i].getBodyBlock().getOperations()) {
        if (&op == seqOp.getOperation()) {
          pastChild = true;
          continue;
        }
        if (!pastChild)
          continue;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        builder.clone(op, mapping);
      }

    } else if (stepChildKind[i] == 1) {
      // Pipeline child: lower inline.
      auto pipOp = stepPipOps[i];
      std::string pipPrefix = "loop" + std::to_string(loopCounter++);
      Value pipDone;
      if (failed(lowerPipelineChild(pipOp, builder, loc, hwBody, mapping,
                                    clk, rst,
                                    childStartSignals[childIndexForStep[i]],
                                    pipPrefix, pipDone,
                                    perStepPorts[i])))
        return failure();
      childDoneBEs[childIndexForStep[i]].setValue(pipDone);

      // Clone post-child ops.
      builder.setInsertionPointToEnd(hwBody);
      bool pastChild = false;
      for (auto &op : topSteps[i].getBodyBlock().getOperations()) {
        if (&op == pipOp.getOperation()) {
          pastChild = true;
          continue;
        }
        if (!pastChild)
          continue;
        if (isa<LoopScheduleRegisterOp>(&op))
          continue;
        builder.clone(op, mapping);
      }

    } else {
      // Leaf step: lower body with step_running as wrEn gate. The function-
      // level FSM doesn't support multi-cycle steps, so a single gate is
      // used for the entire body.
      Value oneGate = stepRunningSignals[i];
      if (failed(lowerStepBody(&topSteps[i].getBodyBlock(), builder, mapping,
                               ArrayRef<Value>(oneGate), perStepPorts[i])))
        return failure();
    }

    // Register step results for use by subsequent steps.
    auto stepRegOp = cast<LoopScheduleRegisterOp>(
        topSteps[i].getBodyBlock().getTerminator());
    for (auto [stepResult, regOperand] :
         llvm::zip(topSteps[i].getResults(), stepRegOp.getOperands())) {
      Value val = mapping.lookup(regOperand);
      // Create a CE-gated register to hold the result.
      auto regName = builder.getStringAttr(
          funcName + "_step" + std::to_string(i) + "_result_" +
          std::to_string(stepResult.getResultNumber()));
      Value resetVal = createZeroConstant(builder, loc, val.getType());
      auto reg = seq::CompRegClockEnabledOp::create(
          builder, loc, val, clk, stepRunningSignals[i], rst, resetVal,
          regName);
      mapping.map(stepResult, reg);
    }
  }

  // Merge per-step memory ports.
  builder.setInsertionPointToEnd(hwBody);
  DenseMap<Value, MemPortMapping> mergedMemPorts;
  for (auto &memInfo : memrefArgs)
    mergedMemPorts[memInfo.originalArg].rdData =
        memPortMap[memInfo.originalArg].rdData;
  mergeStepMemPorts(builder, loc, perStepPorts, stepRunningSignals,
                    memrefArgs, mergedMemPorts);

  // Copy merged ports to function-level memPortMap.
  for (auto &memInfo : memrefArgs) {
    memPortMap[memInfo.originalArg].addr =
        mergedMemPorts[memInfo.originalArg].addr;
    memPortMap[memInfo.originalArg].wrData =
        mergedMemPorts[memInfo.originalArg].wrData;
    memPortMap[memInfo.originalArg].wrEn =
        mergedMemPorts[memInfo.originalArg].wrEn;
  }

  // Resolve local hlmem backedges.
  builder.setInsertionPointToEnd(hwBody);
  for (auto &be : hlmemBEs) {
    auto &ports = memPortMap[be.allocResult];
    unsigned addrWidth = getAddrWidth(
        cast<MemRefType>(be.allocResult.getType()));
    auto addrType = IntegerType::get(ctx, addrWidth);
    auto dataType =
        cast<MemRefType>(be.allocResult.getType()).getElementType();
    be.addrBE.setValue(ports.addr ? ports.addr
        : hw::ConstantOp::create(builder, loc, addrType, 0));
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

void LoopScheduleToFSMPass::runOnOperation() {
  auto moduleOp = getOperation();

  SmallVector<func::FuncOp> funcs;
  moduleOp.walk([&](func::FuncOp f) { funcs.push_back(f); });

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
