//=== LoopScheduleToCalyx.cpp - LoopSchedule to Calyx pass entry point-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LoopSchedule to Calyx conversion pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LoopScheduleToCalyx.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <iterator>
#include <set>
#include <string>
#include <unordered_set>
#include <variant>

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::cf;
using namespace mlir::func;
using namespace circt::loopschedule;

namespace circt {
namespace loopscheduletocalyx {

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

class LoopWrapper
    : public calyx::WhileOpInterface<loopschedule::LoopInterface> {
public:
  explicit LoopWrapper(loopschedule::LoopInterface op)
      : calyx::WhileOpInterface<loopschedule::LoopInterface>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getBodyArgs();
  }

  Operation::operand_range getInits() override {
    return getOperation().getInits();
  }

  Block *getBodyBlock() override { return getOperation().getBodyBlock(); }

  Block *getConditionBlock() override {
    return getOperation().getConditionBlock();
  }

  Value getConditionValue() override {
    return getOperation().getConditionValue();
  }

  std::optional<int64_t> getBound() override {
    return getOperation().getBound();
  }

  bool isPipelined() { return getOperation().isPipelined(); }
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

/// A variant of types representing schedulable operations.
using Schedulable =
    std::variant<calyx::StaticGroupOp, LoopWrapper, PhaseInterface>;

using PhaseRegister = std::variant<calyx::RegisterOp, Value>;

/// Holds additional information required for scheduling Pipeline pipelines.
class PhaseScheduler : public calyx::SchedulerInterface<Schedulable> {
public:
  /// Register reg as being the idx'th loop register for the step/stage.
  void addPhaseReg(Operation *phase, PhaseRegister reg, unsigned idx) {
    assert(phaseRegs[phase].count(idx) == 0);
    assert(idx < phase->getNumResults());
    if (auto *valuePtr = std::get_if<Value>(&reg); valuePtr) {
      if (auto cell =
              dyn_cast<calyx::CellInterface>(valuePtr->getDefiningOp())) {
        assert(!cell.isCombinational());
      }
    }
    phaseRegs[phase][idx] = reg;
  }

  /// Return a mapping of step/stage result indices to sink registers.
  const DenseMap<unsigned, PhaseRegister> &getPhaseRegs(Operation *phase) {
    return phaseRegs[phase];
  }

  void setLoopIterValue(LoopSchedulePipelineOp loop, Value v) {
    loopIterValues[loop] = v;
  }

  Value getLoopIterValue(LoopSchedulePipelineOp loop) {
    return loopIterValues[loop];
  }

  void setCondReg(LoopInterface loop, calyx::RegisterOp reg) {
    condRegs[loop] = reg;
  }

  calyx::RegisterOp getCondReg(LoopInterface loop) {
    assert(condRegs.contains(loop));
    return condRegs[loop];
  }

  void setCondGroup(LoopInterface loop, calyx::StaticGroupOp group) {
    condGroups[loop] = group;
  }

  calyx::StaticGroupOp getCondGroup(LoopInterface loop) {
    assert(condGroups.contains(loop));
    return condGroups[loop];
  }

  void setIncrGroup(LoopSchedulePipelineOp loop, calyx::StaticGroupOp group) {
    incrGroup[loop] = group;
  }

  calyx::StaticGroupOp getIncrGroup(LoopSchedulePipelineOp loop) {
    return incrGroup[loop];
  }

  void setGuardValue(PhaseInterface phase, Value v) { guardValues[phase] = v; }

  std::optional<Value> getGuardValue(PhaseInterface phase) {
    if (!guardValues.contains(phase))
      return std::nullopt;
    return guardValues[phase];
  }

  void setGuardRegister(PhaseInterface phase, calyx::RegisterOp v) {
    guardRegisters[phase] = v;
  }

  std::optional<calyx::RegisterOp> getGuardRegister(PhaseInterface phase) {
    if (!guardRegisters.contains(phase))
      return std::nullopt;
    return guardRegisters[phase];
  }

  void setStallValue(LoopInterface loop, Value v) { stallValues[loop] = v; }

  std::optional<Value> getStallValue(LoopInterface loop) {
    if (!stallValues.contains(loop))
      return std::nullopt;
    return stallValues[loop];
  }

  void addPhaseDoneValue(PhaseInterface phase, Value v) {
    doneValues[phase].push_back(v);
  }

  SmallVector<Value> getPhaseDoneValues(PhaseInterface phase) {
    return doneValues[phase];
  }

  void addStallPort(LoopInterface loop, Value v) {
    stallPorts[loop].push_back(v);
  }

  SmallVector<Value> getStallPorts(LoopInterface loop) {
    return stallPorts[loop];
  }

  void memoryInterfaceReadEnSet(const calyx::MemoryInterface &interface) {
    readEnSet.push_back(interface);
  }

  void memoryInterfaceWriteEnSet(const calyx::MemoryInterface &interface) {
    writeEnSet.push_back(interface);
  }

  SmallVector<calyx::MemoryInterface> interfacesReadEnNotSet() {
    SmallVector<calyx::MemoryInterface> interfaces;

    for (auto interface : writeEnSet) {
      if (interface.readEnOpt().has_value()) {
        int count = 0;
        for (const auto &readInterface : readEnSet) {
          if (readInterface == interface)
            count++;
        }
        if (count == 0)
          interfaces.push_back(interface);
      }
    }

    return interfaces;
  }

  SmallVector<calyx::MemoryInterface> interfacesWriteEnNotSet() {
    SmallVector<calyx::MemoryInterface> interfaces;

    for (auto interface : readEnSet) {
      if (interface.writeEnOpt().has_value()) {
        int count = 0;
        for (const auto &writeInterface : writeEnSet) {
          if (writeInterface == interface)
            count++;
        }
        if (count == 0)
          interfaces.push_back(interface);
      }
    }

    return interfaces;
  }

private:
  /// A mapping from steps/stages to their registers.
  DenseMap<Operation *, DenseMap<unsigned, PhaseRegister>> phaseRegs;

  DenseMap<LoopInterface, Value> loopIterValues;

  DenseMap<LoopInterface, calyx::RegisterOp> condRegs;

  DenseMap<LoopInterface, calyx::StaticGroupOp> condGroups;

  DenseMap<LoopInterface, calyx::StaticGroupOp> incrGroup;

  // Values that guard the execution of the phase
  DenseMap<PhaseInterface, Value> guardValues;

  DenseMap<PhaseInterface, calyx::RegisterOp> guardRegisters;

  DenseMap<LoopInterface, Value> stallValues;

  DenseMap<PhaseInterface, SmallVector<Value>> doneValues;

  DenseMap<LoopInterface, SmallVector<Value>> stallPorts;

  SmallVector<calyx::MemoryInterface> readEnSet;

  SmallVector<calyx::MemoryInterface> writeEnSet;
};

/// Handles the current state of lowering of a Calyx component. It is mainly
/// used as a key/value store for recording information during partial lowering,
/// which is required at later lowering passes.
class ComponentLoweringState
    : public calyx::ComponentLoweringStateInterface,
      public calyx::LoopLoweringStateInterface<LoopWrapper,
                                               calyx::StaticGroupOp>,
      public PhaseScheduler {
public:
  ComponentLoweringState(calyx::ComponentOp component)
      : calyx::ComponentLoweringStateInterface(component) {}
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Iterate through the operations of a source function and instantiate
/// components or primitives based on the type of the operations.
class BuildOpGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(op)
              .template Case<arith::ConstantOp, ReturnOp, BranchOpInterface,
                             /// memref
                             memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                             memref::StoreOp,
                             /// memory interface
                             calyx::StoreLoweringInterface,
                             calyx::LoadLoweringInterface,
                             calyx::AllocLoweringInterface,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp,
                             AndIOp, XOrIOp, OrIOp, ExtUIOp, TruncIOp, MulIOp,
                             DivUIOp, RemUIOp, RemSIOp, IndexCastOp,
                             /// loop schedule
                             LoopInterface, LoopScheduleTerminatorOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<FuncOp, LoopScheduleRegisterOp, PhaseInterface>(
                  [&](auto) {
                    /// Skip: these special cases will be handled separately.
                    return true;
                  })
              .Default([&](auto op) {
                op->dump();
                op->emitError() << "Unhandled operation during BuildOpGroups() test " << op;
                return false;
              });

      return opBuiltSuccessfully ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    return success(opBuiltSuccessfully);
  }

private:
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter,
                        BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        arith::ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShLIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocaOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::LoadOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::StoreOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        calyx::LoadLoweringInterface op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        calyx::StoreLoweringInterface op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        calyx::AllocLoweringInterface op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, LoopInterface op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        LoopScheduleTerminatorOp op) const;

  /// buildLibraryOp will build a TCalyxLibOp inside a TGroupOp based on the
  /// source operation TSrcOp.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               TypeRange srcTypes, TypeRange dstTypes) const {
    SmallVector<Type> types;
    llvm::append_range(types, srcTypes);
    llvm::append_range(types, dstTypes);

    auto calyxOp =
        getState<ComponentLoweringState>().getNewLibraryOpInstance<TCalyxLibOp>(
            rewriter, op.getLoc(), types);

    auto directions = calyxOp.portDirections();
    SmallVector<Value, 4> opInputPorts;
    SmallVector<Value, 4> opOutputPorts;
    for (auto dir : enumerate(directions)) {
      if (dir.value() == calyx::Direction::Input)
        opInputPorts.push_back(calyxOp.getResult(dir.index()));
      else
        opOutputPorts.push_back(calyxOp.getResult(dir.index()));
    }
    assert(
        opInputPorts.size() == op->getNumOperands() &&
        opOutputPorts.size() == op->getNumResults() &&
        "Expected an equal number of in/out ports in the Calyx library op with "
        "respect to the number of operands/results of the source operation.");

    /// Create assignments to the inputs of the library op.
    auto group = createGroupForOp<TGroupOp>(rewriter, op);
    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    for (auto dstOp : enumerate(opInputPorts))
      rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                       op->getOperand(dstOp.index()));

    /// Replace the result values of the source operator with the new operator.
    for (auto res : enumerate(opOutputPorts)) {
      getState<ComponentLoweringState>().registerEvaluatingGroup(res.value(),
                                                                 group);
      op->getResult(res.index()).replaceAllUsesWith(res.value());
    }
    return success();
  }

  /// buildLibraryOp which provides in- and output types based on the operands
  /// and results of the op argument.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op) const {
    return buildLibraryOp<TGroupOp, TCalyxLibOp, TSrcOp>(
        rewriter, op, op.getOperandTypes(), op->getResultTypes());
  }

  /// Creates a group named by the basic block which the input op resides in.
  template <typename TGroupOp>
  TGroupOp createGroupForOp(PatternRewriter &rewriter, Operation *op) const {
    Block *block = op->getBlock();
    auto groupName = getState<ComponentLoweringState>().getUniqueName(
        loweringState().blockName(block));
    return calyx::createGroup<TGroupOp>(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
        op->getLoc(), groupName);
  }

  calyx::StaticGroupOp createStaticGroupForOp(PatternRewriter &rewriter,
                                              Operation *op,
                                              uint64_t latency) const {
    auto name = op->getName().getStringRef().split(".").second;
    auto groupName = getState<ComponentLoweringState>().getUniqueName(name);
    return calyx::createStaticGroup(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
        op->getLoc(), groupName, latency);
  }

  /// buildLibraryBinaryPipeOp will build a TCalyxLibBinaryPipeOp, to
  /// deal with MulIOp, DivUIOp, RemUIOp, RemSIOp, DivSIOp.
  template <typename TOpType, typename TSrcOp>
  LogicalResult buildLibraryBinaryPipeOp(PatternRewriter &rewriter, TSrcOp op,
                                         TOpType opPipe, Value out) const {
    StringRef opName = TSrcOp::getOperationName().split(".").second;
    Location loc = op.getLoc();
    Type width = op.getResult().getType();
    // Pass the result from the Operation to the Calyx primitive.
    op.getResult().replaceAllUsesWith(out);
    PhaseInterface parent = cast<PhaseInterface>(op->getParentOp());
    auto latency = parent.isStatic() ? 1 : 4;
    auto group = createStaticGroupForOp(rewriter, op, latency);
    // getState<ComponentLoweringState>().addBlockSchedulable(op->getBlock(),
    //                                                         group);

    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getLeft(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getRight(), op.getRhs());
    // rewriter.create<calyx::AssignOp>(
    //     loc, opPipe.getGo(),
    //     createConstant(loc, rewriter, getComponent(), 1, 1));
    // getState<ComponentLoweringState>().registerStartGroup(out, startGroup);

    // auto endGroup = createGroupForOp<calyx::CombGroupOp>(rewriter, op);

    // Register the values for the pipeline.
    getState<ComponentLoweringState>().registerEvaluatingGroup(out, group);
    // getState<ComponentLoweringState>().registerEvaluatingGroup(opPipe.getLeft(),
    //                                                            endGroup);
    // getState<ComponentLoweringState>().registerEvaluatingGroup(
    //     opPipe.getRight(), endGroup);

    return success();
  }

  template <typename TOpType, typename TSrcOp>
  LogicalResult buildLibraryBinarySeqOp(PatternRewriter &rewriter, TSrcOp op,
                                        TOpType opPipe, Value out) const {
    StringRef opName = TSrcOp::getOperationName().split(".").second;
    Location loc = op.getLoc();
    Type width = op.getResult().getType();
    // Pass the result from the Operation to the Calyx primitive.
    op.getResult().replaceAllUsesWith(out);
    PhaseInterface parent = cast<PhaseInterface>(op->getParentOp());
    auto latency = 4;
    auto group = createStaticGroupForOp(rewriter, op, latency);
    // getState<ComponentLoweringState>().addBlockSchedulable(op->getBlock(),
    //                                                         group);

    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getLeft(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getRight(), op.getRhs());
    rewriter.create<calyx::AssignOp>(
        loc, opPipe.getGo(),
        createConstant(loc, rewriter, getComponent(), 1, 1));
    // getState<ComponentLoweringState>().registerStartGroup(out, startGroup);

    // auto endGroup = createGroupForOp<calyx::CombGroupOp>(rewriter, op);

    // Register the values for the pipeline.
    getState<ComponentLoweringState>().registerEvaluatingGroup(out, group);
    // getState<ComponentLoweringState>().registerEvaluatingGroup(opPipe.getLeft(),
    //                                                            endGroup);
    // getState<ComponentLoweringState>().registerEvaluatingGroup(
    //     opPipe.getRight(), endGroup);

    return success();
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          calyx::MemoryInterface memoryInterface,
                          Operation::operand_range addressValues) const {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryInterface.addrPorts();
    if (addressValues.empty()) {
      assert(
          addrPorts.size() == 1 &&
          "We expected a 1 dimensional memory of size 1 because there were no "
          "address assignment values");
      // Assign 1'd0 to the address port.
      rewriter.create<calyx::AssignOp>(
          loc, addrPorts[0],
          createConstant(loc, rewriter, getComponent(), 1, 0));
    } else {
      assert(addrPorts.size() == addressValues.size() &&
             "Mismatch between number of address ports of the provided memory "
             "and address assignment values");
      for (auto address : enumerate(addressValues))
        rewriter.create<calyx::AssignOp>(loc, addrPorts[address.index()],
                                         address.value());
    }
  }
};

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::LoadOp loadOp) const {
  Value memref = loadOp.getMemref();
  Block *block = loadOp->getBlock();

  if (!calyx::singleLoadFromMemoryInBlock(memref, block)) {
    loadOp->emitOpError("LoadOp has more than one load in block");
    return failure();
  }

  if (!calyx::noStoresToMemoryInBlock(memref, block)) {
    loadOp->emitOpError("LoadOp has stores in block");
    return failure();
  }

  auto memoryInterface =
      getState<ComponentLoweringState>().getMemoryInterface(memref);

  getState<ComponentLoweringState>().memoryInterfaceReadEnSet(memoryInterface);

  // TODO: Check only one access to this memory per cycle
  // Single load from memory; we do not need to write the
  // output to a register. This is essentially a "combinational read" under
  // current Calyx semantics with memory, and thus can be done in a
  // combinational group. Note that if any stores are done to this memory,
  // we require that the load and store be in separate non-combinational
  // groups to avoid reading and writing to the same memory in the same group.
  auto group = createStaticGroupForOp(rewriter, loadOp, 1);
  assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryInterface,
                     loadOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  auto one =
      calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
  rewriter.create<calyx::AssignOp>(loadOp.getLoc(), memoryInterface.readEn(),
                                   one);

  // We refrain from replacing the loadOp result with
  // memoryInterface.readData, since multiple loadOp's need to be converted
  // to a single memory's ReadData. If this replacement is done now, we lose
  // the link between which SSA memref::LoadOp values map to which groups for
  // loading a value from the Calyx memory. At this point of lowering, we
  // keep the memref::LoadOp SSA value, and do value replacement _after_
  // control has been generated (see LateSSAReplacement). This is *vital* for
  // things such as InlineCombGroups to be able to properly track which
  // memory assignment groups belong to which accesses.
  getState<ComponentLoweringState>().registerEvaluatingGroup(loadOp.getResult(),
                                                             group);

  // loadOp.replaceAllUsesWith(memoryInterface.readData());
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::StoreOp storeOp) const {
  auto memoryInterface = getState<ComponentLoweringState>().getMemoryInterface(
      storeOp.getMemref());

  getState<ComponentLoweringState>().memoryInterfaceWriteEnSet(memoryInterface);

  auto group = createStaticGroupForOp(rewriter, storeOp, 1);

  // This is a sequential group, so register it as being schedulable for the
  // block.
  getState<ComponentLoweringState>().addBlockSchedulable(storeOp->getBlock(),
                                                         group);
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryInterface,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeData(), storeOp.getValueToStore());
  auto constant =
      calyx::createConstant(storeOp.getLoc(), rewriter, getComponent(), 1, 1);
  rewriter.create<calyx::AssignOp>(storeOp.getLoc(), memoryInterface.writeEn(),
                                   constant.getResult());
  // rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(),
  // memoryInterface.done());

  // getState<ComponentLoweringState>().registerSinkOperations(storeOp,
  //                                                           group);

  return success();
}

LogicalResult
BuildOpGroups::buildOp(PatternRewriter &rewriter,
                       calyx::LoadLoweringInterface loadOp) const {
  Value memref = loadOp.getMemoryValue();

  auto memoryInterface =
      getState<ComponentLoweringState>().getMemoryInterface(memref);

  getState<ComponentLoweringState>().memoryInterfaceReadEnSet(memoryInterface);

  auto group = createStaticGroupForOp(rewriter, loadOp, 1);
  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  auto &state = getState<ComponentLoweringState>();
  std::optional<Block *> blockOpt;
  auto res = loadOp.connectToMemInterface(rewriter, group, getComponent(),
                                          state, blockOpt);
  if (res.failed())
    return failure();

  return success();
}

LogicalResult
BuildOpGroups::buildOp(PatternRewriter &rewriter,
                       calyx::StoreLoweringInterface storeOp) const {
  auto memoryInterface = getState<ComponentLoweringState>().getMemoryInterface(
      storeOp.getMemoryValue());

  getState<ComponentLoweringState>().memoryInterfaceWriteEnSet(memoryInterface);

  auto group = createStaticGroupForOp(rewriter, storeOp, 1);

  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  auto &state = getState<ComponentLoweringState>();
  std::optional<Block *> blockOpt;
  auto res = storeOp.connectToMemInterface(rewriter, group, getComponent(),
                                           state, blockOpt);
  if (res.failed())
    return failure();

  if (blockOpt.has_value()) {
    state.addBlockSchedulable(blockOpt.value(), group);
  }

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulIOp op) const {
  Location loc = op.getLoc();
  Type width = op.getResult().getType(), one = rewriter.getI1Type();
  if (isa<LoopSchedulePipelineStageOp>(op->getParentOp())) {
    auto pipeline = op->getParentOfType<LoopSchedulePipelineOp>();
    if (pipeline.canStall()) {
      auto mulPipe =
          getState<ComponentLoweringState>()
              .getNewLibraryOpInstance<calyx::StallableMultLibOp>(
                  rewriter, loc, {one, one, one, width, width, width});
      getState<ComponentLoweringState>().addStallPort(pipeline,
                                                      mulPipe.getStall());
      return buildLibraryBinaryPipeOp<calyx::StallableMultLibOp>(
          rewriter, op, mulPipe,
          /*out=*/mulPipe.getOut());
    }
    auto mulPipe = getState<ComponentLoweringState>()
                       .getNewLibraryOpInstance<calyx::PipelinedMultLibOp>(
                           rewriter, loc, {one, one, width, width, width});
    return buildLibraryBinaryPipeOp<calyx::PipelinedMultLibOp>(
        rewriter, op, mulPipe,
        /*out=*/mulPipe.getOut());
  }

  auto mulSeq =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::SeqMultLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});

  return buildLibraryBinarySeqOp<calyx::SeqMultLibOp>(rewriter, op, mulSeq,
                                                      mulSeq.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivUIOp op) const {
  Location loc = op.getLoc();
  Type width = op.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::SeqDivULibOp>(
              rewriter, loc, {one, one, one, width, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::SeqDivULibOp>(
      rewriter, op, divPipe,
      /*out=*/divPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemUIOp op) const {
  Location loc = op.getLoc();
  Type width = op.getResult().getType(), one = rewriter.getI1Type();
  if (isa<LoopSchedulePipelineStageOp>(op->getParentOp())) {
    op.emitError() << "RemUI is not pipelineable";
    return failure();
  }

  auto remUSeq =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::SeqRemULibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinarySeqOp<calyx::SeqRemULibOp>(rewriter, op, remUSeq,
                                                      remUSeq.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemSIOp op) const {
  Location loc = op.getLoc();
  Type width = op.getResult().getType(), one = rewriter.getI1Type();
  if (isa<LoopSchedulePipelineStageOp>(op->getParentOp())) {
    op.emitError() << "RemSI is not pipelineable";
    return failure();
  }

  auto remSSeq =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::SeqRemSLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinarySeqOp<calyx::SeqRemSLibOp>(rewriter, op, remSSeq,
                                                      remSSeq.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivSIOp op) const {
  Location loc = op.getLoc();
  Type width = op.getResult().getType(), one = rewriter.getI1Type();
  if (isa<LoopSchedulePipelineStageOp>(op->getParentOp())) {
    op.emitError() << "DivSI is not pipelineable";
    return failure();
  }

  auto divSSeq =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::SeqDivSLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinarySeqOp<calyx::SeqDivSLibOp>(rewriter, op, divSSeq,
                                                      divSSeq.getOut());
}

template <typename TAllocOp>
static LogicalResult buildAllocOp(ComponentLoweringState &componentState,
                                  PatternRewriter &rewriter, TAllocOp allocOp) {
  rewriter.setInsertionPointToStart(
      componentState.getComponentOp().getBodyBlock());
  MemRefType memtype = allocOp.getType();
  SmallVector<int64_t> addrSizes;
  SmallVector<int64_t> sizes;
  for (int64_t dim : memtype.getShape()) {
    sizes.push_back(dim);
    addrSizes.push_back(calyx::handleZeroWidth(dim));
  }
  // If memref has no size (e.g., memref<i32>) create a 1 dimensional memory of
  // size 1.
  if (sizes.empty() && addrSizes.empty()) {
    sizes.push_back(1);
    addrSizes.push_back(1);
  }
  auto memoryOp = rewriter.create<calyx::SeqMemoryOp>(
      allocOp.getLoc(), componentState.getUniqueName("mem"),
      memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);
  // Externalize memories by default. This makes it easier for the native
  // compiler to provide initialized memories.
  // memoryOp->setAttr("external",
  //                   IntegerAttr::get(rewriter.getI1Type(), llvm::APInt(1,
  //                   1)));
  componentState.registerMemoryInterface(allocOp.getResult(),
                                         calyx::MemoryInterface(memoryOp));
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocaOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult
BuildOpGroups::buildOp(PatternRewriter &rewriter,
                       calyx::AllocLoweringInterface allocOp) const {
  rewriter.setInsertionPointToStart(getComponent().getBodyBlock());
  allocOp.insertMemory(rewriter, getState<ComponentLoweringState>());

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     LoopInterface op) const {
  LoopWrapper loop(op);

  /// Create iteration argument registers.
  /// The iteration argument registers will be referenced:
  /// - In the "before" part of the while loop, calculating the conditional,
  /// - In the "after" part of the while loop,
  /// - Outside the while loop, rewriting the while loop return values.
  for (auto arg : enumerate(loop.getBodyArgs())) {
    std::string name = getState<ComponentLoweringState>()
                           .getUniqueName(loop.getOperation())
                           .str() +
                       "_arg" + std::to_string(arg.index());

    auto reg =
        createRegister(arg.value().getLoc(), rewriter, getComponent(),
                       arg.value().getType().getIntOrFloatBitWidth(), name);
    getState<ComponentLoweringState>().addLoopIterReg(loop, reg, arg.index());

    arg.value().replaceAllUsesWith(reg.getOut());

    loop.getConditionBlock()
        ->getArgument(arg.index())
        .replaceAllUsesWith(loop.getInits()[arg.index()]);
  }

  /// Create iter args initial value assignment group(s), one per register.
  auto numOperands = loop.getOperation()->getNumOperands();
  for (size_t i = 0; i < numOperands; ++i) {
    auto initGroupOp =
        getState<ComponentLoweringState>().buildLoopIterArgAssignments(
            rewriter, loop, getState<ComponentLoweringState>().getComponentOp(),
            getState<ComponentLoweringState>().getUniqueName(
                loop.getOperation()) +
                "_init_" + std::to_string(i),
            loop.getOperation()->getOpOperand(i));
    getState<ComponentLoweringState>().addLoopInitGroup(loop, initGroupOp);
  }

  if (loop.isPipelined()) {
    auto groupName = getState<ComponentLoweringState>().getUniqueName("incr");
    auto incrGroup = calyx::createStaticGroup(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
        op->getLoc(), groupName, 1);
    auto incrReg =
        createRegister(op.getLoc(), rewriter, getComponent(), 32,
                       getState<ComponentLoweringState>().getUniqueName("idx"));
    auto width = rewriter.getI32Type();
    auto addOp = getState<ComponentLoweringState>()
                     .getNewLibraryOpInstance<calyx::AddLibOp>(
                         rewriter, op.getLoc(), {width, width, width});
    rewriter.setInsertionPointToEnd(incrGroup.getBodyBlock());
    rewriter.create<calyx::AssignOp>(op.getLoc(), addOp.getLeft(),
                                     incrReg.getOut());
    auto constant =
        calyx::createConstant(op.getLoc(), rewriter, getComponent(), 32, 1);
    rewriter.create<calyx::AssignOp>(op.getLoc(), addOp.getRight(), constant);
    rewriter.create<calyx::AssignOp>(op.getLoc(), incrReg.getIn(),
                                     addOp.getOut());
    auto oneI1 =
        calyx::createConstant(op.getLoc(), rewriter, getComponent(), 1, 1);
    rewriter.create<calyx::AssignOp>(op.getLoc(), incrReg.getWriteEn(), oneI1);
    getState<ComponentLoweringState>().registerEvaluatingGroup(addOp.getOut(),
                                                               incrGroup);
    getState<ComponentLoweringState>().registerEvaluatingGroup(addOp.getLeft(),
                                                               incrGroup);
    getState<ComponentLoweringState>().registerEvaluatingGroup(addOp.getRight(),
                                                               incrGroup);

    // Build reset for increment counter
    auto initName =
        getState<ComponentLoweringState>().getUniqueName("incr_init");
    auto incrInit = calyx::createStaticGroup(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
        op->getLoc(), initName, 1);
    rewriter.setInsertionPointToEnd(incrInit.getBodyBlock());
    auto zero =
        calyx::createConstant(op.getLoc(), rewriter, getComponent(), 32, 0);
    rewriter.create<calyx::AssignOp>(op.getLoc(), incrReg.getIn(), zero);
    rewriter.create<calyx::AssignOp>(op.getLoc(), incrReg.getWriteEn(), oneI1);
    getState<ComponentLoweringState>().addLoopInitGroup(loop, incrInit);

    // Set pipeline iter value stuff
    auto pipeline = cast<LoopSchedulePipelineOp>(loop.getOperation());
    // if (pipeline.getII() != 1) {
    //   return pipeline.emitOpError("LoopScheduleToCalyx currently does not "
    //                               "support pipelines with II > 1");
    // }
    getState<ComponentLoweringState>().setIncrGroup(pipeline, incrGroup);
    getState<ComponentLoweringState>().setLoopIterValue(pipeline,
                                                        addOp.getOut());
  }

  /// Add the while op to the list of schedulable things in the current
  /// block.
  getState<ComponentLoweringState>().addBlockSchedulable(
      loop.getOperation()->getBlock(), loop);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     LoopScheduleTerminatorOp op) const {
  if (op.getOperands().empty())
    return success();

  // Replace the pipeline's result(s) with the terminator's results.
  auto *pipeline = op->getParentOp();
  for (size_t i = 0, e = pipeline->getNumResults(); i < e; ++i)
    pipeline->getResult(i).replaceAllUsesWith(op.getResults()[i]);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     BranchOpInterface brOp) const {
  /// Branch argument passing group creation
  /// Branch operands are passed through registers. In BuildBasicBlockRegs we
  /// created registers for all branch arguments of each block. We now
  /// create groups for assigning values to these registers.
  Block *srcBlock = brOp->getBlock();
  for (auto succBlock : enumerate(brOp->getSuccessors())) {
    auto succOperands = brOp.getSuccessorOperands(succBlock.index());
    if (succOperands.empty())
      continue;
    // Create operand passing group
    std::string groupName = loweringState().blockName(srcBlock) + "_to_" +
                            loweringState().blockName(succBlock.value());
    auto groupOp = calyx::createStaticGroup(rewriter, getComponent(),
                                            brOp.getLoc(), groupName, 1);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs =
        getState<ComponentLoweringState>().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getForwardedOperands())) {
      auto reg = dstBlockArgRegs[arg.index()];
      calyx::buildAssignmentsForRegisterWrite(
          rewriter, groupOp,
          getState<ComponentLoweringState>().getComponentOp(), reg,
          arg.value());
    }
    /// Register the group as a block argument group, to be executed
    /// when entering the successor block from this block (srcBlock).
    getState<ComponentLoweringState>().addBlockArgGroup(
        srcBlock, succBlock.value(), groupOp);
  }
  return success();
}

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName =
      getState<ComponentLoweringState>().getUniqueName("ret_assign");
  auto groupOp = calyx::createStaticGroup(rewriter, getComponent(),
                                          retOp.getLoc(), groupName, 1);
  for (auto op : enumerate(retOp.getOperands())) {
    auto reg = getState<ComponentLoweringState>().getReturnReg(op.index());
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, groupOp, getState<ComponentLoweringState>().getComponentOp(),
        reg, op.value());
  }
  /// Schedule group for execution for when executing the return op block.
  getState<ComponentLoweringState>().addBlockSchedulable(retOp->getBlock(),
                                                         groupOp);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     arith::ConstantOp constOp) const {
  /// Move constant operations to the compOp body as hw::ConstantOp's.
  APInt value;
  calyx::matchConstantOp(constOp, value);
  auto hwConstOp =
      calyx::createConstant(constOp.getLoc(), rewriter, getComponent(),
                            value.getBitWidth(), value.getLimitedValue());
  rewriter.replaceAllUsesWith(constOp.getResult(), hwConstOp.getResult());

  // auto groupName = getState<ComponentLoweringState>().getUniqueName(
  //     "const");
  // auto group = createGroupForOp<calyx::CombGroupOp>(
  //     rewriter, constOp);

  // hwConstOp.dump();
  // llvm::errs() << "test\n";
  // getState<ComponentLoweringState>().registerEvaluatingGroup(hwConstOp.getResult(),
  //                                                            group);
  // getState<ComponentLoweringState>().registerEvaluatingGroup(constOp.getResult(),
  //                                                            group);
  // auto evalGroup =
  // getState<ComponentLoweringState>().getEvaluatingGroup(hwConstOp.getResult());
  // evalGroup.dump();

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AddLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SubIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SubLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShLIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     OrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  switch (op.getPredicate()) {
  case CmpIPredicate::eq:
    return buildLibraryOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  case CmpIPredicate::ne:
    return buildLibraryOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpIPredicate::uge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GeLibOp>(rewriter, op);
  case CmpIPredicate::ult:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LtLibOp>(rewriter, op);
  case CmpIPredicate::ugt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GtLibOp>(rewriter, op);
  case CmpIPredicate::ule:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LeLibOp>(rewriter, op);
  case CmpIPredicate::sge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpIPredicate::slt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpIPredicate::sgt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpIPredicate::sle:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  }
  llvm_unreachable("unsupported comparison predicate");
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = calyx::convIndexType(rewriter, op.getOperand().getType());
  Type targetType = calyx::convIndexType(rewriter, op.getResult().getType());
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
  LogicalResult res = success();

  if (targetBits == sourceBits) {
    /// Drop the index cast and replace uses of the target value with the source
    /// value.
    op.getResult().replaceAllUsesWith(op.getOperand());
  } else {
    /// pad/slice the source operand.
    if (sourceBits > targetBits)
      res = buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
          rewriter, op, {sourceType}, {targetType});
    else
      res = buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
          rewriter, op, {sourceType}, {targetType});
  }
  rewriter.eraseOp(op);
  return res;
}

/// Builds condition checks for each loop.
class BuildConditionChecks : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](LoopInterface loop) {
      getState<ComponentLoweringState>().setUniqueName(loop, "loop");

      if (loop.isPipelined()) {
        return;
      }

      /// Create condition register.
      auto condValue = loop.getConditionValue();
      std::string name = getState<ComponentLoweringState>()
                             .getUniqueName(loop.getOperation())
                             .str() +
                         "_cond";

      auto condReg =
          createRegister(condValue.getLoc(), rewriter, getComponent(), 1, name);
      getState<ComponentLoweringState>().setCondReg(loop, condReg);

      // Create condition init group.
      auto initGroupName =
          getState<ComponentLoweringState>().getUniqueName("cond_init");
      auto initGroup = calyx::createStaticGroup(
          rewriter, getState<ComponentLoweringState>().getComponentOp(),
          loop->getLoc(), initGroupName, 1);
      getState<ComponentLoweringState>().addLoopInitGroup(LoopWrapper(loop),
                                                          initGroup);

      // Create cond group
      auto groupName = getState<ComponentLoweringState>().getUniqueName("cond");
      auto condGroup = calyx::createStaticGroup(
          rewriter, getState<ComponentLoweringState>().getComponentOp(),
          loop->getLoc(), groupName, 1);
      getState<ComponentLoweringState>().setCondGroup(loop, condGroup);

      rewriter.setInsertionPointToEnd(initGroup.getBodyBlock());
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getIn(),
                                       condValue);
      auto one =
          calyx::createConstant(loop.getLoc(), rewriter, getComponent(), 1, 1);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getWriteEn(),
                                       one);

      auto term =
          cast<LoopScheduleTerminatorOp>(loop.getBodyBlock()->getTerminator());

      auto termArg = term.getIterArgs()[0];
      auto phase = termArg.getDefiningOp<PhaseInterface>();
      auto result = termArg.cast<OpResult>();
      auto *phaseTerm = phase.getBodyBlock().getTerminator();
      auto newIterArg = phaseTerm->getOpOperand(result.getResultNumber()).get();
      Value newCondValue;
      for (auto &op : loop.getConditionBlock()->getOperations()) {
        if (!isa<LoopScheduleRegisterOp>(op)) {
          auto *clonedOp = rewriter.clone(op);
          clonedOp->moveBefore(phaseTerm);
          newCondValue = clonedOp->getResult(0);
        }
      }
      auto condArg = loop.getConditionBlock()->getArgument(0);
      rewriter.replaceUsesWithIf(condArg, newIterArg, [&](OpOperand &operand) {
        return operand.getOwner()->getParentOp() == phase;
      });

      rewriter.setInsertionPointToEnd(condGroup.getBodyBlock());
      assert(newCondValue != nullptr);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getIn(),
                                       newCondValue);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getWriteEn(),
                                       one);

      // Add condition eval to first phase
      getState<ComponentLoweringState>().addBlockSchedulable(
          &phase.getBodyBlock(), condGroup);

      return;
    });
    return success();
  }
};

class BuildStallableConditionChecks
    : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](LoopInterface loop) {
      if (!loop.isPipelined() || !loop.canStall()) {
        return;
      }

      /// Create condition register.
      auto condValue = loop.getConditionValue();
      std::string name = getState<ComponentLoweringState>()
                             .getUniqueName(loop.getOperation())
                             .str() +
                         "_cond";

      auto condReg =
          createRegister(condValue.getLoc(), rewriter, getComponent(), 1, name);
      getState<ComponentLoweringState>().setCondReg(loop, condReg);

      // Create condition init group.
      auto initGroupName =
          getState<ComponentLoweringState>().getUniqueName("cond_init");
      auto initGroup = calyx::createStaticGroup(
          rewriter, getState<ComponentLoweringState>().getComponentOp(),
          loop->getLoc(), initGroupName, 1);
      getState<ComponentLoweringState>().addLoopInitGroup(LoopWrapper(loop),
                                                          initGroup);

      auto pipeline = dyn_cast<LoopSchedulePipelineOp>(loop.getOperation());
      assert(pipeline != nullptr);
      auto bound = pipeline.getBound();
      assert(bound.has_value());
      auto incrGroup =
          getState<ComponentLoweringState>().getIncrGroup(pipeline);
      auto incrVal =
          getState<ComponentLoweringState>().getLoopIterValue(pipeline);

      rewriter.setInsertionPointToEnd(initGroup.getBodyBlock());
      auto one =
          calyx::createConstant(loop.getLoc(), rewriter, getComponent(), 1, 1);
      auto zero =
          calyx::createConstant(loop.getLoc(), rewriter, getComponent(), 1, 0);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getIn(),
                                       bound.value() == 0 ? zero : one);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getWriteEn(),
                                       one);

      auto idxType = incrVal.getType();
      auto bitwidth = idxType.getIntOrFloatBitWidth();
      auto i1Type = rewriter.getI1Type();

      auto ltOp = getState<ComponentLoweringState>()
                      .getNewLibraryOpInstance<calyx::LtLibOp>(
                          rewriter, loop.getLoc(), {idxType, idxType, i1Type});
      rewriter.setInsertionPointToStart(incrGroup.getBodyBlock());
      rewriter.create<calyx::AssignOp>(loop.getLoc(), ltOp.getLeft(), incrVal);
      auto constant = calyx::createConstant(
          loop.getLoc(), rewriter, getComponent(), bitwidth,
          *bound + pipeline.getBodyLatency() - 1);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), ltOp.getRight(),
                                       constant);
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getIn(),
                                       ltOp.getOut());
      rewriter.create<calyx::AssignOp>(loop.getLoc(), condReg.getWriteEn(),
                                       one);
      return;
    });
    return success();
  }
};

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// Maintain a mapping between funcOp input arguments and the port index
    /// which the argument will eventually map to.
    DenseMap<Value, unsigned> funcOpArgRewrites;

    /// Maintain a mapping between funcOp output indexes and the component
    /// output port index which the return value will eventually map to.
    DenseMap<unsigned, unsigned> funcOpResultMapping;

    /// Maintain a mapping between an external memory argument (identified by a
    /// memref) and eventual component input- and output port indices that will
    /// map to the memory ports. The pair denotes the start index of the memory
    /// ports in the in- and output ports of the component. Ports are expected
    /// to be ordered in the same manner as they are added by
    /// calyx::appendPortsForExternalMemref.
    DenseMap<Value, std::pair<unsigned, unsigned>> extMemoryCompPortIndices;

    /// Create I/O ports. Maintain separate in/out port vectors to determine
    /// which port index each function argument will eventually map to.
    SmallVector<calyx::PortInfo> inPorts, outPorts;
    FunctionType funcType = funcOp.getFunctionType();
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (!arg.value().getType().isa<MemRefType>()) {
        /// Single-port arguments
        auto inName = "in" + std::to_string(arg.index());
        funcOpArgRewrites[arg.value()] = inPorts.size();
        inPorts.push_back(calyx::PortInfo{
            rewriter.getStringAttr(inName),
            calyx::convIndexType(rewriter, arg.value().getType()),
            calyx::Direction::Input,
            DictionaryAttr::get(rewriter.getContext(), {})});
      }
    }
    for (auto res : enumerate(funcType.getResults())) {
      funcOpResultMapping[res.index()] = outPorts.size();
      outPorts.push_back(calyx::PortInfo{
          rewriter.getStringAttr("out" + std::to_string(res.index())),
          calyx::convIndexType(rewriter, res.value()), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});
    }

    /// We've now recorded all necessary indices. Merge in- and output ports
    /// and add the required mandatory component ports.
    auto ports = inPorts;
    llvm::append_range(ports, outPorts);
    calyx::addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getSymName()), ports);

    /// Mark this component as the toplevel.
    compOp->setAttr("toplevel", rewriter.getUnitAttr());

    /// Store the function-to-component mapping.
    functionMapping[funcOp] = compOp;
    auto *compState = loweringState().getState<ComponentLoweringState>(compOp);
    compState->setFuncOpResultMapping(funcOpResultMapping);

    /// Rewrite funcOp SSA argument values to the CompOp arguments.
    for (auto &mapping : funcOpArgRewrites)
      mapping.getFirst().replaceAllUsesWith(
          compOp.getArgument(mapping.getSecond()));

    unsigned extMemCounter = 0;
    rewriter.setInsertionPointToStart(compOp.getBodyBlock());
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (auto memtype = arg.value().getType().dyn_cast<MemRefType>()) {
        SmallVector<int64_t> addrSizes;
        SmallVector<int64_t> sizes;
        for (int64_t dim : memtype.getShape()) {
          sizes.push_back(dim);
          addrSizes.push_back(calyx::handleZeroWidth(dim));
        }
        auto memName = "ext_mem_" + std::to_string(extMemCounter);
        auto bitwidth = memtype.getElementType().getIntOrFloatBitWidth();
        auto memoryOp = rewriter.create<calyx::SeqMemoryOp>(
            funcOp.getLoc(), memName, bitwidth, sizes, addrSizes);
        // Externalize top level memories.
        memoryOp->setAttr("external", IntegerAttr::get(rewriter.getI1Type(),
                                                       llvm::APInt(1, 1)));
        compState->registerMemoryInterface(arg.value(),
                                           calyx::MemoryInterface(memoryOp));
        extMemCounter++;
      }
    }

    return success();
  }
};

/// Builds registers for each phase in the program.
class BuildIntermediateRegs : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    DenseMap<Value, calyx::RegisterOp> regMap;
    auto res = funcOp.walk([&](LoopScheduleRegisterOp op) {
      // Condition registers are handled in BuildWhileGroups.
      auto *parent = op->getParentOp();
      auto phase = dyn_cast<PhaseInterface>(parent);
      if (!phase)
        return WalkResult::advance();

      // Create a register for each phase.
      for (auto &operand : op->getOpOperands()) {
        Value value = operand.get();
        unsigned i = operand.getOperandNumber();
        // Iter args are created in BuildWhileGroups, so just mark the iter arg
        // register as the appropriate pipeline register.
        Value phaseResult = phase->getResult(i);
        bool isIterArg = false;
        for (auto &use : phaseResult.getUses()) {
          if (auto term = dyn_cast<LoopScheduleTerminatorOp>(use.getOwner())) {
            if (use.getOperandNumber() < term.getIterArgs().size()) {
              LoopWrapper loop(dyn_cast<LoopInterface>(phase->getParentOp()));
              auto reg = getState<ComponentLoweringState>().getLoopIterReg(
                  loop, use.getOperandNumber());
              getState<ComponentLoweringState>().addPhaseReg(phase, reg, i);
              regMap[phaseResult] = reg;
              isIterArg = true;
            }
          }
        }
        if (isIterArg)
          continue;

        // If value is produced by a sequential op just pass it
        // on to next phase.
        if (auto cell = value.getDefiningOp<calyx::CellInterface>();
            cell && !cell.isCombinational() && !isa<calyx::RegisterOp>(cell)) {
          auto *op = cell.getOperation();
          Value v;
          if (auto mul = dyn_cast<calyx::PipelinedMultLibOp>(op); mul) {
            v = mul.getOut();
          } else if (auto divu = dyn_cast<calyx::SeqDivULibOp>(op); divu) {
            v = divu.getOut();
          } else if (auto seqMem = dyn_cast<calyx::SeqMemoryOp>(op); seqMem) {
            v = seqMem.readData();
          } else if (auto seqMul = dyn_cast<calyx::SeqMultLibOp>(op); seqMul) {
            v = seqMul.getOut();
          } else if (auto seqRemU = dyn_cast<calyx::SeqRemULibOp>(op);
                     seqRemU) {
            v = seqRemU.getOut();
          } else if (auto seqRemS = dyn_cast<calyx::SeqRemSLibOp>(op);
                     seqRemS) {
            v = seqRemS.getOut();
          } else if (auto seqDivS = dyn_cast<calyx::SeqDivSLibOp>(op);
                     seqDivS) {
            v = seqDivS.getOut();
          } else if (auto stallMul = dyn_cast<calyx::StallableMultLibOp>(op);
                     stallMul) {
            v = stallMul.getOut();
          } else {
            funcOp->getParentOfType<ModuleOp>().dump();
            phase.dump();
            op->dump();
            // assert(false && "Unsupported pipelined cell op");
            funcOp.emitOpError("Unsupported pipelined cell op ") << op;
            return WalkResult::interrupt();
          }
          getState<ComponentLoweringState>().addPhaseReg(phase, v, i);
          continue;
        }

        if (isa<memref::LoadOp, calyx::LoadLoweringInterface>(
                value.getDefiningOp())) {
          getState<ComponentLoweringState>().addPhaseReg(phase, value, i);
          continue;
        }

        // Create a register for passing this result to later phases.
        Type resultType = value.getType();
        assert(resultType.isa<IntegerType>() &&
               "unsupported pipeline result type");

        auto name =
            SmallString<20>(getState<ComponentLoweringState>().getUniqueName(
                phase->getParentOfType<LoopInterface>()));
        name += "_";
        name += phase.getRegisterNamePrefix();
        name += "_register_";
        name += std::to_string(i);
        unsigned width = resultType.getIntOrFloatBitWidth();
        auto reg = createRegister(value.getLoc(), rewriter, getComponent(),
                                  width, name);
        getState<ComponentLoweringState>().addPhaseReg(phase, reg, i);
        regMap[phaseResult] = reg;

        // Note that we do not use replace all uses with here as in
        // BuildBasicBlockRegs. Instead, we wait until after BuildOpGroups, and
        // replace all uses inside BuildPipelineGroups, once the pipeline
        // register created here has been assigned to.
      }
      return WalkResult::advance();
    });
    return res.wasInterrupted() ? failure() : success();
  }
};

/// Builds groups for assigning registers for pipeline stages.
class BuildPhaseGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    // Build all phases contained in loops
    // getComponent()->getParentOfType<ModuleOp>().dump();
    auto res = funcOp.walk([&](LoopInterface loop) {
      auto *bodyBlock = loop.getBodyBlock();
      auto condValue = loop.getConditionValue();
      std::optional<calyx::CombGroupOp> condGroup;

      if (!loop.isPipelined()) {
        condGroup = getState<ComponentLoweringState>()
                        .getEvaluatingGroup<calyx::CombGroupOp>(condValue);
      }

      for (auto phase : bodyBlock->getOps<PhaseInterface>()) {
        if (failed(
                buildPhaseGroups(loop, bodyBlock, phase, condGroup, rewriter)))
          return WalkResult::interrupt();
        condGroup = std::nullopt;
      }
      return WalkResult::advance();
    });

    if (res.wasInterrupted())
      return failure();

    // Build groups for all top-level phases, should be only steps
    auto *funcBlock = &funcOp.getBlocks().front();
    for (auto phase : funcOp.getOps<PhaseInterface>()) {
      if (failed(buildPhaseGroups(funcOp, funcBlock, phase, std::nullopt,
                                  rewriter)))
        return failure();
    }

    return success();
  }

  LogicalResult buildPhaseGroups(Operation *op, Block *block,
                                 PhaseInterface phase,
                                 std::optional<calyx::CombGroupOp> condGroup,
                                 PatternRewriter &rewriter) const {
    // Collect pipeline registers for stage.
    auto pipelineRegisters =
        getState<ComponentLoweringState>().getPhaseRegs(phase);
    // Get the number of pipeline stages in the stages block, excluding the
    // terminator. The verifier guarantees there is at least one stage followed
    // by a terminator.
    auto phases = block->getOps<PhaseInterface>();
    size_t numPhases = std::distance(phases.begin(), phases.end());
    assert(numPhases > 0);

    buildPhaseGuards(op, phase, rewriter);
    buildPhaseStallValues(op, phase, rewriter);
    // phase.dump();

    getState<ComponentLoweringState>().addBlockSchedulable(phase->getBlock(),
                                                           phase);

    // Collect group names for the prologue or epilogue.
    SmallVector<StringAttr> bodyGroups;

    auto addBodyGroup = [&](calyx::StaticGroupOp group) {
      // Mark the group for scheduling in the pipeline's block.
      getState<ComponentLoweringState>().addBlockSchedulable(
          &phase.getBodyBlock(), group);

      bodyGroups.push_back(group.getSymNameAttr());
    };

    MutableArrayRef<OpOperand> operands =
        phase.getBodyBlock().getTerminator()->getOpOperands();

    for (auto &operand : operands) {
      unsigned i = operand.getOperandNumber();
      Value value = operand.get();
      // value.dump();

      // Get the pipeline register for that result.
      auto reg = pipelineRegisters[i];

      if (auto *valuePtr = std::get_if<Value>(&reg); valuePtr) {
        auto evaluatingGroup =
            getState<ComponentLoweringState>().getEvaluatingGroup(value);
        assert(isa<calyx::StaticGroupOp>(evaluatingGroup.value()));
        addBodyGroup(dyn_cast<calyx::StaticGroupOp>(
            evaluatingGroup.value().getOperation()));
        phase->getResult(i).replaceAllUsesWith(*valuePtr);
        auto name =
            getState<ComponentLoweringState>().getUniqueName("phase_reg");
        auto newGroup = calyx::createGroup<calyx::CombGroupOp>(
            rewriter, getComponent(), value.getLoc(), name);
        getState<ComponentLoweringState>().registerEvaluatingGroup(value,
                                                                   newGroup);
        continue;
      }

      auto *pipelineRegisterPtr = std::get_if<calyx::RegisterOp>(&reg);
      assert(pipelineRegisterPtr);
      auto pipelineRegister = *pipelineRegisterPtr;

      if (!isa<LoopSchedulePipelineOp>(phase->getParentOp()) &&
          isa<calyx::RegisterOp>(value.getDefiningOp())) {
        phase->getResult(i).replaceAllUsesWith(pipelineRegister.getOut());
        continue;
      }
      // Get the evaluating group for that value.
      auto evaluatingGroup =
          getState<ComponentLoweringState>().getEvaluatingGroup(value);

      if (!evaluatingGroup.has_value()) {
        auto name =
            getState<ComponentLoweringState>().getUniqueName("phase_reg");
        auto newGroup = calyx::createGroup<calyx::CombGroupOp>(
            rewriter, getComponent(), value.getLoc(), name);
        evaluatingGroup = newGroup;
      }

      assert(isa<calyx::CombGroupOp>(evaluatingGroup.value().getOperation()));
      // Stitch the register in, depending on whether the group was
      // combinational or sequential.
      calyx::StaticGroupOp group =
          buildRegisterGroup(phase.getLoc(), pipelineRegister, value, rewriter);

      // Replace the stage result uses with the register out.
      phase->getResult(i).replaceAllUsesWith(pipelineRegister.getOut());

      addBodyGroup(group);
    }

    // If cond group was given, add to phase
    if (condGroup.has_value()) {
    }

    return success();
  }

  calyx::StaticGroupOp buildRegisterGroup(Location loc,
                                          calyx::RegisterOp pipelineRegister,
                                          Value value,
                                          PatternRewriter &rewriter) const {
    // Create a sequential group and replace the comb group.
    PatternRewriter::InsertionGuard g(rewriter);
    auto groupName =
        getState<ComponentLoweringState>().getUniqueName("phase_reg");
    auto group =
        calyx::createStaticGroup(rewriter, getComponent(), loc, groupName, 1);

    // Stitch evaluating group to register.
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, group, getState<ComponentLoweringState>().getComponentOp(),
        pipelineRegister, value);

    // Mark the new group as the evaluating group.
    // for (auto assign : group.getOps<calyx::AssignOp>()) {
    //   auto *src = assign.getSrc().getDefiningOp();
    //   if (!isa<calyx::CellInterface>(*src)
    //       // ||
    //       // dyn_cast<calyx::CellInterface>(*src).isCombinational()
    //       ) {
    //     getState<ComponentLoweringState>().registerEvaluatingGroup(
    //         assign.getSrc(), group);
    //   }
    // }

    return group;
  }

  void buildPhaseGuards(Operation *op, PhaseInterface phase,
                        PatternRewriter &rewriter) const {
    SmallVector<Value> guards;
    if (auto pipeline = dyn_cast<LoopSchedulePipelineOp>(op); pipeline) {
      assert(pipeline.getTripCount().has_value() &&
             "Unbounded pipelines not currently supported");
      auto stage = cast<LoopSchedulePipelineStageOp>(phase);
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(
          getComponent().getWiresOp().getBodyBlock());
      auto startIter = stage.getStartTime().value();
      auto idxValue =
          getState<ComponentLoweringState>().getLoopIterValue(pipeline);
      auto idxType = idxValue.getType();
      auto bitwidth = idxType.getIntOrFloatBitWidth();
      auto i1Type = rewriter.getI1Type();
      auto incrGroup =
          getState<ComponentLoweringState>().getIncrGroup(pipeline);

      if (startIter == 0) {
        // First stage guard
        auto ltOp =
            getState<ComponentLoweringState>()
                .getNewLibraryOpInstance<calyx::LtLibOp>(
                    rewriter, stage.getLoc(), {idxType, idxType, i1Type});
        guards.push_back(ltOp.getOut());

        // Update increment group for upper bound
        rewriter.setInsertionPointToEnd(incrGroup.getBodyBlock());
        rewriter.create<calyx::AssignOp>(stage.getLoc(), ltOp.getLeft(),
                                         idxValue);
        auto endIter = pipeline.getTripCount().value() * pipeline.getII();
        auto ubConst = calyx::createConstant(stage.getLoc(), rewriter,
                                             getComponent(), bitwidth, endIter);
        rewriter.create<calyx::AssignOp>(stage.getLoc(), ltOp.getRight(),
                                         ubConst);
        getState<ComponentLoweringState>().registerEvaluatingGroup(
            ltOp.getOut(), incrGroup);

        // Handle II > 1
        if (pipeline.getII() > 1) {
          // We insert a counter that counts up to II - 1 then resets to zero
          // When the counter reaches II - 1 we trigger the first stage

          // II counter register
          std::string regName =
              getState<ComponentLoweringState>().getUniqueName(
                  "ii_" + std::to_string(pipeline.getII()) + "_counter_reg");
          auto bitwidth = llvm::bit_width(pipeline.getII());
          auto counterReg = createRegister(phase.getLoc(), rewriter,
                                           getComponent(), bitwidth, regName);

          // II counter increment
          auto widthType = rewriter.getIntegerType(bitwidth);
          auto counterAdd = getState<ComponentLoweringState>()
                                .getNewLibraryOpInstance<calyx::AddLibOp>(
                                    rewriter, stage.getLoc(),
                                    {widthType, widthType, widthType});
          rewriter.create<calyx::AssignOp>(stage.getLoc(), counterAdd.getLeft(),
                                           counterReg.getOut());
          auto one = calyx::createConstant(stage.getLoc(), rewriter,
                                           getComponent(), bitwidth, 1);
          rewriter.create<calyx::AssignOp>(stage.getLoc(),
                                           counterAdd.getRight(), one);

          // II counter init group
          std::string groupName =
              getState<ComponentLoweringState>().getUniqueName(
                  "ii_" + std::to_string(pipeline.getII()) + "_counter_init");
          auto iiGroup = calyx::createStaticGroup(rewriter, getComponent(),
                                                  phase.getLoc(), groupName, 1);
          getState<ComponentLoweringState>().addLoopInitGroup(
              LoopWrapper(pipeline), iiGroup);
          {
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToEnd(iiGroup.getBodyBlock());

            // Set II counter to zero before loop runs
            auto zero = calyx::createConstant(stage.getLoc(), rewriter,
                                              getComponent(), bitwidth, 0);
            auto oneI1 = calyx::createConstant(stage.getLoc(), rewriter,
                                               getComponent(), 1, 1);
            rewriter.create<calyx::AssignOp>(stage.getLoc(), counterReg.getIn(),
                                             zero);
            rewriter.create<calyx::AssignOp>(stage.getLoc(),
                                             counterReg.getWriteEn(), oneI1);
          }

          // Check if counter = II - 1
          auto counterEq =
              getState<ComponentLoweringState>()
                  .getNewLibraryOpInstance<calyx::EqLibOp>(
                      rewriter, stage.getLoc(),
                      {widthType, widthType, rewriter.getI1Type()});
          rewriter.create<calyx::AssignOp>(stage.getLoc(), counterEq.getLeft(),
                                           counterReg.getOut());
          auto iiMinusOne =
              calyx::createConstant(stage.getLoc(), rewriter, getComponent(),
                                    bitwidth, pipeline.getII() - 1);
          rewriter.create<calyx::AssignOp>(stage.getLoc(), counterEq.getRight(),
                                           iiMinusOne);

          // If eq assign to zero, otherwise assign to add result
          auto zero = calyx::createConstant(stage.getLoc(), rewriter,
                                            getComponent(), bitwidth, 0);
          rewriter.create<calyx::AssignOp>(stage.getLoc(), counterReg.getIn(),
                                           zero, counterEq.getOut());
          auto oneI1 = calyx::createConstant(stage.getLoc(), rewriter,
                                             getComponent(), 1, 1);
          auto notEq = rewriter.create<comb::XorOp>(stage.getLoc(),
                                                    counterEq.getOut(), oneI1);
          rewriter.create<calyx::AssignOp>(stage.getLoc(), counterReg.getIn(),
                                           counterAdd.getOut(),
                                           notEq.getResult());
          rewriter.create<calyx::AssignOp>(stage.getLoc(),
                                           counterReg.getWriteEn(), oneI1);

          // Add eq result to guard values
          guards.push_back(counterEq.getOut());
        }
      } else {
        // Pass guards to later stages
        auto prevPhase = cast<PhaseInterface>(phase->getPrevNode());
        assert(prevPhase != nullptr);
        auto startTimeDiff =
            phase.getStartTime().value() - prevPhase.getStartTime().value();
        auto prevReg =
            getState<ComponentLoweringState>().getGuardRegister(prevPhase);
        assert(prevReg.has_value());
        for (unsigned i = 0; i < startTimeDiff - 1; ++i) {
          auto regName =
              getState<ComponentLoweringState>().getUniqueName("guard_reg");
          auto reg = createRegister(phase.getLoc(), rewriter, getComponent(), 1,
                                    regName);
          rewriter.setInsertionPointToEnd(incrGroup.getBodyBlock());
          rewriter.create<calyx::AssignOp>(stage.getLoc(), reg.getIn(),
                                           prevReg.value().getOut());
          auto oneI1 = calyx::createConstant(stage.getLoc(), rewriter,
                                             getComponent(), 1, 1);
          rewriter.create<calyx::AssignOp>(stage.getLoc(), reg.getWriteEn(),
                                           oneI1);
          prevReg = reg;
        }
        guards.push_back(prevReg.value().getOut());
      }

      // Create init group for guard
      std::string groupName =
          getState<ComponentLoweringState>().getUniqueName("guard_init");
      auto guardGroup = calyx::createStaticGroup(rewriter, getComponent(),
                                                 phase.getLoc(), groupName, 1);
      getState<ComponentLoweringState>().addLoopInitGroup(LoopWrapper(pipeline),
                                                          guardGroup);
      std::string regName =
          getState<ComponentLoweringState>().getUniqueName("guard_reg");
      auto reg =
          createRegister(phase.getLoc(), rewriter, getComponent(), 1, regName);
      // Store guard register for passing to future phases
      getState<ComponentLoweringState>().setGuardRegister(phase, reg);
      rewriter.setInsertionPointToEnd(guardGroup.getBodyBlock());
      auto zeroI1 =
          calyx::createConstant(stage.getLoc(), rewriter, getComponent(), 1, 0);
      auto oneI1 =
          calyx::createConstant(stage.getLoc(), rewriter, getComponent(), 1, 1);
      // Stages with a start time of zero must have their lower bound guard
      // initialized to 1
      rewriter.create<calyx::AssignOp>(stage.getLoc(), reg.getIn(),
                                       startIter == 0 ? oneI1 : zeroI1);
      rewriter.create<calyx::AssignOp>(stage.getLoc(), reg.getWriteEn(), oneI1);
      getState<ComponentLoweringState>().registerEvaluatingGroup(reg.getOut(),
                                                                 incrGroup);
      getState<ComponentLoweringState>().registerEvaluatingGroup(reg.getDone(),
                                                                 incrGroup);
      getState<ComponentLoweringState>().setGuardValue(phase, reg.getOut());

      // Update incr group for guard
      rewriter.setInsertionPointToEnd(incrGroup.getBodyBlock());

      auto guardVal = calyx::buildCombAndTree(
          rewriter, getState<ComponentLoweringState>(), stage.getLoc(), guards);
      rewriter.create<calyx::AssignOp>(stage.getLoc(), reg.getIn(), guardVal);
      rewriter.create<calyx::AssignOp>(stage.getLoc(), reg.getWriteEn(), oneI1);
    }
  }

  void buildPhaseStallValues(Operation *op, PhaseInterface phase,
                             PatternRewriter &rewriter) const {
    auto pipeline = dyn_cast<LoopSchedulePipelineOp>(op);
    if (!pipeline) {
      return;
    }
    auto stallValue =
        getState<ComponentLoweringState>().getStallValue(pipeline);
    auto guardVal = getState<ComponentLoweringState>().getGuardValue(phase);
    assert(guardVal.has_value());
    auto doneVals =
        getState<ComponentLoweringState>().getPhaseDoneValues(phase);

    if (!doneVals.empty()) {
      std::optional<Value> notDoneValue;
      auto i1Type = rewriter.getI1Type();
      rewriter.setInsertionPointToStart(getState<ComponentLoweringState>()
                                            .getComponentOp()
                                            .getWiresOp()
                                            .getBodyBlock());
      for (auto doneVal : doneVals) {
        auto notOp = getState<ComponentLoweringState>()
                         .getNewLibraryOpInstance<calyx::NotLibOp>(
                             rewriter, phase.getLoc(), {i1Type, i1Type});
        rewriter.create<calyx::AssignOp>(phase.getLoc(), notOp.getIn(),
                                         doneVal);
        if (notDoneValue.has_value()) {
          auto orOp =
              getState<ComponentLoweringState>()
                  .getNewLibraryOpInstance<calyx::OrLibOp>(
                      rewriter, phase.getLoc(), {i1Type, i1Type, i1Type});
          rewriter.create<calyx::AssignOp>(phase.getLoc(), orOp.getLeft(),
                                           notOp.getOut());
          rewriter.create<calyx::AssignOp>(phase.getLoc(), orOp.getRight(),
                                           *notDoneValue);
          notDoneValue = orOp.getOut();
        } else {
          notDoneValue = notOp.getOut();
        }
      }

      auto andOp = getState<ComponentLoweringState>()
                       .getNewLibraryOpInstance<calyx::AndLibOp>(
                           rewriter, phase.getLoc(), {i1Type, i1Type, i1Type});
      rewriter.create<calyx::AssignOp>(phase.getLoc(), andOp.getLeft(),
                                       *guardVal);
      rewriter.create<calyx::AssignOp>(phase.getLoc(), andOp.getRight(),
                                       *notDoneValue);

      if (stallValue.has_value()) {
        auto orOp = getState<ComponentLoweringState>()
                        .getNewLibraryOpInstance<calyx::OrLibOp>(
                            rewriter, phase.getLoc(), {i1Type, i1Type, i1Type});
        rewriter.create<calyx::AssignOp>(phase.getLoc(), orOp.getLeft(),
                                         andOp.getOut());
        rewriter.create<calyx::AssignOp>(phase.getLoc(), orOp.getRight(),
                                         *stallValue);
        getState<ComponentLoweringState>().setStallValue(pipeline,
                                                         orOp.getOut());
      } else {
        getState<ComponentLoweringState>().setStallValue(pipeline,
                                                         andOp.getOut());
      }
    }

    if (phase->getNextNode() == nullptr)
      return;

    auto nextPhase = dyn_cast<PhaseInterface>(phase->getNextNode());

    phase.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<calyx::LoadLoweringInterface>(op)) {
        auto done = getState<ComponentLoweringState>()
                        .getMemoryInterface(loadOp.getMemoryValue())
                        .readDoneOpt();
        if (done.has_value())
          getState<ComponentLoweringState>().addPhaseDoneValue(nextPhase,
                                                               *done);
      } else if (auto storeOp = dyn_cast<calyx::LoadLoweringInterface>(op)) {
        auto done = getState<ComponentLoweringState>()
                        .getMemoryInterface(storeOp.getMemoryValue())
                        .writeDoneOpt();
        if (done.has_value())
          getState<ComponentLoweringState>().addPhaseDoneValue(nextPhase,
                                                               *done);
      }
    });
  }
};

/// Builds a control schedule by traversing the CFG of the function and
/// associating this with the previously created groups.
/// For simplicity, the generated control flow is expanded for all possible
/// paths in the input DAG. This elaborated control flow is later reduced in
/// the runControlFlowSimplification passes.
class BuildControl : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();
    rewriter.setInsertionPointToStart(
        getComponent().getControlOp().getBodyBlock());
    auto topLevelSeqOp = rewriter.create<calyx::SeqOp>(funcOp.getLoc());
    DenseSet<Block *> path;
    return buildCFGControl(path, rewriter, topLevelSeqOp.getBodyBlock(),
                           nullptr, entryBlock);
  }

private:
  LogicalResult buildCFGControl(DenseSet<Block *> path,
                                PatternRewriter &rewriter,
                                mlir::Block *parentCtrlBlock,
                                mlir::Block *preBlock,
                                mlir::Block *block) const {
    if (path.count(block) != 0)
      return preBlock->getTerminator()->emitError()
             << "CFG backedge detected. Loops must be raised to 'scf.while' or "
                "'scf.for' operations.";

    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    LogicalResult bbSchedResult =
        scheduleBasicBlock(rewriter, path, parentCtrlBlock, block);
    if (bbSchedResult.failed())
      return bbSchedResult;

    path.insert(block);
    auto successors = block->getSuccessors();
    auto nSuccessors = successors.size();
    if (nSuccessors > 0) {
      auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator());
      assert(brOp);
      if (nSuccessors > 1) {
        assert(false);
        //   /// TODO(mortbopet): we could choose to support ie. std.switch, but
        //   it
        //   /// would probably be easier to just require it to be lowered
        //   /// beforehand.
        //   assert(nSuccessors == 2 &&
        //          "only conditional branches supported for now...");
        //   /// Wrap each branch inside an if/else.
        //   auto cond = brOp->getOperand(0);
        //   auto condGroup = getState<ComponentLoweringState>()
        //                        .getEvaluatingGroup<calyx::CombGroupOp>(cond);
        //   auto symbolAttr = FlatSymbolRefAttr::get(
        //       StringAttr::get(getContext(), condGroup.getSymName()));

        //   auto ifOp = rewriter.create<calyx::IfOp>(
        //       brOp->getLoc(), cond, symbolAttr, /*initializeElseBody=*/true);
        //   rewriter.setInsertionPointToStart(ifOp.getThenBody());
        //   auto thenSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());
        //   rewriter.setInsertionPointToStart(ifOp.getElseBody());
        //   auto elseSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());

        //   bool trueBrSchedSuccess =
        //       schedulePath(rewriter, path, brOp.getLoc(), block,
        //       successors[0],
        //                    thenSeqOp.getBodyBlock())
        //           .succeeded();
        //   bool falseBrSchedSuccess = true;
        //   if (trueBrSchedSuccess) {
        //     falseBrSchedSuccess =
        //         schedulePath(rewriter, path, brOp.getLoc(), block,
        //         successors[1],
        //                      elseSeqOp.getBodyBlock())
        //             .succeeded();
        //   }

        //   return success(trueBrSchedSuccess && falseBrSchedSuccess);
      }
      /// Schedule sequentially within the current parent control block.
      return schedulePath(rewriter, path, brOp.getLoc(), block,
                          successors.front(), parentCtrlBlock);
    }
    return success();
  }

  /// Sequentially schedules the groups that registered themselves with
  /// 'block'.
  LogicalResult scheduleBasicBlock(PatternRewriter &rewriter,
                                   DenseSet<Block *> &path,
                                   mlir::Block *parentCtrlBlock,
                                   mlir::Block *block) const {
    auto compBlockSchedulables =
        getState<ComponentLoweringState>().getBlockSchedulables(block);
    auto loc = block->front().getLoc();

    // if (isa<FuncOp>(block->getParentOp())) {
    //   auto seqOp = rewriter.create<calyx::StaticSeqOp>(loc);
    //   parentCtrlBlock = seqOp.getBodyBlock();
    // }
    for (auto &sched : compBlockSchedulables) {
      rewriter.setInsertionPointToEnd(parentCtrlBlock);
      if (auto *groupPtr = std::get_if<calyx::StaticGroupOp>(&sched);
          groupPtr) {
        rewriter.create<calyx::EnableOp>(groupPtr->getLoc(),
                                         groupPtr->getSymName());
      } else if (auto *phasePtr = std::get_if<PhaseInterface>(&sched);
                 phasePtr) {
        auto &phaseOp = *phasePtr;
        auto guardValue =
            getState<ComponentLoweringState>().getGuardValue(phaseOp);
        if (guardValue.has_value()) {
          auto val = guardValue.value();
          auto ifOp = rewriter.create<calyx::StaticIfOp>(phaseOp.getLoc(), val);
          rewriter.setInsertionPointToEnd(ifOp.getBodyBlock());
        }
        Block *bodyBlock;
        if (phaseOp.isStatic()) {
          auto op = rewriter.create<calyx::StaticParOp>(phaseOp.getLoc());
          bodyBlock = op.getBodyBlock();
        } else {
          auto op = rewriter.create<calyx::ParOp>(phaseOp.getLoc());
          bodyBlock = op.getBodyBlock();
        }
        rewriter.setInsertionPointToEnd(bodyBlock);

        path.insert(&phaseOp.getBodyBlock());
        auto res = scheduleBasicBlock(rewriter, path, bodyBlock,
                                      &phaseOp.getBodyBlock());
        if (res.failed())
          phaseOp->emitOpError("Failed to schedule phase op block");
      } else if (auto *loopSchedPtr = std::get_if<LoopWrapper>(&sched);
                 loopSchedPtr) {
        auto &loopOp = *loopSchedPtr;

        auto loopParentCtrlOp = rewriter.create<calyx::SeqOp>(loc);
        rewriter.setInsertionPointToEnd(loopParentCtrlOp.getBodyBlock());
        auto initGroups =
            getState<ComponentLoweringState>().getLoopInitGroups(loopOp);
        auto *loopCtrlOp = buildLoopCtrlOp(loopOp, initGroups, rewriter);
        rewriter.setInsertionPointToEnd(&loopCtrlOp->getRegion(0).front());
        Block *loopBodyOpBlock;
        if (loopOp.isPipelined()) {
          auto loopBodyOp = rewriter.create<calyx::StaticParOp>(
              loopOp.getOperation()->getLoc());
          rewriter.setInsertionPointToEnd(loopBodyOp.getBodyBlock());
          auto pipeline = cast<LoopSchedulePipelineOp>(loopOp.getOperation());
          auto incrGroup =
              getState<ComponentLoweringState>().getIncrGroup(pipeline);
          rewriter.create<calyx::EnableOp>(loopOp.getLoc(),
                                           incrGroup.getSymName());
          loopBodyOpBlock = loopBodyOp.getBodyBlock();
        } else {
          auto loopBodyOp =
              rewriter.create<calyx::SeqOp>(loopOp.getOperation()->getLoc());
          rewriter.setInsertionPointToEnd(loopBodyOp.getBodyBlock());
          loopBodyOpBlock = loopBodyOp.getBodyBlock();
        }

        /// Only schedule the 'after' block. The 'before' block is
        /// implicitly scheduled when evaluating the while condition.
        LogicalResult res = buildCFGControl(path, rewriter, loopBodyOpBlock,
                                            block, loopOp.getBodyBlock());

        rewriter.setInsertionPointAfter(loopParentCtrlOp);
        if (res.failed())
          return loopOp.getOperation()->emitError("Cannot schedule loop body");
      } else
        llvm_unreachable("Unknown schedulable");
    }
    return success();
  }

  /// Schedules a block by inserting a branch argument assignment block (if any)
  /// before recursing into the scheduling of the block innards.
  /// Blocks 'from' and 'to' refer to blocks in the source program.
  /// parentCtrlBlock refers to the control block wherein control operations are
  /// to be inserted.
  LogicalResult schedulePath(PatternRewriter &rewriter,
                             const DenseSet<Block *> &path, Location loc,
                             Block *from, Block *to,
                             Block *parentCtrlBlock) const {
    /// Schedule any registered block arguments to be executed before the body
    /// of the branch.
    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    auto preSeqOp = rewriter.create<calyx::SeqOp>(loc);
    rewriter.setInsertionPointToEnd(preSeqOp.getBodyBlock());
    for (auto barg :
         getState<ComponentLoweringState>().getBlockArgGroups(from, to))
      rewriter.create<calyx::EnableOp>(barg.getLoc(), barg.symName());

    return buildCFGControl(path, rewriter, parentCtrlBlock, from, to);
  }

  Operation *
  buildLoopCtrlOp(LoopWrapper loopOp,
                  const SmallVector<calyx::GroupInterface> &initGroups,
                  PatternRewriter &rewriter) const {
    Location loc = loopOp.getLoc();

    /// Insert while iter arg initialization group(s). Emit a
    /// parallel group to assign one or more registers all at once.
    calyx::StaticSeqOp seqOp;
    {
      PatternRewriter::InsertionGuard g(rewriter);
      seqOp = rewriter.create<calyx::StaticSeqOp>(loc);
      rewriter.setInsertionPointToEnd(seqOp.getBodyBlock());
      auto parOp = rewriter.create<calyx::StaticParOp>(loc);
      rewriter.setInsertionPointToEnd(parOp.getBodyBlock());
      for (calyx::GroupInterface group : initGroups)
        rewriter.create<calyx::EnableOp>(group.getLoc(), group.symName());
    }

    /// Check if loop is a pipeline with trip count
    if (isa<LoopSchedulePipelineOp>(loopOp.getOperation()) &&
        loopOp.getBound().has_value() && !loopOp.getOperation().canStall()) {
      // Can use repeat op instead of while op
      auto pipeline = cast<LoopSchedulePipelineOp>(loopOp.getOperation());
      auto bound = loopOp.getBound().value() * pipeline.getII();
      auto iterCount = bound + pipeline.getBodyLatency() - 1;
      auto repeatCtrlOp =
          rewriter.create<calyx::StaticRepeatOp>(loc, iterCount);
      return repeatCtrlOp;
    }

    /// Get condition for while loop
    auto cond = getState<ComponentLoweringState>()
                    .getCondReg(loopOp.getOperation())
                    .getOut();

    /// Build WhileOp with condition
    auto whileCtrlOp = rewriter.create<calyx::WhileOp>(loc, cond);

    if (isa<LoopSchedulePipelineOp>(loopOp.getOperation()) &&
        loopOp.getOperation().canStall()) {
      auto cond = getState<ComponentLoweringState>().getStallValue(
          loopOp.getOperation());
      if (cond.has_value()) {
        rewriter.setInsertionPointToStart(getState<ComponentLoweringState>()
                                              .getComponentOp()
                                              .getWiresOp()
                                              .getBodyBlock());
        auto i1Type = rewriter.getI1Type();
        auto notOp = getState<ComponentLoweringState>()
                         .getNewLibraryOpInstance<calyx::NotLibOp>(
                             rewriter, loc, {i1Type, i1Type});
        rewriter.create<calyx::AssignOp>(loc, notOp.getIn(), *cond);
        auto stallPorts = getState<ComponentLoweringState>().getStallPorts(
            loopOp.getOperation());
        for (auto port : stallPorts) {
          rewriter.create<calyx::AssignOp>(loc, port, *cond);
        }
        rewriter.setInsertionPointToEnd(whileCtrlOp.getBodyBlock());
        auto ifCtrlOp = rewriter.create<calyx::StaticIfOp>(loc, notOp.getOut());
        return ifCtrlOp;
      }
    }
    return whileCtrlOp;
  }
};

/// LateSSAReplacement contains various functions for replacing SSA values that
/// were not replaced during op construction.
class LateSSAReplacement : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult partiallyLowerFuncToComp(FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](memref::LoadOp loadOp) {
      /// In buildOpGroups we did not replace loadOp's results, to ensure a
      /// link between evaluating groups (which fix the input addresses of a
      /// memory op) and a readData result. Now, we may replace these SSA
      /// values with their memoryOp readData output.
      loadOp.getResult().replaceAllUsesWith(
          getState<ComponentLoweringState>()
              .getMemoryInterface(loadOp.getMemref())
              .readData());
    });

    funcOp.walk([&](calyx::LoadLoweringInterface loadOp) {
      /// In buildOpGroups we did not replace loadOp's results, to ensure a
      /// link between evaluating groups (which fix the input addresses of a
      /// memory op) and a readData result. Now, we may replace these SSA
      /// values with their memoryOp readData output.
      loadOp.getResult().replaceAllUsesWith(
          getState<ComponentLoweringState>()
              .getMemoryInterface(loadOp.getMemoryValue())
              .readData());
    });

    return success();
  }
};

class ZeroUnusedMemoryEnables : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {

    DenseSet<Value> alreadyAssigned;
    auto compOp = getState<ComponentLoweringState>().getComponentOp();
    auto wiresOp = compOp.getWiresOp();
    rewriter.setInsertionPointToStart(wiresOp.getBodyBlock());

    auto zero = calyx::createConstant(funcOp.getLoc(), rewriter, compOp, 1, 0);
    auto readEnNotSet =
        getState<ComponentLoweringState>().interfacesReadEnNotSet();
    for (auto interface : readEnNotSet) {
      auto readEn = interface.readEn();
      if (alreadyAssigned.count(readEn) == 0) {
        rewriter.create<calyx::AssignOp>(funcOp.getLoc(), readEn, zero);
        alreadyAssigned.insert(readEn);
      }
    }

    auto writeEnNotSet =
        getState<ComponentLoweringState>().interfacesWriteEnNotSet();
    for (auto interface : writeEnNotSet) {
      auto writeEn = interface.writeEn();
      if (alreadyAssigned.count(writeEn) == 0) {
        rewriter.create<calyx::AssignOp>(funcOp.getLoc(), writeEn, zero);
        alreadyAssigned.insert(writeEn);
      }
    }

    return success();
  }
};

/// Erases FuncOp operations.
class CleanupFuncOps : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class LoopScheduleToCalyxPass
    : public LoopScheduleToCalyxBase<LoopScheduleToCalyxPass> {
public:
  LoopScheduleToCalyxPass()
      : LoopScheduleToCalyxBase<LoopScheduleToCalyxPass>(),
        partialPatternRes(success()) {}
  void runOnOperation() override;

  LogicalResult setTopLevelFunction(mlir::ModuleOp moduleOp,
                                    std::string &topLevelFunction) {
    if (!topLevelFunctionOpt.empty()) {
      if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunctionOpt) ==
          nullptr) {
        moduleOp.emitError() << "Top level function '" << topLevelFunctionOpt
                             << "' not found in module.";
        return failure();
      }
      topLevelFunction = topLevelFunctionOpt;
    } else {
      /// No top level function set; infer top level if the module only contains
      /// a single function, else, throw error.
      auto funcOps = moduleOp.getOps<FuncOp>();
      if (std::distance(funcOps.begin(), funcOps.end()) == 1)
        topLevelFunction = (*funcOps.begin()).getSymName().str();
      else {
        moduleOp.emitError()
            << "Module contains multiple functions, but no top level "
               "function was set. Please see --top-level-function";
        return failure();
      }
    }
    return success();
  }

  struct LoweringPattern {
    enum class Strategy { Once, Greedy };
    RewritePatternSet pattern;
    Strategy strategy;
  };

  //// Labels the entry point of a Calyx program.
  /// Furthermore, this function performs validation on the input function,
  /// to ensure that we've implemented the capabilities necessary to convert
  /// it.
  LogicalResult labelEntryPoint(StringRef topLevelFunction) {
    // Program legalization - the partial conversion driver will not run
    // unless some pattern is provided - provide a dummy pattern.
    struct DummyPattern : public OpRewritePattern<mlir::ModuleOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(mlir::ModuleOp,
                                    PatternRewriter &) const override {
        return failure();
      }
    };

    ConversionTarget target(getContext());
    target.addLegalDialect<calyx::CalyxDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<hw::HWDialect>();
    target.addIllegalDialect<comb::CombDialect>();

    // For loops should have been lowered to while loops
    target.addIllegalOp<scf::ForOp>();

    // Only accept std operations which we've added lowerings for
    target.addIllegalDialect<FuncDialect>();
    target.addIllegalDialect<ArithDialect>();
    target.addLegalOp<AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp, AndIOp,
                      XOrIOp, OrIOp, ExtUIOp, TruncIOp, CondBranchOp, BranchOp,
                      MulIOp, DivUIOp, DivSIOp, RemUIOp, RemSIOp, DivSIOp,
                      ReturnOp, arith::ConstantOp, IndexCastOp, FuncOp,
                      ExtSIOp>();

    RewritePatternSet legalizePatterns(&getContext());
    legalizePatterns.add<DummyPattern>(&getContext());
    DenseSet<Operation *> legalizedOps;
    if (applyPartialConversion(getOperation(), target,
                               std::move(legalizePatterns))
            .failed())
      return failure();

    // Program conversion
    return calyx::applyModuleOpConversion(getOperation(), topLevelFunction);
  }

  /// 'Once' patterns are expected to take an additional LogicalResult&
  /// argument, to forward their result state (greedyPatternRewriteDriver
  /// results are skipped for Once patterns).
  template <typename TPattern, typename... PatternArgs>
  void addOncePattern(SmallVectorImpl<LoweringPattern> &patterns,
                      PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), partialPatternRes, args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Once});
  }

  template <typename TPattern, typename... PatternArgs>
  void addGreedyPattern(SmallVectorImpl<LoweringPattern> &patterns,
                        PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Greedy});
  }

  LogicalResult runPartialPattern(RewritePatternSet &pattern, bool runOnce) {
    assert(pattern.getNativePatterns().size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    if (runOnce)
      config.maxIterations = 1;

    /// Can't return applyPatternsAndFoldGreedily. Root isn't
    /// necessarily erased so it will always return failed(). Instead,
    /// forward the 'succeeded' value from PartialLoweringPatternBase.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(pattern),
                                       config);
    return partialPatternRes;
  }

private:
  LogicalResult partialPatternRes;
  std::shared_ptr<calyx::CalyxLoweringState> loweringState = nullptr;
};

void LoopScheduleToCalyxPass::runOnOperation() {
  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  loweringState.reset();
  partialPatternRes = LogicalResult::failure();

  std::string topLevelFunction;
  if (failed(setTopLevelFunction(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  if (failed(labelEntryPoint(topLevelFunction))) {
    signalPassFailure();
    return;
  }
  loweringState = std::make_shared<calyx::CalyxLoweringState>(getOperation(),
                                                              topLevelFunction);

  /// --------------------------------------------------------------------------
  /// If you are a developer, it may be helpful to add a
  /// 'getOperation()->dump()' call after the execution of each stage to
  /// view the transformations that's going on.
  /// --------------------------------------------------------------------------

  /// A mapping is maintained between a function operation and its corresponding
  /// Calyx component.
  DenseMap<FuncOp, calyx::ComponentOp> funcMap;
  SmallVector<LoweringPattern, 8> loweringPatterns;
  calyx::PatternApplicationState patternState;

  /// Creates a new Calyx component for each FuncOp in the input module.
  addOncePattern<FuncOpConversion>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pattern converts all index typed values to an i32 integer.
  addOncePattern<calyx::ConvertIndexTypes>(loweringPatterns, patternState,
                                           funcMap, *loweringState);

  /// This pattern creates registers for all basic-block arguments.
  addOncePattern<calyx::BuildBasicBlockRegs>(loweringPatterns, patternState,
                                             funcMap, *loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<calyx::BuildReturnRegs>(loweringPatterns, patternState,
                                         funcMap, *loweringState);

  /// This pattern .
  addOncePattern<BuildConditionChecks>(loweringPatterns, patternState, funcMap,
                                       *loweringState);

  /// This pattern converts operations within basic blocks to Calyx library
  /// operators. Combinational operations are assigned inside a
  /// calyx::CombGroupOp, and sequential inside calyx::StaticGroupOps.
  /// Sequential groups are registered with the Block* of which the operation
  /// originated from. This is used during control schedule generation. By
  /// having a distinct group for each operation, groups are analogous to SSA
  /// values in the source program.
  addOncePattern<BuildOpGroups>(loweringPatterns, patternState, funcMap,
                                *loweringState);

  addOncePattern<BuildStallableConditionChecks>(loweringPatterns, patternState,
                                                funcMap, *loweringState);

  /// This pattern creates registers for all pipeline stages.
  addOncePattern<BuildIntermediateRegs>(loweringPatterns, patternState, funcMap,
                                        *loweringState);

  /// This pattern creates groups for all pipeline stages.
  addOncePattern<BuildPhaseGroups>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pattern traverses the CFG of the program and generates a control
  /// schedule based on the calyx::StaticGroupOp's which were registered for
  /// each basic block in the source function.
  addOncePattern<BuildControl>(loweringPatterns, patternState, funcMap,
                               *loweringState);

  /// This pass recursively inlines use-def chains of combinational logic (from
  /// non-stateful groups) into groups referenced in the control schedule.
  addOncePattern<calyx::InlineCombGroups>(loweringPatterns, patternState,
                                          *loweringState);

  /// This pattern performs various SSA replacements that must be done
  /// after control generation.
  addOncePattern<LateSSAReplacement>(loweringPatterns, patternState, funcMap,
                                     *loweringState);

  addOncePattern<ZeroUnusedMemoryEnables>(loweringPatterns, patternState,
                                          funcMap, *loweringState);

  /// Eliminate any unused combinational groups. This is done before
  /// calyx::RewriteMemoryAccesses to avoid inferring slice components for
  /// groups that will be removed.
  addGreedyPattern<calyx::EliminateUnusedCombGroups>(loweringPatterns);

  /// This pattern rewrites accesses to memories which are too wide due to
  /// index types being converted to a fixed-width integer type.
  addOncePattern<calyx::RewriteMemoryAccesses>(loweringPatterns, patternState,
                                               *loweringState);

  /// This pattern removes the source FuncOp which has now been converted into
  /// a Calyx component.
  addOncePattern<CleanupFuncOps>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  /// Sequentially apply each lowering pattern.
  for (auto &pat : loweringPatterns) {
    LogicalResult partialPatternRes = runPartialPattern(
        pat.pattern,
        /*runOnce=*/pat.strategy == LoweringPattern::Strategy::Once);
    if (succeeded(partialPatternRes)) {
      continue;
    }
    signalPassFailure();
    return;
  }

  //===----------------------------------------------------------------------===//
  // Cleanup patterns
  //===----------------------------------------------------------------------===//
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<calyx::MultipleGroupDonePattern,
                      calyx::NonTerminatingGroupDonePattern>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }

  if (ciderSourceLocationMetadata) {
    // Debugging information for the Cider debugger.
    // Reference: https://docs.calyxir.org/debug/cider.html
    SmallVector<Attribute, 16> sourceLocations;
    getOperation()->walk([&](calyx::ComponentOp component) {
      return getCiderSourceLocationMetadata(component, sourceLocations);
    });

    MLIRContext *context = getOperation()->getContext();
    getOperation()->setAttr("calyx.metadata",
                            ArrayAttr::get(context, sourceLocations));
  }
}

} // namespace loopscheduletocalyx

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createLoopScheduleToCalyxPass() {
  return std::make_unique<loopscheduletocalyx::LoopScheduleToCalyxPass>();
}

} // namespace circt
