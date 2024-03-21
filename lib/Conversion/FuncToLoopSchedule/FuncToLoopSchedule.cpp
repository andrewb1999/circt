//===- FuncToLoopSchedule.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/FuncToLoopSchedule.h"
#include "../PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Dialect/SSP/SSPInterfaces.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <limits>
#include <math.h>
#include <optional>
#include <queue>
#include <string>
#include <utility>

#define DEBUG_TYPE "affine-to-loopschedule"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::loopschedule;

namespace {

struct FuncToLoopSchedule
    : public FuncToLoopScheduleBase<FuncToLoopSchedule> {
  FuncToLoopSchedule(bool disableBitwidthMinimization) {
    this->disableBitwidthMinimization = disableBitwidthMinimization;
  }
  void runOnOperation() override;

private:
  LogicalResult createFuncLoopSchedule(FuncOp &funcOp,
                                       SharedOperatorsProblem &problem);

  std::optional<MemoryDependenceAnalysis> dependenceAnalysis;
};

} // namespace

void FuncToLoopSchedule::runOnOperation() {
  // Get dependence analysis for the whole function.
  dependenceAnalysis = getAnalysis<MemoryDependenceAnalysis>();

  if (failed(postLoweringOptimizations(getContext(), getOperation())))
    return signalPassFailure();

  if (failed(replaceMemoryAccesses(getContext(), getOperation(),
                                   *dependenceAnalysis)))
    return signalPassFailure();

  if (!disableBitwidthMinimization) {
    if (failed(bitwidthMinimization(getContext(), getOperation(),
                                    *dependenceAnalysis)))
      return signalPassFailure();
  }

  // Schedule whole function
  auto funcOp = cast<FuncOp>(getOperation());
  auto problem = getSharedOperatorsProblem(funcOp, *dependenceAnalysis);

  // Populate the target operator types.
  if (failed(populateOperatorTypes(funcOp.getOperation(), funcOp.getBody(),
                                   problem)))
    return signalPassFailure();

  // Solve the scheduling problem computed by the analysis.
  if (failed(solveSharedOperatorsProblem(funcOp.getBody(), problem)))
    return signalPassFailure();

  // Convert the IR.
  if (failed(createFuncLoopSchedule(funcOp, problem)))
    return signalPassFailure();

  // getOperation().dump();
}

static int64_t opOrParentStartTime(Problem &problem, Operation *op) {
  Operation *currentOp = op;

  while (!isa<func::FuncOp>(currentOp)) {
    if (problem.hasOperation(currentOp)) {
      return problem.getStartTime(currentOp).value();
    }
    currentOp = currentOp->getParentOp();
  }
  op->emitOpError("Operation or parent does not have start time");
  return -1;
}

/// Create the loopschedule ops for an entire function.
LogicalResult
FuncToLoopSchedule::createFuncLoopSchedule(FuncOp &funcOp,
                                             SharedOperatorsProblem &problem) {
  auto *anchor = funcOp.getBody().back().getTerminator();

  auto opMap = getOperationCycleMap(problem);

  // auto outerLoop = loopNest.front();
  // auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(funcOp.getLoc(), funcOp);

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // for (size_t i = 0; i < iterArgs.size(); ++i)
  //   valueMap.map(whileOp.getBefore().getArgument(i),
  //                stgWhile.getCondBlock().getArgument(i));

  builder.setInsertionPointToStart(&funcOp.getBody().front());

  // auto condConst = builder.create<arith::ConstantOp>(whileOp.getLoc(),
  // builder.getIntegerAttr(builder.getI1Type(), 1)); auto *conditionReg =
  // stgWhile.getCondBlock().getTerminator(); conditionReg->insertOperands(0,
  // condConst.getResult()); for (auto &op :
  // whileOp.getBefore().front().getOperations()) {
  //   if (isa<scf::ConditionOp>(op)) {
  //     auto condOp = cast<scf::ConditionOp>(op);
  //     auto cond = condOp.getCondition();
  //     auto condNew = valueMap.lookupOrNull(cond);
  //     assert(condNew);
  //     conditionReg->insertOperands(0, condNew);
  //   } else {
  //     auto *newOp = builder.clone(op, valueMap);
  //     for (size_t i = 0; i < newOp->getNumResults(); ++i) {
  //       auto newValue = newOp->getResult(i);
  //       auto oldValue = op.getResult(i);
  //       valueMap.map(oldValue, newValue);
  //     }
  //   }
  // }

  // builder.setInsertionPointToStart(&stgWhile.getScheduleBlock());

  // auto termConst = builder.create<arith::ConstantOp>(whileOp.getLoc(),
  // builder.getIndexAttr(1)); auto term = stgWhile.getTerminator();
  // term.getIterArgsMutable().append(termConst.getResult());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp, func::ReturnOp, memref::AllocaOp,
            arith::ConstantOp, memref::AllocOp, AllocInterface>(op))
      continue;
    if (auto schedOp = dyn_cast<SchedulableInterface>(op)) {
      if (schedOp.isInitOp())
        continue;
    }
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // for (auto *op : problem.getOperations()) {
  //   if (isa<memref::AllocaOp>(op)) {
  //     op.
  //   }
  // }

  SmallVector<SmallVector<Operation *>> scheduleGroups;
  auto totalLatency = problem.getStartTime(anchor).value();

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  // valueMap.clear();
  // for (size_t i = 0; i < iterArgs.size(); ++i)
  //   valueMap.map(whileOp.getAfter().getArgument(i),
  //                stgWhile.getScheduleBlock().getArgument(i));

  // Create the stages.
  Block &funcBlock = funcOp.getBody().front();
  auto *funcReturn = funcOp.getBody().back().getTerminator();
  builder.setInsertionPoint(funcReturn);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    OpBuilder::InsertionGuard g(builder);

    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isFuncTerminator = [funcOp](Operation *op) {
      return isa<func::ReturnOp>(op) && op->getParentOp() == funcOp;
    };
    SmallVector<Type> stepTypes;
    DenseSet<Operation *> opsWithReturns;
    for (auto *op : group) {
      for (auto *user : op->getUsers()) {
        if (opOrParentStartTime(problem, user) > startTime ||
            isFuncTerminator(user)) {
          if (!opsWithReturns.contains(op)) {
            opsWithReturns.insert(op);
            stepTypes.append(op->getResultTypes().begin(),
                             op->getResultTypes().end());
          }
        }
      }
    }

    // Create the stage itself.
    auto stage = builder.create<LoopScheduleStepOp>(stepTypes);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    // Sort the group according to original dominance.
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });

    // Move over the operations and add their results to the terminator.
    SmallVector<std::tuple<Operation *, Operation *, unsigned>> movedOps;
    for (auto *op : group) {
      unsigned resultIndex = stageTerminator->getNumOperands();
      auto *newOp = builder.clone(*op, valueMap);
      dependenceAnalysis->replaceOp(op, newOp);
      if (opsWithReturns.contains(op)) {
        stageTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }
      // All further uses in this step should used the cloned-version of values
      // So we update the mapping in this stage
      for (auto result : op->getResults())
        valueMap.map(result, newOp->getResult(result.getResultNumber()));
    }

    // Add the stage results to the value map for the original op.
    for (auto tuple : movedOps) {
      Operation *op = std::get<0>(tuple);
      Operation *newOp = std::get<1>(tuple);
      unsigned resultIndex = std::get<2>(tuple);
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = stage->getResult(resultIndex + i);
        auto oldValue = op->getResult(i);
        valueMap.map(oldValue, newValue);
      }
    }
  }

  // Update return with correct values
  auto *returnOp = funcOp.getBody().back().getTerminator();
  int numOperands = returnOp->getNumOperands();
  for (int i = 0; i < numOperands; ++i) {
    auto operand = returnOp->getOperand(i);
    auto newValue = valueMap.lookupOrDefault(operand);
    returnOp->setOperand(i, newValue);
  }

  std::function<bool(Operation *)> inTopLevelStepOp = [&](Operation *op) {
    auto parent = op->getParentOfType<LoopScheduleStepOp>();
    if (!parent)
      return false;

    if (isa<func::FuncOp>(parent->getParentOp()))
      return true;

    return inTopLevelStepOp(parent);
  };

  // Remove the loop nest from the IR.
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](Operation *op) {
    if ((isa<LoopScheduleStepOp>(op) && isa<FuncOp>(op->getParentOp())) ||
        inTopLevelStepOp(op) ||
        isa<func::ReturnOp, memref::AllocaOp, arith::ConstantOp,
            memref::AllocOp, AllocInterface>(op))
      return;
    if (auto schedOp = dyn_cast<SchedulableInterface>(op)) {
      if (schedOp.isInitOp())
        return;
    }
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

std::unique_ptr<mlir::Pass>
circt::createFuncToLoopSchedule(bool disableBitwidthMinimization) {
  return std::make_unique<FuncToLoopSchedule>(disableBitwidthMinimization);
}
