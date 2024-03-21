//===- AffineToLoopSchedule.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToLoopSchedule.h"
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

struct AffineToLoopSchedule
    : public AffineToLoopScheduleBase<AffineToLoopSchedule> {
  AffineToLoopSchedule(bool disableBitwidthMinimization) {
    this->disableBitwidthMinimization = disableBitwidthMinimization;
  }
  void runOnOperation() override;

private:
  LogicalResult createLoopSchedulePipeline(AffineForOp &loop,
                                           ModuloProblem &problem);
  LogicalResult createLoopScheduleSequential(AffineForOp &loop,
                                             SharedOperatorsProblem &problem);

  std::optional<MemoryDependenceAnalysis> dependenceAnalysis;
};

} // namespace

void AffineToLoopSchedule::runOnOperation() {

  // Collect loops to pipeline and work on them.
  SmallVector<AffineForOp> loops;

  auto hasPipelinedParent = [](Operation *op) {
    Operation *currentOp = op;

    while (!isa<ModuleOp>(currentOp->getParentOp())) {
      if (currentOp->getParentOp()->hasAttr("hls.pipeline"))
        return true;
      currentOp = currentOp->getParentOp();
    }

    return false;
  };

  getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<AffineForOp>(op) || !op->hasAttr("hls.pipeline"))
      return;

    if (hasPipelinedParent(op))
      return;

    loops.push_back(cast<AffineForOp>(op));
  });

  // Unroll loops within this loop to make pipelining possible
  for (auto loop : llvm::make_early_inc_range(loops)) {
    if (failed(unrollSubLoops(loop)))
      return signalPassFailure();
  }

  // Get dependence analysis for the whole function.
  dependenceAnalysis = getAnalysis<MemoryDependenceAnalysis>();

  // for (auto op : getOperation().getOps<LoopInterface>()){
  //   ArrayRef<MemoryDependence> dependences =
  //       dependenceAnalysis->getDependences(op);
  //   if (dependences.empty())
  //     continue;
  //   op->dump();
  //   llvm::errs() << "===============================\n";
  //   for (auto &memoryDep : dependences) {
  //     if (!hasDependence(memoryDep.dependenceType))
  //       continue;
  //     llvm::errs() << "deps: ";
  //     memoryDep.source->dump();
  //   }
  // }

  // getOperation()->getParentOfType<ModuleOp>().dump();

  // getOperation().walk([&](Operation *op) {
  //   ArrayRef<MemoryDependence> dependences =
  //       dependenceAnalysis->getDependences(op);
  //   if (dependences.empty())
  //     return;
  //   op->dump();
  //   for (auto &memoryDep : dependences) {
  //     if (!hasDependence(memoryDep.dependenceType))
  //       continue;
  //     llvm::errs() << "deps: ";
  //     memoryDep.source->dump();
  //   }
  //   llvm::errs() << "===============================\n\n";
  // });

  // After dependence analysis, materialize affine structures.
  if (failed(lowerAffineStructures(getContext(), getOperation(),
                                   *dependenceAnalysis)))
    return signalPassFailure();

  if (failed(postLoweringOptimizations(getContext(), getOperation())))
    return signalPassFailure();

  for (auto affineFor : getOperation().getOps<AffineForOp>()) {
    if (failed(replaceMemoryAccesses(getContext(), affineFor,
                                     *dependenceAnalysis)))
      return signalPassFailure();
  }

  if (!disableBitwidthMinimization) {
    for (auto affineFor : getOperation().getOps<AffineForOp>()) {
      if (failed(bitwidthMinimization(getContext(), affineFor,
                                      *dependenceAnalysis)))
        return signalPassFailure();
    }
  }

  // Schedule all pipelined loops first
  for (auto loop : llvm::make_early_inc_range(loops)) {

    // Populate the target operator types.
    ModuloProblem moduloProblem = getModuloProblem(loop, *dependenceAnalysis);

    if (failed(populateOperatorTypes(loop.getOperation(), loop.getRegion(),
                                     moduloProblem)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveModuloProblem(loop, moduloProblem)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createLoopSchedulePipeline(loop, moduloProblem)))
      return signalPassFailure();
  }

  // Schedule all remaining loops
  SmallVector<AffineForOp> seqLoops;

  getOperation().walk([&](AffineForOp loop) {
    seqLoops.push_back(loop);
    return WalkResult::advance();
  });

  // Schedule loops
  for (auto loop : seqLoops) {
    // getOperation().dump();
    // loop.dump();
    assert(loop.getLoopRegions().size() == 1);
    auto problem = getSharedOperatorsProblem(loop, *dependenceAnalysis);

    // Populate the target operator types.
    if (failed(populateOperatorTypes(loop.getOperation(),
                                     *loop.getLoopRegions().front(), problem)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSharedOperatorsProblem(*loop.getLoopRegions().front(),
                                           problem)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createLoopScheduleSequential(loop, problem)))
      return signalPassFailure();
  }
}


/// Create the pipeline op for a loop nest.
LogicalResult
AffineToLoopSchedule::createLoopSchedulePipeline(AffineForOp &loop,
                                                 ModuloProblem &problem) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  // loop.dump();

  builder.setInsertionPointToStart(
      &loop->getParentOfType<FuncOp>().getBody().front());

  // Create Values for the loop's lower and upper bounds.
  Value lowerBound;
  Value upperBound;
  Type boundType = builder.getIndexType();
  if (loop.hasConstantBounds()) {
    auto lower = loop.getConstantLowerBound();
    auto upper = loop.getConstantUpperBound();
    // int64_t largestValue;
    // // bool isSigned;
    // if (lower >= 0 && upper >= 0) {
    //   // isSigned = false;
    //   largestValue = std::max(lower, upper);
    // } else {
    //   assert(false && "not handling negative affine bounds yet");
    // }
    boundType = loop.getInductionVar().getType();
    lowerBound =
        builder.create<arith::ConstantOp>(IntegerAttr::get(boundType, lower));
    upperBound =
        builder.create<arith::ConstantOp>(IntegerAttr::get(boundType, upper));
  } else {
    lowerBound = lowerAffineLowerBound(loop, builder);
    upperBound = lowerAffineUpperBound(loop, builder);
  }
  int64_t stepValue = loop.getStep().getSExtValue();
  auto step =
      builder.create<arith::ConstantOp>(IntegerAttr::get(boundType, stepValue));

  builder.setInsertionPoint(loop);

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = loop.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(loop.getInits().begin(), loop.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount = getConstantTripCount(loop))
    tripCountAttr = builder.getI64IntegerAttr(*tripCount);

  auto pipeline = builder.create<LoopSchedulePipelineOp>(
      resultTypes, ii, tripCountAttr, iterArgs);

  // Create the condition, which currently just compares the induction variable
  // to the upper bound.
  Block &condBlock = pipeline.getCondBlock();
  builder.setInsertionPointToStart(&condBlock);
  auto cmpResult = builder.create<arith::CmpIOp>(
      builder.getI1Type(), arith::CmpIPredicate::ult, condBlock.getArgument(0),
      upperBound);
  condBlock.getTerminator()->insertOperands(0, {cmpResult});

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // Nested loops are not supported yet.
  assert(iterArgs.size() == loop.getBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(loop.getBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));

  // Create the stages.
  Block &stagesBlock = pipeline.getStagesBlock();
  builder.setInsertionPointToStart(&stagesBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());

  // Keys for translating values in each stage
  SmallVector<SmallVector<Value>> registerValues;
  SmallVector<SmallVector<Type>> registerTypes;

  // The maps that ensure a stage uses the correct version of a value
  SmallVector<IRMapping> stageValueMaps;

  // For storing the range of stages an operation's results need to be valid for
  DenseMap<Value, std::pair<unsigned, unsigned>> pipeTimes;

  DenseSet<unsigned> newStartTimes;
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    newStartTimes.insert(startTime);
    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isLoopTerminator = [loop](Operation *op) {
      return isa<AffineYieldOp>(op) && op->getParentOp() == loop;
    };

    // Initialize set of registers up until this point in time
    for (unsigned i = registerValues.size(); i <= startTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());

    // Check each operation to see if its results need plumbing
    for (auto *op : group) {
      if (op->getUsers().empty())
        continue;

      unsigned pipeEndTime = 0;
      for (auto *user : op->getUsers()) {
        unsigned userStartTime = *problem.getStartTime(user);
        if (*problem.getStartTime(user) > startTime)
          pipeEndTime = std::max(pipeEndTime, userStartTime);
        else if (isLoopTerminator(user))
          // Manually forward the value into the terminator's valueMap
          pipeEndTime = std::max(pipeEndTime, userStartTime + 1);
      }

      // Insert the range of pipeline stages the value needs to be valid for
      for (auto res : op->getResults())
        pipeTimes[res] = std::pair(startTime, pipeEndTime);

      // Add register stages for each time slice we need to pipe to
      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Keep a collection of this stages results as keys to our valueMaps
      for (auto result : op->getResults()) {
        for (auto *user : result.getUsers()) {
          auto inThisGroup = false;
          for (auto *op : group) {
            if (user == op) {
              inThisGroup = true;
            }
          }
          if (!inThisGroup) {
            registerValues[startTime].push_back(result);
            break;
          }
        }
      }

      // Other stages that use the value will need these values as keys too
      unsigned firstUse = std::max(
          startTime + 1,
          startTime + *problem.getLatency(*problem.getLinkedOperatorType(op)));
      for (unsigned i = firstUse; i < pipeEndTime; ++i) {
        for (auto result : op->getResults())
          registerValues[i].push_back(result);
      }
    }
  }

  // loop.dump();
  assert(loop.getLoopRegions().size() == 1);
  for (auto it : enumerate(loop.getLoopRegions().front()->getArguments())) {
    auto iterArg = it.value();
    if (iterArg.getUsers().empty())
      continue;

    unsigned startPipeTime = 0;
    if (it.index() > 0) {
      // Handle extra iter args
      auto *term = loop.getLoopRegions().front()->back().getTerminator();
      auto &termOperand = term->getOpOperand(it.index() - 1);
      auto *definingOp = termOperand.get().getDefiningOp();
      assert(definingOp != nullptr);
      startPipeTime = *problem.getStartTime(definingOp);
    }

    unsigned pipeEndTime = 0;
    for (auto *user : iterArg.getUsers()) {
      unsigned userStartTime = *problem.getStartTime(user);
      if (userStartTime >= startPipeTime)
        pipeEndTime = std::max(pipeEndTime, userStartTime);
    }

    // Do not need to pipe result if there are no later uses
    // iterArg.dump();
    // llvm::errs() << startPipeTime << " : " << pipeEndTime << "\n";
    if (startPipeTime >= pipeEndTime)
      continue;

    // Make sure a stage exists for every time between startTime and
    // pipeEndTime
    // for (unsigned i = startTime; i < pipeEndTime; ++i)
    //   newStartTimes.insert(i);

    // Insert the range of pipeline stages the value needs to be valid for
    pipeTimes[iterArg] = std::pair(startPipeTime, pipeEndTime);

    // Add register stages for each time slice we need to pipe to
    for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
      registerValues.push_back(SmallVector<Value>());

    // Keep a collection of this stages results as keys to our valueMaps
    registerValues[startPipeTime].push_back(iterArg);

    // Other stages that use the value will need these values as keys too
    unsigned firstUse = startPipeTime + 1;
    for (unsigned i = firstUse; i < pipeEndTime; ++i) {
      registerValues[i].push_back(iterArg);
    }
  }

  // Now make register Types and stageValueMaps
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    if (!registerValues[i].empty()) {
      newStartTimes.insert(i);
    }
    SmallVector<mlir::Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());

    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  // llvm::errs() << "startTime\n";
  // for (auto startTime : startTimes) {
  //   llvm::errs() << "time = " << startTime << "\n";
  //   for (auto *op : startGroups[startTime]) {
  //     op->dump();
  //   }
  // }
  // llvm::errs() << "after startTime\n";

  // One more map is needed for the pipeline stages terminator
  stageValueMaps.push_back(valueMap);

  startTimes.clear();
  startTimes.append(newStartTimes.begin(), newStartTimes.end());
  llvm::sort(startTimes);
  SmallVector<bool> iterArgNeedsForwarding;
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    iterArgNeedsForwarding.push_back(false);
  }
  // Create stages along with maps
  for (auto i : enumerate(startTimes)) {
    auto startTime = i.value();
    auto lastStage = i.index() == startTimes.size() - 1;
    auto group = startGroups[startTime];
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });
    auto stageTypes = registerTypes[startTime];
    uint64_t largestLatency = 1;
    if (lastStage) {
      // Last stage must end after all ops have finished
      for (auto *op : group) {
        auto oprType = problem.getLinkedOperatorType(op).value();
        uint64_t latency = problem.getLatency(oprType).value();
        if (latency > largestLatency) {
          largestLatency = latency;
        }
      }
    }
    uint64_t endTime = startTime + largestLatency;

    // Add the induction variable increment in the first stage.
    if (startTime == 0) {
      stageTypes.push_back(lowerBound.getType());
    }

    // Create the stage itself.
    builder.setInsertionPoint(stagesBlock.getTerminator());
    auto startTimeAttr =
        builder.getIntegerAttr(builder.getIntegerType(64), startTime);
    auto endTimeAttr =
        builder.getIntegerAttr(builder.getIntegerType(64), endTime);
    auto stage = builder.create<LoopSchedulePipelineStageOp>(
        stageTypes, startTimeAttr, endTimeAttr);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    for (auto *op : group) {
      auto *newOp = builder.clone(*op, stageValueMaps[startTime]);
      // llvm::errs() << memAnalysis.getDependences(op).size() << "\n";
      // llvm::errs() << "before\n";
      // op->dump();
      // newOp->dump();
      dependenceAnalysis->replaceOp(op, newOp);
      // llvm::errs() << "after\n";

      // All further uses in this stage should used the cloned-version of values
      // So we update the mapping in this stage
      for (auto result : op->getResults())
        stageValueMaps[startTime].map(
            result, newOp->getResult(result.getResultNumber()));
    }

    // Register all values in the terminator, using their mapped value
    SmallVector<Value> stageOperands;
    unsigned resIndex = 0;
    for (auto res : registerValues[startTime]) {
      stageOperands.push_back(stageValueMaps[startTime].lookup(res));
      // Additionally, update the map of the stage that will consume the
      // registered value
      unsigned destTime = startTime + 1;
      if (!isa<BlockArgument>(res)) {
        unsigned latency = *problem.getLatency(
            *problem.getLinkedOperatorType(res.getDefiningOp()));
        // Multi-cycle case
        if (*problem.getStartTime(res.getDefiningOp()) == startTime &&
            latency > 1)
          destTime = startTime + latency;
      }
      destTime = std::min((unsigned)(stageValueMaps.size() - 1), destTime);
      stageValueMaps[destTime].map(res, stage.getResult(resIndex++));
    }
    // Add these mapped values to pipeline.register
    stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                    stageOperands);

    // Add the induction variable increment to the first stage.
    if (startTime == 0) {
      auto incResult =
          builder.create<arith::AddIOp>(stagesBlock.getArgument(0), step);
      stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                      incResult->getResults());
    }
  }

  // Add the iter args and results to the terminator.
  auto stagesTerminator =
      cast<LoopScheduleTerminatorOp>(stagesBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  termIterArgs.push_back(
      stagesBlock.front().getResult(stagesBlock.front().getNumResults() - 1));

  for (auto value : loop.getBody()->getTerminator()->getOperands()) {
    unsigned lookupTime = std::min((unsigned)(stageValueMaps.size() - 1),
                                   pipeTimes[value].second);

    termIterArgs.push_back(stageValueMaps[lookupTime].lookup(value));
    termResults.push_back(stageValueMaps[lookupTime].lookup(value));
  }

  stagesTerminator.getIterArgsMutable().append(termIterArgs);
  stagesTerminator.getResultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < loop.getNumResults(); ++i)
    loop.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  dependenceAnalysis->replaceOp(loop, pipeline);

  loop.walk(
      [&](Operation *op) { assert(!dependenceAnalysis->containsOp(op)); });

  // Remove the loop nest from the IR.
  loop.walk([&](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

/// Create the stg ops for a loop nest.
LogicalResult AffineToLoopSchedule::createLoopScheduleSequential(
    AffineForOp &loop, SharedOperatorsProblem &problem) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  builder.setInsertionPointToStart(
      &loop->getParentOfType<FuncOp>().getBody().front());

  // Create Values for the loop's lower and upper bounds.
  Value lowerBound;
  Value upperBound;
  Type boundType = builder.getIndexType();
  if (loop.hasConstantBounds()) {
    auto lower = loop.getConstantLowerBound();
    auto upper = loop.getConstantUpperBound();
    // int64_t largestValue;
    // // bool isSigned;
    // if (lower >= 0 && upper >= 0) {
    //   // isSigned = false;
    //   largestValue = std::max(lower, upper);
    // } else {
    //   assert(false && "not handling negative affine bounds yet");
    // }
    boundType = loop.getInductionVar().getType();
    lowerBound =
        builder.create<arith::ConstantOp>(IntegerAttr::get(boundType, lower));
    upperBound =
        builder.create<arith::ConstantOp>(IntegerAttr::get(boundType, upper));
  } else {
    lowerBound = lowerAffineLowerBound(loop, builder);
    upperBound = lowerAffineUpperBound(loop, builder);
  }
  int64_t stepValue = loop.getStep().getSExtValue();
  auto incr =
      builder.create<arith::ConstantOp>(IntegerAttr::get(boundType, stepValue));

  builder.setInsertionPoint(loop);

  auto *anchor = loop.getBody()->getTerminator();

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = loop.getResultTypes();

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(loop.getInits().begin(), loop.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount = getConstantTripCount(loop))
    tripCountAttr = builder.getI64IntegerAttr(*tripCount);

  auto opMap = getOperationCycleMap(problem);

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  // Optional<IntegerAttr> tripCountAttr;
  // if (loop->hasAttr("stg.tripCount")) {
  //   tripCountAttr = loop->getAttr("stg.tripCount").cast<IntegerAttr>();
  // }

  // auto condValue = builder.getIntegerAttr(builder.getIndexType(), 1);
  // auto cond = builder.create<arith::ConstantOp>(loop.getLoc(), condValue);

  auto sequential = builder.create<LoopScheduleSequentialOp>(
      loop.getLoc(), resultTypes, tripCountAttr, iterArgs);

  // Create the condition, which currently just compares the induction variable
  // to the upper bound.
  Block &condBlock = sequential.getCondBlock();
  builder.setInsertionPointToStart(&condBlock);
  auto cmpResult = builder.create<arith::CmpIOp>(
      builder.getI1Type(), arith::CmpIPredicate::ult, condBlock.getArgument(0),
      upperBound);
  condBlock.getTerminator()->insertOperands(0, {cmpResult});

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // for (size_t i = 0; i < iterArgs.size(); ++i)
  //   valueMap.map(loop.getBefore().getArgument(i),
  //                stgWhile.getCondBlock().getArgument(i));

  // builder.setInsertionPointToStart(&sequential.getCondBlock());

  // // auto condConst = builder.create<arith::ConstantOp>(loop.getLoc(),
  // // builder.getIntegerAttr(builder.getI1Type(), 1));
  // auto *conditionReg = stgWhile.getCondBlock().getTerminator();
  // // conditionReg->insertOperands(0, condConst.getResult());
  // for (auto &op : loop.getBefore().front().getOperations()) {
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

  builder.setInsertionPointToStart(&sequential.getScheduleBlock());

  // auto termConst = builder.create<arith::ConstantOp>(loop.getLoc(),
  // builder.getIndexAttr(1));
  // auto term = stgWhile.getTerminator();
  // term.getIterArgsMutable().append(termConst.getResult());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  unsigned endTime = 0;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
    if (startTime > endTime)
      endTime = *startTime;
  }

  auto hasLaterUse = [&](Operation *op, uint32_t resTime) {
    for (uint32_t i = resTime + 1; i < endTime; ++i) {
      if (startGroups.contains(i)) {
        auto startGroup = startGroups[i];
        for (auto *operation : startGroup) {
          for (auto &operand : operation->getOpOperands()) {
            if (operand.get().getDefiningOp() == op)
              return true;
          }
        }
      }
    }
    return false;
  };

  // Must re-register return values of memories if they are used later
  for (auto *op : problem.getOperations()) {
    if (isa<LoadOp, AffineLoadOp>(op)) {
      auto startTime = problem.getStartTime(op);
      auto resTime = *startTime + 1;
      if (hasLaterUse(op, resTime) && !startGroups.contains(resTime)) {
        startGroups[resTime] = SmallVector<Operation *>();
      }
    }
    if (auto load = dyn_cast<LoadInterface>(op)) {
      auto startTime = problem.getStartTime(op);
      auto latency = load.getLatency();
      auto resTime = *startTime + latency;
      if (hasLaterUse(op, resTime) && !startGroups.contains(resTime)) {
        startGroups[resTime] = SmallVector<Operation *>();
      }
    }
  }

  Block &scheduleBlock = sequential.getScheduleBlock();

  assert(loop.getLoopRegions().size() == 1);
  if (!loop.getLoopRegions().front()->getArgument(0).getUsers().empty()) {
    // llvm::errs() << "Add extra start group\n";
    auto containsLoop = false;
    for (auto *op : startGroups[endTime]) {
      if (isa<LoopInterface>(op)) {
        containsLoop = true;
        break;
      }
    }
    if (containsLoop)
      startGroups[endTime + 1] = SmallVector<Operation *>();
  }

  SmallVector<SmallVector<Operation *>> scheduleGroups;
  auto totalLatency = problem.getStartTime(anchor).value();

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  valueMap.clear();
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(loop.getLoopRegions().front()->getArgument(i),
                 sequential.getScheduleBlock().getArgument(i));

  // Create the stages.
  builder.setInsertionPointToStart(&scheduleBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DenseMap<uint32_t, SmallVector<Value>> reregisterValues;

  LoopScheduleStepOp lastStep;
  DominanceInfo dom(getOperation());
  for (auto i : enumerate(startTimes)) {
    auto startTime = i.value();
    auto group = startGroups[startTime];
    OpBuilder::InsertionGuard g(builder);

    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isLoopTerminator = [loop](Operation *op) {
      return isa<AffineYieldOp>(op) && op->getParentOp() == loop;
    };

    SmallVector<Type> stepTypes;
    DenseSet<Operation *> opsWithReturns;
    for (auto *op : group) {
      for (auto *user : op->getUsers()) {
        auto *userOrAncestor =
            loop.getLoopRegions().front()->findAncestorOpInRegion(*user);
        auto startTimeOpt = problem.getStartTime(userOrAncestor);
        if ((startTimeOpt.has_value() && *startTimeOpt > startTime) ||
            isLoopTerminator(user)) {
          if (!opsWithReturns.contains(op)) {
            opsWithReturns.insert(op);
            stepTypes.append(op->getResultTypes().begin(),
                             op->getResultTypes().end());
          }
        }
      }
    }

    for (auto val : reregisterValues[startTime]) {
      stepTypes.push_back(val.getType());
    }

    if (i.index() == startTimes.size() - 1) {
      // Add index increment to first step
      stepTypes.push_back(builder.getIndexType());
    }

    // Create the step itself.
    auto step = builder.create<LoopScheduleStepOp>(stepTypes);
    auto &stepBlock = step.getBodyBlock();
    auto *stepTerminator = stepBlock.getTerminator();
    builder.setInsertionPointToStart(&stepBlock);

    // Sort the group according to original dominance.
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });

    // Move over the operations and add their results to the terminator.
    SmallVector<std::tuple<Operation *, Operation *, unsigned>> movedOps;
    for (auto *op : group) {
      unsigned resultIndex = stepTerminator->getNumOperands();
      auto *newOp = builder.clone(*op, valueMap);
      dependenceAnalysis->replaceOp(op, newOp);
      std::queue<Operation *> oldOps;
      op->walk([&](Operation *op) { oldOps.push(op); });
      newOp->walk([&](Operation *op) {
        Operation *oldOp = oldOps.front();
        dependenceAnalysis->replaceOp(oldOp, op);
        oldOps.pop();
      });
      if (opsWithReturns.contains(op)) {
        stepTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }
      // All further uses in this stage should used the cloned-version of values
      // So we update the mapping in this stage
      for (auto result : op->getResults())
        valueMap.map(result, newOp->getResult(result.getResultNumber()));
    }

    // Reregister values
    for (auto val : reregisterValues[startTime]) {
      unsigned resultIndex = stepTerminator->getNumOperands();
      stepTerminator->insertOperands(resultIndex, valueMap.lookup(val));
      auto newValue = step->getResult(resultIndex);
      valueMap.map(val, newValue);
    }

    // Add the step results to the value map for the original op.
    for (auto tuple : movedOps) {
      Operation *op = std::get<0>(tuple);
      Operation *newOp = std::get<1>(tuple);
      unsigned resultIndex = std::get<2>(tuple);
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = step->getResult(resultIndex + i);
        auto oldValue = op->getResult(i);
        valueMap.map(oldValue, newValue);
      }
    }

    // Add values that need to be reregistered in the future
    for (auto *op : group) {
      if (auto load = dyn_cast<LoadOp>(op)) {
        if (hasLaterUse(op, startTime + 1)) {
          reregisterValues[startTime + 1].push_back(load.getResult());
        }
      } else if (auto load = dyn_cast<LoadInterface>(op)) {
        auto latency = load.getLatency();
        if (hasLaterUse(op, startTime + latency)) {
          auto resTime = startTime + latency;
          reregisterValues[resTime].push_back(load.getResult());
        }
      }
    }

    if (i.index() == startTimes.size() - 1) {
      auto incResult =
          builder.create<arith::AddIOp>(scheduleBlock.getArgument(0), incr);
      stepTerminator->insertOperands(stepTerminator->getNumOperands(),
                                     incResult->getResults());
      lastStep = step;
    }
  }

  // Add the iter args and results to the terminator.
  auto scheduleTerminator =
      cast<LoopScheduleTerminatorOp>(scheduleBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  termIterArgs.push_back(lastStep.getResult(lastStep.getNumResults() - 1));
  for (int i = 0, vals = anchor->getNumOperands(); i < vals; ++i) {
    auto value = anchor->getOperand(i);
    auto result = loop.getResult(i);
    termIterArgs.push_back(valueMap.lookup(value));
    auto numUses =
        std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      termResults.push_back(valueMap.lookup(value));
    }
  }

  scheduleTerminator.getIterArgsMutable().append(termIterArgs);
  scheduleTerminator.getResultsMutable().append(termResults);

  // Replace loop results with while results.
  auto resultNum = 0;
  for (size_t i = 0; i < loop.getNumResults(); ++i) {
    auto result = loop.getResult(i);
    auto numUses =
        std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      loop.getResult(i).replaceAllUsesWith(sequential.getResult(resultNum++));
    }
  }

  dependenceAnalysis->replaceOp(loop, sequential);

  loop.walk(
      [&](Operation *op) { assert(!dependenceAnalysis->containsOp(op)); });

  // Remove the loop nest from the IR.
  loop.walk([&](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

std::unique_ptr<mlir::Pass>
circt::createAffineToLoopSchedule(bool disableBitwidthMinimization) {
  return std::make_unique<AffineToLoopSchedule>(disableBitwidthMinimization);
}
