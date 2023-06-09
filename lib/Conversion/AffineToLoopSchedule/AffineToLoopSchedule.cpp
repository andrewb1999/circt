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
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <limits>
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

struct AffineToLoopSchedule : public AffineToLoopScheduleBase<AffineToLoopSchedule> {
  void runOnOperation() override;

private:
  ModuloProblem getModuloProblem(CyclicProblem &prob);
  LogicalResult
  lowerAffineStructures(MemoryDependenceAnalysis &dependenceAnalysis);
  LogicalResult unrollSubLoops(AffineForOp &forOp);
  LogicalResult populateOperatorTypes(AffineForOp &loop,
                                      ModuloProblem &problem);
  LogicalResult solveSchedulingProblem(AffineForOp &loop,
                                       ModuloProblem &problem);
  LogicalResult createLoopSchedulePipeline(AffineForOp &loop,
                                       ModuloProblem &problem);
  LogicalResult createLoopScheduleSequential(WhileOp &loop);

  CyclicSchedulingAnalysis *schedulingAnalysis;
  unsigned resII = 1;
  Optional<Problem::OperatorType> limitingOpr;
};

} // namespace

ModuloProblem AffineToLoopSchedule::getModuloProblem(CyclicProblem &prob) {
  auto modProb = ModuloProblem::get(prob.getContainingOp());
  for (auto *op : prob.getOperations()) {
    auto opr = prob.getLinkedOperatorType(op);
    if (opr.has_value()) {
      modProb.setLinkedOperatorType(op, opr.value());
      auto latency = prob.getLatency(opr.value());
      if (latency.has_value())
        modProb.setLatency(opr.value(), latency.value());
    }
    modProb.insertOperation(op);
  }

  for (auto *op : prob.getOperations()) {
    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary())
        assert(modProb.insertDependence(dep).succeeded());
      auto distance = prob.getDistance(dep);
      if (distance.has_value())
        modProb.setDistance(dep, distance.value());
    }
  }

  return modProb;
}

LogicalResult AffineToLoopSchedule::unrollSubLoops(AffineForOp &forOp) {
  auto result = forOp.getBody()->walk<WalkOrder::PostOrder>([](AffineForOp op) {
    if (loopUnrollFull(op).failed())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    forOp.emitOpError("Could not unroll sub loops");
    return failure();
  }

  return success();
}

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
  auto dependenceAnalysis = getAnalysis<MemoryDependenceAnalysis>();

  // After dependence analysis, materialize affine structures.
  if (failed(lowerAffineStructures(dependenceAnalysis)))
    return signalPassFailure();

  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<CyclicSchedulingAnalysis>();

  // Schedule all pipelined loops first
  for (auto loop : llvm::make_early_inc_range(loops)) {
    // Populate the target operator types.
    ModuloProblem moduloProblem =
        getModuloProblem(schedulingAnalysis->getProblem(loop));

    // Insert memory dependences into the problem.
    loop.getBody()->walk([&](Operation *op) {
      ArrayRef<MemoryDependence> dependences =
          dependenceAnalysis.getDependences(op);
      if (dependences.empty())
        return;

      for (MemoryDependence memoryDep : dependences) {
        // Don't insert a dependence into the problem if there is no dependence.
        if (!hasDependence(memoryDep.dependenceType))
          continue;

        memoryDep.source->dump();

        // Insert a dependence into the problem.
        Problem::Dependence dep(memoryDep.source, op);
        auto depInserted = moduloProblem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;

        // Use the lower bound of the innermost loop for this dependence. This
        // assumes outer loops execute sequentially, i.e. one iteration of the
        // inner loop completes before the next iteration is initiated. With
        // proper analysis and lowerings, this can be relaxed.
        unsigned distance = memoryDep.dependenceComponents.back().lb.value();
        if (distance > 0)
          moduloProblem.setDistance(dep, distance);
      }
    });

    if (failed(populateOperatorTypes(loop, moduloProblem)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSchedulingProblem(loop, moduloProblem)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createLoopSchedulePipeline(loop, moduloProblem)))
      return signalPassFailure();
  }

  // Schedule all remaining loops
}

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
/// Also replaces the affine load with the memref load in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineLoadLowering : public OpConversionPattern<AffineLoadOp> {
public:
  AffineLoadLowering(MLIRContext *context,
                     MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands.has_value())
      return failure();

    // Build memref.load memref[expandedMap.results].
    auto memrefLoad = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, op.getMemRef(), *resultOperands);

    dependenceAnalysis.replaceOp(op, memrefLoad);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
/// Also replaces the affine store with the memref store in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineStoreLowering : public OpConversionPattern<AffineStoreOp> {
public:
  AffineStoreLowering(MLIRContext *context,
                      MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap.has_value())
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    auto memrefStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);

    dependenceAnalysis.replaceOp(op, memrefStore);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Helper to hoist computation out of scf::IfOp branches, turning it into a
/// mux-like operation, and exposing potentially concurrent execution of its
/// branches.
struct IfOpHoisting : OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
      }
    });

    return success();
  }
};

/// Helper to determine if an scf::IfOp is in mux-like form.
static bool ifOpLegalityCallback(IfOp op) {
  return op.thenBlock()->without_terminator().empty() &&
         (!op.elseBlock() || op.elseBlock()->without_terminator().empty());
}

/// Helper to mark AffineYieldOp legal, unless it is inside a partially
/// converted scf::IfOp.
static bool yieldOpLegalityCallback(AffineYieldOp op) {
  return !op->getParentOfType<IfOp>();
}

/// After analyzing memory dependences, and before creating the schedule, we
/// want to materialize affine operations with arithmetic, scf, and memref
/// operations, which make the condition computation of addresses, etc.
/// explicit. This is important so the schedule can consider potentially complex
/// computations in the condition of ifs, or the addresses of loads and stores.
/// The dependence analysis will be updated so the dependences from the affine
/// loads and stores are now on the memref loads and stores.
LogicalResult AffineToLoopSchedule::lowerAffineStructures(
    MemoryDependenceAnalysis &dependenceAnalysis) {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, ArithDialect, MemRefDialect,
                         SCFDialect>();
  target.addIllegalOp<AffineIfOp, AffineLoadOp, AffineStoreOp, AffineApplyOp>();
  target.addDynamicallyLegalOp<IfOp>(ifOpLegalityCallback);
  target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);
  patterns.add<AffineLoadLowering>(context, dependenceAnalysis);
  patterns.add<AffineStoreLowering>(context, dependenceAnalysis);
  patterns.add<IfOpHoisting>(context);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  // Loop invariant code motion to hoist produced constants out of loop
  op->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

  return success();
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult
AffineToLoopSchedule::populateOperatorTypes(AffineForOp &loop,
                                        ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loop;

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  Operation *unsupported;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AddIOp, IfOp, AffineYieldOp, arith::ConstantOp, CmpIOp,
              IndexCastOp, memref::AllocaOp, YieldOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<AddIOp, CmpIOp>([&](Operation *seqOp) {
          // These ops need to be sequential for now because we do not
          // have enough information to chain them together yet.
          problem.setLinkedOperatorType(seqOp, seqOpr);
          return WalkResult::advance();
        })
        .Case<AffineStoreOp, memref::StoreOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<memref::StoreOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLimit(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<AffineLoadOp, memref::LoadOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<memref::LoadOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLimit(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return forOp.emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult
AffineToLoopSchedule::solveSchedulingProblem(AffineForOp &loop,
                                         ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loop;

  // Optionally debug problem inputs.
  LLVM_DEBUG(forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::dbgs() << "Scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    llvm::dbgs() << "\n  limit = " << problem.getLimit(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

/// Create the pipeline op for a loop nest.
LogicalResult
AffineToLoopSchedule::createLoopSchedulePipeline(AffineForOp &loop,
                                         ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loop;

  auto innerLoop = loop;
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  // Create Values for the loop's lower and upper bounds.
  Value lowerBound = lowerAffineLowerBound(innerLoop, builder);
  Value upperBound = lowerAffineUpperBound(innerLoop, builder);
  int64_t stepValue = innerLoop.getStep();
  auto step = builder.create<arith::ConstantOp>(
      IntegerAttr::get(builder.getIndexType(), stepValue));

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = innerLoop.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(innerLoop.getIterOperands().begin(),
                  innerLoop.getIterOperands().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount = getConstantTripCount(forOp))
    tripCountAttr = builder.getI64IntegerAttr(*tripCount);

  auto pipeline =
      builder.create<LoopSchedulePipelineOp>(resultTypes, ii, tripCountAttr, iterArgs);

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
  assert(iterArgs.size() == forOp.getBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(forOp.getBody()->getArgument(i),
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
  DenseMap<Operation *, std::pair<unsigned, unsigned>> pipeTimes;

  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];

    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isLoopTerminator = [forOp](Operation *op) {
      return isa<AffineYieldOp>(op) && op->getParentOp() == forOp;
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
      pipeTimes[op] = std::pair(startTime, pipeEndTime);

      // Add register stages for each time slice we need to pipe to
      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Keep a collection of this stages results as keys to our valueMaps
      for (auto result : op->getResults())
        registerValues[startTime].push_back(result);

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

  // Now make register Types and stageValueMaps
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    SmallVector<mlir::Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());

    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  // One more map is needed for the pipeline stages terminator
  stageValueMaps.push_back(valueMap);

  // Create stages along with maps
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });
    auto stageTypes = registerTypes[startTime];
    // Add the induction variable increment in the first stage.
    if (startTime == 0)
      stageTypes.push_back(lowerBound.getType());

    // Create the stage itself.
    builder.setInsertionPoint(stagesBlock.getTerminator());
    auto startTimeAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), startTime);
    auto stage =
        builder.create<LoopSchedulePipelineStageOp>(stageTypes, startTimeAttr);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    for (auto *op : group) {
      auto *newOp = builder.clone(*op, stageValueMaps[startTime]);

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
      unsigned latency = *problem.getLatency(
          *problem.getLinkedOperatorType(res.getDefiningOp()));
      // Multi-cycle case
      if (*problem.getStartTime(res.getDefiningOp()) == startTime &&
          latency > 1)
        destTime = startTime + latency;
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

  for (auto value : forOp.getBody()->getTerminator()->getOperands()) {
    unsigned lookupTime = std::min((unsigned)(stageValueMaps.size() - 1),
                                   pipeTimes[value.getDefiningOp()].second);

    termIterArgs.push_back(stageValueMaps[lookupTime].lookup(value));
    termResults.push_back(stageValueMaps[lookupTime].lookup(value));
  }

  stagesTerminator.getIterArgsMutable().append(termIterArgs);
  stagesTerminator.getResultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  // Remove the loop nest from the IR.
  loop.walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

DenseMap<int64_t, SmallVector<Operation *>>
getOperationCycleMap(Problem &problem) {
  DenseMap<int64_t, SmallVector<Operation *>> map;

  for (auto *op : problem.getOperations()) {
    auto cycleOpt = problem.getStartTime(op);
    assert(cycleOpt.has_value());
    auto cycle = cycleOpt.value();
    auto vec = map.lookup(cycle);
    vec.push_back(op);
    map.insert(std::pair(cycle, vec));
  }

  return map;
}

int64_t longestOperationStartingAtTime(
    Problem &problem, const DenseMap<int64_t, SmallVector<Operation *>> &opMap,
    int64_t cycle) {
  int64_t longestOp = 0;
  for (auto *op : opMap.lookup(cycle)) {
    auto oprType = problem.getLinkedOperatorType(op);
    assert(oprType.has_value());
    auto latency = problem.getLatency(oprType.value());
    assert(latency.has_value());
    if (latency.value() > longestOp)
      longestOp = latency.value();
  }

  return longestOp;
}

/// Returns true if the value is used outside of the given loop.
bool isUsedOutsideOfRegion(Value val, Block *block) {
  return llvm::any_of(val.getUsers(), [&](Operation *user) {
    Operation *u = user;
    while (!isa<ModuleOp>(u->getParentRegion()->getParentOp())) {
      if (u->getBlock() == block) {
        return false;
      }
      u = u->getParentRegion()->getParentOp();
    }
    return true;
  });
}

/// Create the stg ops for a loop nest.
LogicalResult AffineToLoopSchedule::createLoopScheduleSequential(WhileOp &whileOp) {
  auto anchor = whileOp.getYieldOp();

  // Retrieve the cyclic scheduling problem for this loop.
  SharedOperatorsProblem &problem = schedulingAnalysis->getProblem(whileOp);

  auto opMap = getOperationCycleMap(problem);

  ImplicitLocOpBuilder builder(whileOp.getLoc(), whileOp);

  // Get iter args
  auto iterArgs = whileOp.getInits();

  SmallVector<Type> resultTypes;

  for (size_t i = 0; i < whileOp.getNumResults(); ++i) {
    auto result = whileOp.getResult(i);
    auto numUses =
        std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      resultTypes.push_back(result.getType());
    }
  }

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  // Optional<IntegerAttr> tripCountAttr;
  // if (whileOp->hasAttr("stg.tripCount")) {
  //   tripCountAttr = whileOp->getAttr("stg.tripCount").cast<IntegerAttr>();
  // }

  // auto condValue = builder.getIntegerAttr(builder.getIndexType(), 1);
  // auto cond = builder.create<arith::ConstantOp>(whileOp.getLoc(), condValue);

  auto stgWhile = builder.create<stg::STGWhileOp>(whileOp.getLoc(), resultTypes,
                                                  llvm::None, iterArgs);

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(whileOp.getBefore().getArgument(i),
                 stgWhile.getCondBlock().getArgument(i));

  builder.setInsertionPointToStart(&stgWhile.getCondBlock());

  // auto condConst = builder.create<arith::ConstantOp>(whileOp.getLoc(),
  // builder.getIntegerAttr(builder.getI1Type(), 1));
  auto *conditionReg = stgWhile.getCondBlock().getTerminator();
  // conditionReg->insertOperands(0, condConst.getResult());
  for (auto &op : whileOp.getBefore().front().getOperations()) {
    if (isa<scf::ConditionOp>(op)) {
      auto condOp = cast<scf::ConditionOp>(op);
      auto cond = condOp.getCondition();
      auto condNew = valueMap.lookupOrNull(cond);
      assert(condNew);
      conditionReg->insertOperands(0, condNew);
    } else {
      auto *newOp = builder.clone(op, valueMap);
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = newOp->getResult(i);
        auto oldValue = op.getResult(i);
        valueMap.map(oldValue, newValue);
      }
    }
  }

  builder.setInsertionPointToStart(&stgWhile.getScheduleBlock());

  // auto termConst = builder.create<arith::ConstantOp>(whileOp.getLoc(),
  // builder.getIndexAttr(1));
  auto term = stgWhile.getTerminator();
  // term.getIterArgsMutable().append(termConst.getResult());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  SmallVector<SmallVector<Operation *>> scheduleGroups;
  auto totalLatency = problem.getStartTime(anchor).value();

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  valueMap.clear();
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(whileOp.getAfter().getArgument(i),
                 stgWhile.getScheduleBlock().getArgument(i));

  // Create the stages.
  Block &scheduleBlock = stgWhile.getScheduleBlock();
  builder.setInsertionPointToStart(&scheduleBlock);

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
    auto isLoopTerminator = [whileOp](Operation *op) {
      return isa<YieldOp>(op) && op->getParentOp() == whileOp;
    };
    SmallVector<Type> stepTypes;
    DenseSet<Operation *> opsWithReturns;
    for (auto *op : group) {
      for (auto *user : op->getUsers()) {
        if (*problem.getStartTime(user) > startTime || isLoopTerminator(user)) {
          if (!opsWithReturns.contains(op)) {
            opsWithReturns.insert(op);
            stepTypes.append(op->getResultTypes().begin(),
                             op->getResultTypes().end());
          }
        }
      }
    }

    // Create the stage itself.
    auto stage = builder.create<STGStepOp>(stepTypes);
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
      if (opsWithReturns.contains(op)) {
        stageTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }
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

  // Add the iter args and results to the terminator.
  auto scheduleTerminator =
      cast<STGTerminatorOp>(scheduleBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  // termIterArgs.push_back(
  //     scheduleBlock.front().getResult(scheduleBlock.front().getNumResults() -
  //     1));
  for (int i = 0, vals = whileOp.getYieldOp()->getNumOperands(); i < vals;
       ++i) {
    auto value = whileOp.getYieldOp().getOperand(i);
    auto result = whileOp.getResult(i);
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
  for (size_t i = 0; i < whileOp.getNumResults(); ++i) {
    auto result = whileOp.getResult(i);
    auto numUses =
        std::distance(result.getUses().begin(), result.getUses().end());
    if (numUses > 0) {
      whileOp.getResult(i).replaceAllUsesWith(stgWhile.getResult(resultNum++));
    }
  }

  // Remove the loop nest from the IR.
  whileOp.walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

std::unique_ptr<mlir::Pass> circt::createAffineToLoopSchedule() {
  return std::make_unique<AffineToLoopSchedule>();
}
