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

struct AffineToLoopSchedule
    : public AffineToLoopScheduleBase<AffineToLoopSchedule> {
  AffineToLoopSchedule(bool disableBitwidthMinimization) {
    this->disableBitwidthMinimization = disableBitwidthMinimization;
  }
  void runOnOperation() override;

private:
  LogicalResult populateOperatorTypes(Operation *op, Region &loopBody,
                                      SharedOperatorsProblem &problem);
  LogicalResult solveModuloProblem(AffineForOp &loop, ModuloProblem &problem);
  LogicalResult solveSharedOperatorsProblem(Region &region,
                                            SharedOperatorsProblem &problem);
  LogicalResult createLoopSchedulePipeline(AffineForOp &loop,
                                           ModuloProblem &problem);
  LogicalResult createLoopScheduleSequential(AffineForOp &loop,
                                             SharedOperatorsProblem &problem);
  LogicalResult createFuncLoopSchedule(FuncOp &funcOp,
                                       SharedOperatorsProblem &problem);

  std::optional<MemoryDependenceAnalysis> dependenceAnalysis;
};

} // namespace

void AffineToLoopSchedule::runOnOperation() {

  getOperation()->getParentOfType<ModuleOp>().dump();
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

  if (failed(replaceMemoryAccesses(getContext(), getOperation(), *dependenceAnalysis)))
    return signalPassFailure();

  if (!disableBitwidthMinimization) {
    if (failed(bitwidthMinimization(getContext(), getOperation(), *dependenceAnalysis)))
      return signalPassFailure();
  }

  getOperation()->getParentOfType<ModuleOp>().dump();

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

  // llvm::errs() << "schedule func\n";

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

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult
AffineToLoopSchedule::populateOperatorTypes(Operation *op, Region &loopBody,
                                            SharedOperatorsProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType loopOpr = problem.getOrInsertOperatorType("loop");
  problem.setLatency(loopOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 4);

  Operation *unsupported;
  WalkResult result = loopBody.walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr) {
      return WalkResult::advance();
    }

    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<IfOp, AffineYieldOp, arith::ConstantOp, arith::ExtSIOp,
              arith::ExtUIOp, arith::TruncIOp, CmpIOp, IndexCastOp,
              memref::AllocaOp, memref::AllocOp, loopschedule::AllocInterface,
              YieldOp, func::ReturnOp, arith::SelectOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<AddIOp, SubIOp, CmpIOp, ShLIOp, AndIOp, ShRSIOp, ShRUIOp>(
            [&](Operation *seqOp) {
              // These ops need to be sequential for now because we do not
              // have enough information to chain them together yet.
              problem.setLinkedOperatorType(seqOp, seqOpr);
              return WalkResult::advance();
            })
        .Case<LoopInterface>([&](Operation *loopOp) {
          // llvm::errs() << "loopOp\n";
          // loopOp->dump();
          // llvm::errs() << "loopOp after\n";
          problem.setLinkedOperatorType(loopOp, loopOpr);
          auto loop = cast<LoopInterface>(loopOp);
          loop.getBodyBlock()->walk([&](Operation *op) {
            if (isa<AffineLoadOp, AffineStoreOp, LoopScheduleLoadOp,
                    LoopScheduleStoreOp>(op)) {
              Value memRef = getMemref(op);
              Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
                  "mem_" + std::to_string(hash_value(memRef)));
              problem.setLatency(memOpr, 1);
              // External memories are 1 RW port
              problem.setLimit(memOpr, 1);
              problem.addExtraLimitingType(loopOp, memOpr);
            } else if (isa<LoadInterface>(op)) {
              auto loadOp = cast<loopschedule::LoadInterface>(*op);
              auto latencyOpt = loadOp.getLatency();
              auto limitOpt = loadOp.getLimit();
              Problem::OperatorType portOpr =
                  problem.getOrInsertOperatorType(loadOp.getUniqueId());
              problem.setLatency(portOpr, latencyOpt.value_or(1));
              if (limitOpt.has_value())
                problem.setLimit(portOpr, limitOpt.value());
              problem.addExtraLimitingType(loopOp, portOpr);
            } else if (isa<StoreInterface>(op)) {
              auto storeOp = cast<loopschedule::StoreInterface>(*op);
              auto latencyOpt = storeOp.getLatency();
              auto limitOpt = storeOp.getLimit();
              Problem::OperatorType portOpr =
                  problem.getOrInsertOperatorType(storeOp.getUniqueId());
              problem.setLatency(portOpr, latencyOpt.value_or(1));
              if (limitOpt.has_value())
                problem.setLimit(portOpr, limitOpt.value());
              problem.addExtraLimitingType(loopOp, portOpr);
            }
          });
          return WalkResult::advance();
        })
        .Case<LoopScheduleStoreOp, AffineStoreOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<LoopScheduleStoreOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLimit(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<LoopScheduleLoadOp, AffineLoadOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<LoopScheduleLoadOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLimit(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<loopschedule::LoadInterface>([&](Operation *op) {
          auto loadOp = cast<loopschedule::LoadInterface>(*op);
          auto latencyOpt = loadOp.getLatency();
          auto limitOpt = loadOp.getLimit();
          Problem::OperatorType portOpr =
              problem.getOrInsertOperatorType(loadOp.getUniqueId());
          problem.setLatency(portOpr, latencyOpt.value_or(1));
          if (limitOpt.has_value())
            problem.setLimit(portOpr, limitOpt.value());
          problem.setLinkedOperatorType(op, portOpr);

          return WalkResult::advance();
        })
        .Case<loopschedule::StoreInterface>([&](Operation *op) {
          auto storeOp = cast<loopschedule::StoreInterface>(*op);
          auto latencyOpt = storeOp.getLatency();
          auto limitOpt = storeOp.getLimit();
          Problem::OperatorType portOpr =
              problem.getOrInsertOperatorType(storeOp.getUniqueId());
          problem.setLatency(portOpr, latencyOpt.value_or(1));
          if (limitOpt.has_value())
            problem.setLimit(portOpr, limitOpt.value());
          problem.setLinkedOperatorType(op, portOpr);

          return WalkResult::advance();
        })
        .Case<loopschedule::SchedulableInterface>([&](Operation *op) {
          auto schedOp = cast<SchedulableInterface>(op);
          auto latency = schedOp.getOpLatency();
          auto limitOpt = schedOp.getOpLimit();
          Problem::OperatorType opr =
              problem.getOrInsertOperatorType(schedOp.getUniqueId());
          problem.setLatency(opr, latency);
          if (limitOpt.has_value())
            problem.setLimit(opr, limitOpt.value());
          problem.setLinkedOperatorType(op, opr);

          return WalkResult::advance();
        })
        .Case<MulIOp, RemUIOp, RemSIOp, DivSIOp>([&](Operation *mcOp) {
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
    return op->emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult AffineToLoopSchedule::solveModuloProblem(AffineForOp &loop,
                                                       ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loop;

  LLVM_DEBUG(forOp.dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(for (auto *op
                  : problem.getOperations()) {
    if (auto parent = op->getParentOfType<LoopInterface>(); parent)
      continue;
    llvm::dbgs() << "Modulo scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    llvm::dbgs() << "\n  limit = " << problem.getLimit(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  });

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
      if (auto parent = op->getParentOfType<LoopInterface>(); parent)
        return;
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult AffineToLoopSchedule::solveSharedOperatorsProblem(
    Region &region, SharedOperatorsProblem &problem) {

  LLVM_DEBUG(region.getParentOp()->dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto parent = op->getParentOfType<LoopInterface>(); parent)
      return;
    llvm::dbgs() << "Shared Operator scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    llvm::dbgs() << "\n  limit = " << problem.getLimit(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { "
                     << "source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = region.back().getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto parent = op->getParentOfType<LoopInterface>(); parent)
        return;
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

// Replaces memref loads/stores with loopschedule loads/stores
Operation *cloneOrReplace(OpBuilder builder, Operation *op, IRMapping valueMap) {
  // if (auto load = dyn_cast<memref::LoadOp>(op)) {
  //   SmallVector<Value> indices;
  //   auto memref = valueMap.lookupOrDefault(load.getMemRef());
  //   auto returnType = cast<MemRefType>(memref.getType()).getElementType();
  //   for (auto opValue : load.getIndices())
  //    indices.push_back(valueMap.lookupOrDefault(opValue));
  //   return builder.create<LoopScheduleLoadOp>(op->getLoc(), returnType, 
  //             memref, indices);
  // }

  // if (auto store = dyn_cast<memref::StoreOp>(op)) {
  //   SmallVector<Value> indices;
  //   auto memref = valueMap.lookupOrDefault(store.getMemRef());
  //   auto valToStore = valueMap.lookupOrDefault(store.getValueToStore());
  //   for (auto opValue : store.getIndices())
  //    indices.push_back(valueMap.lookupOrDefault(opValue));
  //   return builder.create<LoopScheduleStoreOp>(op->getLoc(), valToStore, 
  //             memref, indices);
  // }

  return builder.clone(*op, valueMap);
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
    lowerBound = builder.create<arith::ConstantOp>(
        IntegerAttr::get(boundType, lower));
    upperBound = builder.create<arith::ConstantOp>(
        IntegerAttr::get(boundType, upper));
  } else {
    lowerBound = lowerAffineLowerBound(loop, builder);
    upperBound = lowerAffineUpperBound(loop, builder);
  }
  int64_t stepValue = loop.getStep().getSExtValue();
  auto step = builder.create<arith::ConstantOp>(
      IntegerAttr::get(boundType, stepValue));

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
      auto *newOp = cloneOrReplace(builder, op, stageValueMaps[startTime]);
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
LogicalResult AffineToLoopSchedule::createLoopScheduleSequential(
    AffineForOp &loop, SharedOperatorsProblem &problem) {
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
    lowerBound = builder.create<arith::ConstantOp>(
        IntegerAttr::get(boundType, lower));
    upperBound = builder.create<arith::ConstantOp>(
        IntegerAttr::get(boundType, upper));
  } else {
    lowerBound = lowerAffineLowerBound(loop, builder);
    upperBound = lowerAffineUpperBound(loop, builder);
  }
  int64_t stepValue = loop.getStep().getSExtValue();
  auto incr = builder.create<arith::ConstantOp>(
      IntegerAttr::get(boundType, stepValue));

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
      auto latency = load.getLatency().has_value() ? *load.getLatency() : 1;
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
        valueMap.map(
            result, newOp->getResult(result.getResultNumber()));
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
        auto latency = load.getLatency().has_value() ? *load.getLatency() : 1;
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

int64_t opOrParentStartTime(Problem &problem, Operation *op) {
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
AffineToLoopSchedule::createFuncLoopSchedule(FuncOp &funcOp,
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
      auto *newOp = cloneOrReplace(builder, op, valueMap);
      dependenceAnalysis->replaceOp(op, newOp);
      if (opsWithReturns.contains(op)) {
        stageTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }
      // All further uses in this step should used the cloned-version of values
      // So we update the mapping in this stage
      for (auto result : op->getResults())
        valueMap.map(
            result, newOp->getResult(result.getResultNumber()));
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

std::unique_ptr<mlir::Pass> circt::createAffineToLoopSchedule(bool disableBitwidthMinimization) {
  return std::make_unique<AffineToLoopSchedule>(disableBitwidthMinimization);
}
