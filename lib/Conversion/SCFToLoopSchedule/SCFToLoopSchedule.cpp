//===- SCFToLoopSchedule.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToLoopSchedule.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Dialect/SSP/SSPInterfaces.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
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
#include <algorithm>
#include <cassert>
#include <limits>
#include <math.h>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <utility>

#define DEBUG_TYPE "scf-to-loopschedule"

namespace circt {
#define GEN_PASS_DEF_SCFTOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

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

struct SCFToLoopSchedulePass
    : public circt::impl::SCFToLoopScheduleBase<SCFToLoopSchedulePass> {
  using SCFToLoopScheduleBase::SCFToLoopScheduleBase;
  void runOnOperation() override;

private:
  LogicalResult populateOperatorTypes(Operation *op, Region &body,
                                      ChainingSharedOperatorsProblem &problem);
  LogicalResult solveChainingModuloProblem(scf::WhileOp &loop,
                                           ChainingModuloProblem &problem,
                                           float cycleTime);
  LogicalResult solveChainingSharedOperatorsProblem(
      Region &region, ChainingSharedOperatorsProblem &problem, float cycleTime);
  LogicalResult createLoopSchedulePipeline(scf::WhileOp &loop,
                                           CyclicProblem &problem);
  LogicalResult createLoopScheduleSequential(scf::WhileOp &loop,
                                             Problem &problem);
  LogicalResult createFuncLoopSchedule(FuncOp &funcOp, Problem &problem);

  std::optional<LoopScheduleDependenceAnalysis> dependenceAnalysis;
  std::optional<OperatorLibraryAnalysis> operatorLibraryAnalysis;
  PredicateUse predicateUse;
  PredicateMap predicateMap;
};

} // namespace

void SCFToLoopSchedulePass::runOnOperation() {
  float cycleTime = prioritizeII ? 2.0 : 1.0;

  // Collect loops to pipeline and work on them.
  SmallVector<scf::WhileOp> loops;

  auto hasPipelinedParent = [](Operation *op) {
    Operation *currentOp = op;

    while (!isa<ModuleOp>(currentOp->getParentOp())) {
      if (currentOp->getParentOp()->hasAttr("hls.pipeline"))
        return true;
      currentOp = currentOp->getParentOp();
    }

    return false;
  };

  auto res = getOperation()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<scf::WhileOp>(op) || !op->hasAttr("hls.pipeline"))
      return WalkResult::advance();

    if (hasPipelinedParent(op))
      return WalkResult::interrupt();

    loops.push_back(cast<scf::WhileOp>(op));
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    getOperation().emitOpError(
        "Loops marked for pipelining cannot contain other loops");
    signalPassFailure();
  }

  // Get dependence analysis for the whole function.
  dependenceAnalysis = getAnalysis<LoopScheduleDependenceAnalysis>();

  operatorLibraryAnalysis = getAnalysis<OperatorLibraryAnalysis>();

  // Schedule all pipelined loops first
  for (auto loop : llvm::make_early_inc_range(loops)) {
    ResourceMap resourceMap;
    ResourceLimits resourceLimits;
    if (failed(recordMemoryResources(loop.getOperation(), loop.getAfter(),
                                     resourceMap, resourceLimits)))
      return signalPassFailure();

    if (failed(
            ifOpConversion(loop.getOperation(), loop.getAfter(), predicateMap)))
      return signalPassFailure();

    // Populate the target operator types.
    ChainingModuloProblem moduloProblem =
        getChainingModuloProblem(loop, *dependenceAnalysis);

    if (failed(populateOperatorTypes(loop.getOperation(), loop.getAfter(),
                                     moduloProblem)))
      return signalPassFailure();

    if (failed(addMemoryResources(loop.getOperation(), loop.getAfter(),
                                  moduloProblem, resourceMap, resourceLimits)))
      return signalPassFailure();

    addPredicateDependencies(loop.getOperation(), loop.getAfter(),
                             moduloProblem, predicateMap, predicateUse);

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveChainingModuloProblem(loop, moduloProblem, cycleTime))) {
      llvm::errs() << "Failed to solve ChainingModuloProblem\n";
      return signalPassFailure();
    }

    // Convert the IR.
    if (failed(createLoopSchedulePipeline(loop, moduloProblem)))
      return signalPassFailure();
  }

  // Schedule all remaining loops
  SmallVector<scf::WhileOp> seqLoops;

  getOperation().walk([&](scf::WhileOp loop) {
    seqLoops.push_back(loop);
    return WalkResult::advance();
  });

  // Schedule loops
  for (auto loop : seqLoops) {
    ResourceMap resourceMap;
    ResourceLimits resourceLimits;
    if (failed(recordMemoryResources(loop.getOperation(), loop.getAfter(),
                                     resourceMap, resourceLimits)))
      return signalPassFailure();

    if (failed(
            ifOpConversion(loop.getOperation(), loop.getAfter(), predicateMap)))
      return signalPassFailure();

    auto problem = getChainingSharedOperatorsProblem(loop, *dependenceAnalysis);

    // Populate the target operator types.
    if (failed(populateOperatorTypes(loop.getOperation(), loop.getAfter(),
                                     problem)))
      return signalPassFailure();

    if (failed(addMemoryResources(loop.getOperation(), loop.getAfter(), problem,
                                  resourceMap, resourceLimits)))
      return signalPassFailure();

    addPredicateDependencies(loop.getOperation(), loop.getAfter(), problem,
                             predicateMap, predicateUse);

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveChainingSharedOperatorsProblem(loop.getAfter(), problem,
                                                   cycleTime)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createLoopScheduleSequential(loop, problem)))
      return signalPassFailure();
  }

  // Schedule whole function
  auto funcOp = cast<FuncOp>(getOperation());

  ResourceMap resourceMap;
  ResourceLimits resourceLimits;
  if (failed(recordMemoryResources(funcOp.getOperation(), funcOp.getRegion(),
                                   resourceMap, resourceLimits)))
    return signalPassFailure();

  if (failed(ifOpConversion(funcOp.getOperation(), funcOp.getRegion(),
                            predicateMap)))
    return signalPassFailure();

  auto problem = getChainingSharedOperatorsProblem(funcOp, *dependenceAnalysis);

  // Populate the target operator types.
  if (failed(populateOperatorTypes(funcOp.getOperation(), funcOp.getBody(),
                                   problem)))
    return signalPassFailure();

  if (failed(addMemoryResources(funcOp.getOperation(), funcOp.getRegion(),
                                problem, resourceMap, resourceLimits)))
    return signalPassFailure();

  addPredicateDependencies(funcOp.getOperation(), funcOp.getRegion(), problem,
                           predicateMap, predicateUse);

  // Solve the scheduling problem computed by the analysis.
  if (failed(solveChainingSharedOperatorsProblem(funcOp.getBody(), problem,
                                                 cycleTime)))
    return signalPassFailure();

  // Convert the IR.
  if (failed(createFuncLoopSchedule(funcOp, problem)))
    return signalPassFailure();

  // Remove dependencies
  if (funcOp->hasAttrOfType<SymbolRefAttr>("loopschedule.dependencies")) {
    auto depSymbol =
        funcOp->getAttrOfType<SymbolRefAttr>("loopschedule.dependencies");
    auto *depOp = SymbolTable::lookupNearestSymbolFrom(funcOp, depSymbol);
    depOp->walk([](LoopScheduleAccessOp op) { op.erase(); });
    depOp->walk([](LoopScheduleDependsOnOp op) { op.erase(); });
    depOp->erase();
    funcOp->removeAttr("loopschedule.dependencies");
  }

  // Remove access names for now
  // TODO: Probably should make this an independent pass for debugging reasons
  funcOp->walk([](Operation *op) {
    if (op->hasAttrOfType<StringAttr>(NameAnalysis::getAttributeName())) {
      op->removeAttr(NameAnalysis::getAttributeName());
    }
  });
}

static bool onlyUserIsYield(Operation *op) {
  auto users = op->getUsers();
  if (std::distance(users.begin(), users.end()) != 1)
    return false;

  Operation *user = users.begin().getCurrent()->getOwner();

  return isa<scf::YieldOp>(user);
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult SCFToLoopSchedulePass::populateOperatorTypes(
    Operation *op, Region &loopBody, ChainingSharedOperatorsProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType freeOpr = problem.getOrInsertOperatorType("free");
  problem.setLatency(freeOpr, 0);
  problem.setIncomingDelay(freeOpr, 0);
  problem.setOutgoingDelay(freeOpr, 0);
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  problem.setIncomingDelay(combOpr, 0.2);
  problem.setOutgoingDelay(combOpr, 0.2);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  problem.setIncomingDelay(seqOpr, 0.5);
  problem.setOutgoingDelay(seqOpr, 0.5);
  Problem::OperatorType loopOpr = problem.getOrInsertOperatorType("loop");
  problem.setLatency(loopOpr, 1);
  problem.setIncomingDelay(loopOpr, 0.0);
  problem.setOutgoingDelay(loopOpr, 0.0);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 4);
  problem.setIncomingDelay(mcOpr, 0.5);
  problem.setOutgoingDelay(mcOpr, 0.5);
  Problem::OperatorType divOpr = problem.getOrInsertOperatorType("divider");
  problem.setLatency(divOpr, 36);
  problem.setIncomingDelay(divOpr, 0.5);
  problem.setOutgoingDelay(divOpr, 0.5);

  Operation *unsupported;
  WalkResult result = loopBody.walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr) {
      return WalkResult::advance();
    }

    auto potentialOperators =
        operatorLibraryAnalysis->getPotentialOperators(op);
    if (!potentialOperators.empty()) {
      // Just pick the first potential operator for now
      // TODO: Perform better allocation
      auto selectedOperator = potentialOperators.front();
      Problem::OperatorType libOpr =
          problem.getOrInsertOperatorType(selectedOperator);
      problem.setLatency(libOpr, operatorLibraryAnalysis->getOperatorLatency(
                                     selectedOperator));
      problem.setIncomingDelay(
          libOpr,
          operatorLibraryAnalysis->getOperatorIncomingDelay(selectedOperator)
              .value_or(0.0));
      problem.setOutgoingDelay(
          libOpr,
          operatorLibraryAnalysis->getOperatorOutgoingDelay(selectedOperator)
              .value_or(0.0));
      problem.setLinkedOperatorType(op, libOpr);
      op->setAttr("loopschedule.operator",
                  SymbolRefAttr::get(libOpr.getAttr()));
      return WalkResult::advance();
    }

    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::TruncIOp, IndexCastOp, memref::AllocaOp, memref::AllocOp,
              loopschedule::AllocInterface, YieldOp, func::ReturnOp,
              comb::ExtractOp>([&](Operation *freeOp) {
          // Some known free ops.
          problem.setLinkedOperatorType(freeOp, freeOpr);
          return WalkResult::advance();
        })
        .Case<ShLIOp, ShRSIOp, ShRUIOp>([&](Operation *shOp) {
          bool constant =
              llvm::any_of(shOp->getOpOperands(), [](auto &operand) {
                if (isa<BlockArgument>(operand.get())) {
                  return false;
                }
                return isa<arith::ConstantOp>(operand.get().getDefiningOp());
              });
          // Constant shifts are free
          if (constant) {
            problem.setLinkedOperatorType(shOp, freeOpr);
          } else {
            problem.setLinkedOperatorType(shOp, combOpr);
          }
          return WalkResult::advance();
        })
        .Case<CmpIOp, arith::SelectOp, AddIOp, SubIOp, AndIOp, XOrIOp>(
            [&](Operation *combOp) {
              // Some known combinational ops.
              problem.setLinkedOperatorType(combOp, combOpr);
              return WalkResult::advance();
            })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Multiplier
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Case<LoopScheduleBufferOp>([&](Operation *op) {
          problem.setLinkedOperatorType(op, seqOpr);
          return WalkResult::advance();
        })
        .Case<RemUIOp, RemSIOp, DivSIOp>([&](Operation *op) {
          // Divider ops
          auto bitwidth = op->getResult(0).getType().getIntOrFloatBitWidth();
          if (bitwidth != 32 && bitwidth != 64) {
            unsupported = op;
            return WalkResult::interrupt();
          }
          problem.setLinkedOperatorType(op, divOpr);
          return WalkResult::advance();
        })
        .Case<LoopScheduleStoreOp, AffineStoreOp>([&](Operation *memOp) {
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<LoopScheduleStoreOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          problem.setIncomingDelay(memOpr, 0.5);
          problem.setOutgoingDelay(memOpr, 0.5);
          return WalkResult::advance();
        })
        .Case<LoopScheduleLoadOp, AffineLoadOp>([&](Operation *memOp) {
          Value memRef = getMemref(memOp);
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          problem.setIncomingDelay(memOpr, 0.5);
          problem.setOutgoingDelay(memOpr, 0.5);
          return WalkResult::advance();
        })
        .Case<LoadInterface, StoreInterface>([&](Operation *op) {
          unsigned latency;
          std::string uniqueId;
          float incomingDelay = 0.0;
          float outgoingDelay = 0.0;
          if (auto loadOp = dyn_cast<loopschedule::LoadInterface>(*op)) {
            latency = loadOp.getLatency();
            uniqueId = loadOp.getUniqueId();
            incomingDelay = loadOp.getIncomingDelay();
            outgoingDelay = loadOp.getOutgoingDelay();
          } else {
            auto storeOp = cast<loopschedule::StoreInterface>(*op);
            latency = storeOp.getLatency();
            uniqueId = storeOp.getUniqueId();
            incomingDelay = storeOp.getIncomingDelay();
          }
          Problem::OperatorType portOpr =
              problem.getOrInsertOperatorType(uniqueId);
          problem.setLatency(portOpr, latency);
          problem.setLinkedOperatorType(op, portOpr);
          problem.setIncomingDelay(portOpr, incomingDelay);
          problem.setOutgoingDelay(portOpr, outgoingDelay);

          return WalkResult::advance();
        })
        .Case<loopschedule::SchedulableInterface>([&](Operation *op) {
          auto schedOp = cast<SchedulableInterface>(op);
          auto latency = schedOp.getOpLatency();
          auto limitOpt = schedOp.getOpLimit();
          Problem::OperatorType opr =
              problem.getOrInsertOperatorType(schedOp.getUniqueId());
          problem.setLatency(opr, latency);
          if (limitOpt.has_value()) {
            auto rsrc = problem.getOrInsertResourceType(schedOp.getUniqueId());
            problem.setLimit(rsrc, limitOpt.value());
            problem.addLinkedResourceType(op, rsrc);
          }
          problem.setLinkedOperatorType(op, opr);
          problem.setIncomingDelay(opr, schedOp.getIncomingDelay());
          problem.setOutgoingDelay(opr, schedOp.getOutgoingDelay());

          return WalkResult::advance();
        })
        .Case<LoopInterface>([&](Operation *loopOp) {
          problem.setLinkedOperatorType(loopOp, loopOpr);
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
LogicalResult SCFToLoopSchedulePass::solveChainingModuloProblem(
    scf::WhileOp &loop, ChainingModuloProblem &problem, float cycleTime) {
  // Scheduling analyis only considers the innermost loop nest for now.

  std::optional<int32_t> ii;
  if (auto iiAttr = loop->getAttrOfType<IntegerAttr>("hls.pipeline"))
    ii = iiAttr.getInt();

  LLVM_DEBUG(loop.dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(for (auto *op
                  : problem.getOperations()) {
    // if (auto parent = op->getParentOfType<LoopInterface>(); parent)
    //   continue;
    llvm::dbgs() << "Chaining Modulo scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr->getAttr();
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    auto maybeRsrcs = problem.getLinkedResourceTypes(op);
    if (maybeRsrcs.has_value()) {
      for (auto rsrc : *maybeRsrcs)
        llvm::dbgs() << "\n  resource = " << rsrc.getAttr()
                     << " limit = " << problem.getLimit(rsrc);
    }
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  });

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = loop.getAfterBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor, cycleTime, ii)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Verify II
  if (ii.has_value() && problem.getInitiationInterval() != ii) {
    return loop.emitError(
        "Failed to schedule for desired II of " + std::to_string(ii.value()) +
        ", minimum II is " +
        std::to_string(problem.getInitiationInterval().value()));
  }

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    for (auto *op : problem.getOperations()) {
      if (auto parent = op->getParentOfType<LoopInterface>(); parent)
        continue;
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    }
  });

  return success();
}

LogicalResult SCFToLoopSchedulePass::solveChainingSharedOperatorsProblem(
    Region &region, ChainingSharedOperatorsProblem &problem, float cycleTime) {

  LLVM_DEBUG(region.getParentOp()->dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(for (auto op
                  : problem.getOperations()) {
    llvm::dbgs() << "Chaining Shared Operator scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr->getAttr();
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    auto maybeRsrcs = problem.getLinkedResourceTypes(op);
    if (maybeRsrcs.has_value()) {
      for (auto rsrc : *maybeRsrcs)
        llvm::dbgs() << "\n  resource = " << rsrc.getAttr()
                     << " limit = " << problem.getLimit(rsrc);
    }
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { "
                     << "source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  });

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = region.back().getTerminator();
  if (failed(scheduleSimplex(problem, anchor, cycleTime)))
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

/// Create the pipeline op for a loop nest.
LogicalResult
SCFToLoopSchedulePass::createLoopSchedulePipeline(scf::WhileOp &loop,
                                                  CyclicProblem &problem) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  builder.setInsertionPointToStart(
      &loop->getParentOfType<FuncOp>().getBody().front());

  // Create Values for the loop's lower and upper bounds.
  // Value lowerBound = loop.getLowerBound();
  // Value upperBound = loop.getUpperBound();
  // Value step = loop.getStep();

  builder.setInsertionPoint(loop);

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = loop.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  // iterArgs.push_back(lowerBound);
  iterArgs.append(loop.getInits().begin(), loop.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount =
          loop->getAttrOfType<IntegerAttr>("loopschedule.trip_count"))
    tripCountAttr = tripCount;

  auto pipeline = builder.create<LoopSchedulePipelineOp>(
      resultTypes, ii, tripCountAttr, iterArgs);

  // Stash a reference to the scf.before body and its terminator. We will
  // clone the combinational condition ops into the first pipeline stage and
  // register the resulting i1 as an extra result of that stage. The
  // loopschedule.terminator's `condition` operand will be that stage result.
  auto scfCond = cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());

  // Add the non-yield and non-if operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<YieldOp, IfOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // Nested loops are not supported yet.
  assert(iterArgs.size() == loop.getAfterBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    valueMap.map(loop.getAfterBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));
    // The before-body's combinational ops (which compute the loop condition)
    // will be cloned into the first stage and likewise reference the
    // pipeline's iter_args via the stages block arguments.
    valueMap.map(loop.getBeforeArguments()[i],
                 pipeline.getStagesBlock().getArgument(i));
  }

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
      return isa<YieldOp>(op) && op->getParentOp() == loop;
    };

    // Initialize set of registers up until this point in time
    for (unsigned i = registerValues.size(); i <= startTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());

    // Check each operation to see if its results need plumbing
    for (auto *op : group) {
      if (op->getUsers().empty()) {
        if (llvm::none_of(op->getResults(),
                          [&](Value v) { return predicateUse.contains(v); })) {
          continue;
        }
      }

      unsigned pipeEndTime = 0;
      SmallVector<Operation *> users;
      users.append(op->getUsers().begin(), op->getUsers().end());

      // Also check predicate users
      for (auto res : op->getResults()) {
        users.append(predicateUse.lookup(res));
      }
      for (auto *user : users) {
        unsigned userStartTime = *problem.getStartTime(user);
        // if (isLoopTerminator(user)) {
        //   op->dump();
        //   // Manually forward the value into the terminator's valueMap
        //   pipeEndTime = std::max(
        // } else if (*problem.getStartTime(user) > startTime)
        if (*problem.getStartTime(user) > startTime)
          pipeEndTime = std::max(pipeEndTime, userStartTime);
      }

      // Insert the range of pipeline stages the value needs to be valid for
      for (auto res : op->getResults())
        pipeTimes[res] = std::pair(startTime, pipeEndTime);

      // Add register stages for each time slice we need to pipe to
      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Keep a collection of this stages results as keys to our valueMaps
      for (auto result : op->getResults()) {
        bool registered = false;
        for (auto *user : result.getUsers()) {
          auto inThisGroup = false;
          for (auto *op : group) {
            if (user == op) {
              inThisGroup = true;
              break;
            }
          }
          if (!inThisGroup) {
            registerValues[startTime].push_back(result);
            registered = true;
            break;
          }
        }

        // Also keep around results that are used as predicates
        if (!registered) {
          for (auto *user : predicateUse.lookup(result)) {
            auto inThisGroup = false;
            for (auto *op : group) {
              if (user == op) {
                inThisGroup = true;
                break;
              }
            }
            if (!inThisGroup) {
              registerValues[startTime].push_back(result);
              break;
            }
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

  for (auto it : enumerate(loop.getAfter().getArguments())) {
    auto iterArg = it.value();
    if (iterArg.getUsers().empty())
      continue;

    unsigned startPipeTime = 0;

    auto *term = loop.getAfterBody()->getTerminator();
    auto &termOperand = term->getOpOperand(it.index());
    auto *definingOp = termOperand.get().getDefiningOp();
    assert(definingOp != nullptr);
    startPipeTime = *problem.getStartTime(definingOp);

    unsigned pipeEndTime = 0;
    for (auto *user : iterArg.getUsers()) {
      unsigned userStartTime = *problem.getStartTime(user);
      if (userStartTime >= startPipeTime) {
        pipeEndTime = std::max(pipeEndTime, userStartTime);
      }
    }

    // Do not need to pipe result if there are no later uses
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

  startTimes.clear();
  startTimes.append(newStartTimes.begin(), newStartTimes.end());
  llvm::sort(startTimes);
  SmallVector<bool> iterArgNeedsForwarding;
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    iterArgNeedsForwarding.push_back(false);
  }
  // The first stage in `startTimes` order will host the loop condition's
  // combinational ops; track the resulting stage result so the
  // loopschedule.terminator can reference it.
  Value pipelineCondResult;

  // Create stages along with maps
  for (auto i : enumerate(startTimes)) {
    auto startTime = i.value();
    auto lastStage = i.index() == startTimes.size() - 1;
    auto isFirstStage = i.index() == 0;
    auto group = startGroups[startTime];
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });
    auto stageTypes = registerTypes[startTime];
    if (isFirstStage)
      stageTypes.push_back(builder.getI1Type());
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
    // if (startTime == 0) {
    //   stageTypes.push_back(lowerBound.getType());
    // }

    // Create the stage itself. The pipeline op no longer has a default
    // terminator (we'll create one after all stages are built), so insert at
    // end of the stages block.
    builder.setInsertionPointToEnd(&stagesBlock);
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
      OpBuilder::InsertionGuard g(builder);
      LoopScheduleIfOp ifOp;
      if (predicateMap.contains(op)) {
        Value cond = predicateMap.lookup(op);
        if (stageValueMaps[startTime].contains(cond))
          cond = stageValueMaps[startTime].lookup(cond);
        ifOp = builder.create<LoopScheduleIfOp>(op->getLoc(),
                                                op->getResultTypes(), cond);
        builder.setInsertionPointToStart(&ifOp.getBody().front());
      }
      auto *newOp = builder.clone(*op, stageValueMaps[startTime]);
      dependenceAnalysis->replaceOp(op, newOp);
      if (predicateMap.contains(op)) {
        if (!newOp->getResults().empty())
          builder.create<LoopScheduleYieldOp>(op->getLoc(),
                                              newOp->getResults());
        newOp = ifOp;
      }

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

    // For the first stage, clone the scf.before-body's combinational
    // condition ops into this stage and register the resulting i1 condition
    // as the stage's last result.
    if (isFirstStage) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(stageTerminator);
      for (auto &op : loop.getBeforeBody()->getOperations()) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        builder.clone(op, stageValueMaps[startTime]);
      }
      Value mappedCond = stageValueMaps[startTime].lookup(scfCond.getCondition());
      stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                      mappedCond);
      pipelineCondResult = stage.getResult(stage.getNumResults() - 1);
    }
  }

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;

  for (auto value : loop.getAfterBody()->getTerminator()->getOperands()) {
    unsigned lookupTime =
        std::min((unsigned)(stageValueMaps.size() - 1),
                 (unsigned)(pipeTimes[value].first + ii.getInt()));

    termIterArgs.push_back(stageValueMaps[lookupTime].lookup(value));
    termResults.push_back(stageValueMaps[lookupTime].lookup(value));
  }

  // Build the loopschedule.terminator with the condition produced by the
  // first stage plus the iter_args and results.
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<LoopScheduleTerminatorOp>(pipelineCondResult, termIterArgs,
                                           termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < loop.getNumResults(); ++i)
    loop.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  dependenceAnalysis->replaceOp(loop, pipeline);

  loop.walk([&](Operation *op) {
    if (!isa<scf::IfOp>(op)) {
      assert(!dependenceAnalysis->containsOp(op));
    }
  });

  // Remove the loop nest from the IR.
  loop.walk([&](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

static DenseMap<int64_t, SmallVector<Operation *>>
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

/// Create loopschedule seq op for a sequential loop
LogicalResult
SCFToLoopSchedulePass::createLoopScheduleSequential(scf::WhileOp &loop,
                                                    Problem &problem) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  builder.setInsertionPointToStart(
      &loop->getParentOfType<FuncOp>().getBody().front());

  builder.setInsertionPoint(loop);

  auto *anchor = loop.getAfterBody()->getTerminator();

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = loop.getResultTypes();

  SmallVector<Value> iterArgs;
  // iterArgs.push_back(lowerBound);
  iterArgs.append(loop.getInits().begin(), loop.getInits().end());

  // If possible, attach a constant trip count attribute.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount =
          loop->getAttrOfType<IntegerAttr>("loopschedule.trip_count"))
    tripCountAttr = tripCount;

  auto opMap = getOperationCycleMap(problem);

  auto sequential = builder.create<LoopScheduleSequentialOp>(
      loop.getLoc(), resultTypes, tripCountAttr, iterArgs);

  // Stash a reference to the scf.before terminator. We'll clone the
  // combinational condition ops into the first loopschedule.step body and
  // register the resulting i1 as an extra result of that step. The
  // loopschedule.terminator's `condition` operand will be that step result.
  auto scfCond = cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());

  // Maintain mappings of values in the loop body and results of stages
  IRMapping valueMap;

  builder.setInsertionPointToStart(&sequential.getScheduleBlock());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  unsigned endTime = 0;
  for (auto *op : problem.getOperations()) {
    if (isa<YieldOp, IfOp>(op))
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

            // Forward values used for predicates as well
            if (predicateUse.contains(operand.get()))
              return true;
          }
        }
      }
    }
    return false;
  };

  auto valueHasLaterUse = [&](Value v, uint32_t resTime) {
    for (uint32_t i = resTime + 1; i < endTime; ++i) {
      if (startGroups.contains(i)) {
        auto startGroup = startGroups[i];
        for (auto *operation : startGroup) {
          for (auto &operand : operation->getOpOperands()) {
            if (operand.get() == v) {
              return true;
            }

            // Forward values used for predicates as well
            if (predicateUse.contains(operand.get()))
              return true;
          }
        }
      }
    }
    return false;
  };

  // Must re-register return values of memories if they are used later
  for (auto *op : problem.getOperations()) {
    if (isa<LoopScheduleLoadOp>(op)) {
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

  if (!loop.getAfter().getArgument(0).getUsers().empty()) {
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
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    valueMap.map(loop.getAfter().getArgument(i),
                 sequential.getScheduleBlock().getArgument(i));
    // The before-body's combinational ops (which compute the loop condition)
    // will be cloned into the first step and likewise reference the
    // sequential loop's iter_args via the schedule block arguments.
    valueMap.map(loop.getBeforeArguments()[i],
                 sequential.getScheduleBlock().getArgument(i));
  }

  // Create the stages.
  builder.setInsertionPointToStart(&scheduleBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  // === Bucket merge for loopschedule.delay emission ===
  //
  // When `enableDelayMerging` is on, coalesce groups whose start times fall
  // within an earlier group's running latency into a single multi-cycle step.
  // The base group's ops live at offset 0 inside the step body; merged groups
  // are wrapped in a `loopschedule.delay` region with latency = offset.
  //
  // After the merge, `startGroups[t]` only contains the offset-0 ops for the
  // bucket whose baseTime is `t`, and `delayOffsetGroups[t]` (only set for
  // base times of merged buckets) contains a map from offset>0 to the ops
  // that should appear inside that delay region.
  DenseMap<uint32_t, std::map<uint32_t, SmallVector<Operation *>>>
      delayOffsetGroups;
  if (enableDelayMerging) {
    auto opLatency = [&](Operation *op) -> uint32_t {
      auto opr = problem.getLinkedOperatorType(op);
      if (!opr)
        return 1;
      auto lat = problem.getLatency(*opr);
      return lat.value_or(1);
    };
    auto groupMaxLat = [&](ArrayRef<Operation *> g) -> uint32_t {
      uint32_t m = 1;
      for (auto *op : g)
        m = std::max(m, opLatency(op));
      return m;
    };
    auto groupContainsLoop = [](ArrayRef<Operation *> g) {
      for (auto *op : g)
        if (isa<LoopInterface>(op))
          return true;
      return false;
    };

    // Collect ops that produce loop iter_args (operands of the after-body
    // terminator). These must remain at offset 0 of their bucket so the
    // existing iter_arg plumbing keeps working.
    DenseSet<Operation *> iterArgProducers;
    for (auto &operand : anchor->getOpOperands()) {
      if (auto *def = operand.get().getDefiningOp())
        iterArgProducers.insert(def);
    }
    auto groupHasIterArgProducer = [&](ArrayRef<Operation *> g) {
      for (auto *op : g)
        if (iterArgProducers.count(op))
          return true;
      return false;
    };

    // Collect ops referenced by the loop's combinational condition. These
    // also must remain at offset 0 of the first bucket so the cond is
    // available at cycle 0 of the first step.
    DenseSet<Operation *> condTransitiveDeps;
    {
      std::queue<Value> worklist;
      worklist.push(scfCond.getCondition());
      DenseSet<Value> seen;
      while (!worklist.empty()) {
        Value v = worklist.front();
        worklist.pop();
        if (!seen.insert(v).second)
          continue;
        auto *def = v.getDefiningOp();
        if (!def)
          continue;
        if (def->getParentRegion() == &loop.getBefore()) {
          for (auto operand : def->getOperands())
            worklist.push(operand);
          continue;
        }
        condTransitiveDeps.insert(def);
      }
    }
    auto groupHasCondDep = [&](ArrayRef<Operation *> g) {
      for (auto *op : g)
        if (condTransitiveDeps.count(op))
          return true;
      return false;
    };

    struct StepBucket {
      uint32_t baseTime;
      uint32_t latency;
      std::map<uint32_t, SmallVector<Operation *>> opsByOffset;
    };
    SmallVector<StepBucket> buckets;
    for (auto t : startTimes) {
      auto &group = startGroups[t];
      uint32_t lat = groupMaxLat(group);
      bool merged = false;
      if (!group.empty() && !groupContainsLoop(group) &&
          !groupHasIterArgProducer(group) && !groupHasCondDep(group)) {
        for (auto &b : llvm::reverse(buckets)) {
          auto it0 = b.opsByOffset.find(0);
          if (it0 != b.opsByOffset.end() && groupContainsLoop(it0->second))
            continue;
          if (b.baseTime <= t && t < b.baseTime + b.latency) {
            uint32_t off = t - b.baseTime;
            b.opsByOffset[off].append(group.begin(), group.end());
            uint32_t newLat = off + lat;
            if (newLat > b.latency)
              b.latency = newLat;
            merged = true;
            break;
          }
        }
      }
      if (!merged) {
        StepBucket b;
        b.baseTime = t;
        b.latency = lat;
        b.opsByOffset[0] = group;
        buckets.push_back(b);
      }
    }

    // Rewrite startGroups / startTimes to contain only bucket baseTimes with
    // their offset-0 ops, and stash offset>0 groups in `delayOffsetGroups`.
    DenseSet<uint32_t> bucketBases;
    for (auto &b : buckets)
      bucketBases.insert(b.baseTime);
    SmallVector<unsigned> toErase;
    for (auto &kv : startGroups)
      if (!bucketBases.count(kv.first))
        toErase.push_back(kv.first);
    for (auto t : toErase)
      startGroups.erase(t);
    for (auto &b : buckets) {
      startGroups[b.baseTime] = b.opsByOffset[0];
      for (auto &kv : b.opsByOffset)
        if (kv.first != 0)
          delayOffsetGroups[b.baseTime][kv.first] = kv.second;
    }
    startTimes.clear();
    for (auto &b : buckets)
      startTimes.push_back(b.baseTime);
    llvm::sort(startTimes);
  }

  DenseMap<uint32_t, SmallVector<Value>> reregisterValues;

  LoopScheduleStepOp lastStep;
  // The first step in `startTimes` order will host the loop condition's
  // combinational ops; track the resulting step result so the
  // loopschedule.terminator can reference it.
  Value sequentialCondResult;
  DominanceInfo dom(getOperation());
  for (auto i : enumerate(startTimes)) {
    auto startTime = i.value();
    auto isFirstStep = i.index() == 0;
    auto group = startGroups[startTime];
    OpBuilder::InsertionGuard g(builder);

    auto isLoopTerminator = [loop](Operation *op) {
      return isa<YieldOp>(op) && op->getParentOp() == loop;
    };

    // Collect the return types for this stage. Operations whose results are not
    // exclusively used within this stage are returned.
    SmallVector<Type> stepTypes;
    DenseSet<Operation *> opsWithReturns;
    for (auto *op : group) {
      SmallVector<Operation *, 10> users;
      users.append(op->getUsers().begin(), op->getUsers().end());
      for (auto res : op->getResults()) {
        auto predUsers = predicateUse.lookup(res);
        users.append(predUsers.begin(), predUsers.end());
      }
      for (auto *user : users) {
        auto *userOrAncestor = loop.getAfter().findAncestorOpInRegion(*user);
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

      // Add return types for iter_args that are updated in this step but have
      // later uses.
      for (auto &operand : anchor->getOpOperands()) {
        auto iterArgNum = operand.getOperandNumber();
        auto iterArg = loop.getAfterArguments()[iterArgNum];
        if (operand.get().getDefiningOp() == op &&
            valueHasLaterUse(iterArg, startTime)) {
          stepTypes.push_back(iterArg.getType());
        }
      }
    }

    // Add return types for values we already know need to be reregistered.
    for (auto val : reregisterValues[startTime]) {
      stepTypes.push_back(val.getType());
    }

    // Delay-region exports: each op in a delay region whose result has any
    // user with a later original startTime needs its result to escape via
    // the delay's register AND become a step result. We follow the same
    // criterion as opsWithReturns above; in-bucket higher-offset uses are a
    // (harmless) false positive — they'd already get the value via the
    // delay op's own result through valueMap.
    SmallVector<Operation *> delayExportOps; // ops whose results escape the step
    DenseMap<Operation *, unsigned>
        delayExportFirstStepIdx; // op → first step result index
    if (delayOffsetGroups.count(startTime)) {
      for (auto &offsetEntry : delayOffsetGroups[startTime]) {
        for (auto *op : offsetEntry.second) {
          bool exports = false;
          for (auto *user : op->getUsers()) {
            auto *ua = loop.getAfter().findAncestorOpInRegion(*user);
            auto ut = problem.getStartTime(ua);
            if ((ut.has_value() && *ut > startTime) ||
                isLoopTerminator(user)) {
              exports = true;
              break;
            }
          }
          if (exports) {
            delayExportFirstStepIdx[op] = stepTypes.size();
            for (auto t : op->getResultTypes())
              stepTypes.push_back(t);
            delayExportOps.push_back(op);
          }
        }
      }
    }

    // The first step gains an extra i1 result for the loop condition.
    if (isFirstStep)
      stepTypes.push_back(builder.getI1Type());

    // Create the step itself.
    auto step = builder.create<LoopScheduleStepOp>(stepTypes);
    auto &stepBlock = step.getBodyBlock();
    auto *stepTerminator = stepBlock.getTerminator();
    builder.setInsertionPointToStart(&stepBlock);

    // Sort the group according to original dominance.
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });

    SmallVector<std::pair<Value, Value>> newIterArgs;

    // Move over the operations and add their results to the terminator.
    SmallVector<std::tuple<Operation *, Operation *, unsigned>> movedOps;
    for (auto *op : group) {
      unsigned resultIndex = stepTerminator->getNumOperands();
      OpBuilder::InsertionGuard g(builder);
      LoopScheduleIfOp ifOp;
      if (predicateMap.contains(op)) {
        Value cond = predicateMap.lookup(op);
        cond = valueMap.lookupOrDefault(cond);
        ifOp = builder.create<LoopScheduleIfOp>(op->getLoc(),
                                                op->getResultTypes(), cond);
        builder.setInsertionPointToStart(&ifOp.getBody().front());
      }
      auto *newOp = builder.clone(*op, valueMap);
      dependenceAnalysis->replaceOp(op, newOp);
      // Stamp per-op cycle latency so backends know how many cycles a
      // multi-cycle operator (e.g. iterative multiplier) consumes within
      // its step. Only stamped when > 1 to keep IR tidy.
      if (auto opr = problem.getLinkedOperatorType(op)) {
        unsigned lat = problem.getLatency(*opr).value_or(1);
        if (lat > 1)
          newOp->setAttr("loopschedule.cycle_latency",
                         builder.getI64IntegerAttr(lat));
      }
      if (predicateMap.contains(op)) {
        if (!newOp->getResults().empty())
          builder.create<LoopScheduleYieldOp>(op->getLoc(),
                                              newOp->getResults());
        newOp = ifOp;
      }

      std::queue<Operation *> oldOps;
      op->walk([&](Operation *op) { oldOps.push(op); });
      if (isa<LoopInterface>(newOp)) {
        newOp->walk([&](Operation *op) {
          Operation *oldOp = oldOps.front();
          dependenceAnalysis->replaceOp(oldOp, op);
          oldOps.pop();
        });
      }
      if (opsWithReturns.contains(op)) {
        stepTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }

      resultIndex = stepTerminator->getNumOperands();

      // Handle iter_args with later uses
      for (auto &operand : anchor->getOpOperands()) {
        auto iterArgNum = operand.getOperandNumber();
        auto iterArg = loop.getAfterArguments()[iterArgNum];
        if (operand.get().getDefiningOp() == op &&
            valueHasLaterUse(iterArg, startTime)) {
          auto newIterArg = valueMap.lookup(iterArg);
          stepTerminator->insertOperands(resultIndex,
                                         SmallVector<Value>{newIterArg});
          newIterArgs.emplace_back(iterArg, step->getResult(resultIndex));
        }
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

    // Emit delay regions for offsets > 0 in this bucket. Each delay op
    // wraps the ops scheduled at offset cycles into the step. Every cloned
    // op's results are registered as delay results so subsequent in-bucket
    // ops can reference them via valueMap. Ops whose results were marked
    // for export to the step (delayExportFirstStepIdx) also have their
    // delay results inserted into the step's terminator.
    if (delayOffsetGroups.count(startTime)) {
      OpBuilder::InsertionGuard guard(builder);
      // Insert delay ops at the end of the step body (just before the
      // implicit register terminator).
      builder.setInsertionPoint(stepTerminator);
      for (auto &offsetEntry : delayOffsetGroups[startTime]) {
        uint32_t offset = offsetEntry.first;
        auto &delayOps = offsetEntry.second;

        // Sort by dominance for stable ordering inside the delay body.
        llvm::sort(delayOps, [&](Operation *a, Operation *b) {
          return dom.dominates(a, b);
        });

        // Compute delay result types: register every cloned op's results
        // (simple, slightly wasteful but correct).
        SmallVector<Type> delayResultTypes;
        for (auto *op : delayOps)
          for (auto t : op->getResultTypes())
            delayResultTypes.push_back(t);

        Location dLoc = delayOps.front()->getLoc();
        auto delayOp = LoopScheduleDelayOp::create(builder, dLoc,
                                                   (uint64_t)offset,
                                                   delayResultTypes);
        Block &delayBlock = delayOp.getBodyBlock();
        auto *delayTerminator = delayBlock.getTerminator();

        // Clone delay ops into the delay body.
        OpBuilder::InsertionGuard innerGuard(builder);
        builder.setInsertionPointToStart(&delayBlock);
        DenseMap<Operation *, Operation *> delayOldToNew;
        SmallVector<Value> delayRegOperands;
        for (auto *op : delayOps) {
          auto *newOp = builder.clone(*op, valueMap);
          dependenceAnalysis->replaceOp(op, newOp);
          if (auto opr = problem.getLinkedOperatorType(op)) {
            unsigned lat = problem.getLatency(*opr).value_or(1);
            if (lat > 1)
              newOp->setAttr("loopschedule.cycle_latency",
                             builder.getI64IntegerAttr(lat));
          }
          delayOldToNew[op] = newOp;
          // Update valueMap so subsequent in-delay ops see this clone.
          for (auto [orig, clone] :
               llvm::zip(op->getResults(), newOp->getResults()))
            valueMap.map(orig, clone);
          for (auto r : newOp->getResults())
            delayRegOperands.push_back(r);
        }
        // Set the delay's register terminator operands.
        delayTerminator->setOperands(delayRegOperands);

        // Map original op results to the delay op's external results so
        // later ops in the same step (in higher-offset delay regions) see
        // the delay op result through valueMap.
        unsigned delayResultIdx = 0;
        for (auto *op : delayOps) {
          for (auto orig : op->getResults()) {
            valueMap.map(orig, delayOp.getResult(delayResultIdx++));
          }
        }

        // For ops whose results escape this step, append the delay's
        // result to the step terminator at the pre-computed step index.
        // (We assume the existing iteration order over delayOffsetGroups
        // matches the order used during stepTypes computation, which it
        // does because both iterate the std::map in offset order.)
        for (auto *op : delayOps) {
          auto it = delayExportFirstStepIdx.find(op);
          if (it == delayExportFirstStepIdx.end())
            continue;
          // Find the delay op's results corresponding to this op.
          unsigned firstDelayIdx = 0;
          for (auto *o : delayOps) {
            if (o == op)
              break;
            firstDelayIdx += o->getNumResults();
          }
          for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
            unsigned stepIdx = it->second + i;
            (void)stepIdx;
            stepTerminator->insertOperands(
                stepTerminator->getNumOperands(),
                delayOp.getResult(firstDelayIdx + i));
          }
        }
      }
    }

    // For the first step, clone the scf.before-body's combinational
    // condition ops into this step and register the resulting i1 condition
    // as the step's last result.
    if (isFirstStep) {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(stepTerminator);
      for (auto &op : loop.getBeforeBody()->getOperations()) {
        if (op.hasTrait<OpTrait::IsTerminator>())
          continue;
        builder.clone(op, valueMap);
      }
      Value mappedCond = valueMap.lookup(scfCond.getCondition());
      stepTerminator->insertOperands(stepTerminator->getNumOperands(),
                                     mappedCond);
      sequentialCondResult = step->getResult(step->getNumResults() - 1);
    }

    // Add the step results to the value map for the original op.
    for (auto t : movedOps) {
      Operation *op = std::get<0>(t);
      Operation *newOp = std::get<1>(t);
      unsigned resultIndex = std::get<2>(t);
      for (size_t i = 0; i < newOp->getNumResults(); ++i) {
        auto newValue = step->getResult(resultIndex + i);
        auto oldValue = op->getResult(i);
        valueMap.map(oldValue, newValue);
      }
    }

    // Same for delay-region exports: map the original op result to the
    // step result so later steps reference the step boundary.
    for (auto *op : delayExportOps) {
      unsigned firstIdx = delayExportFirstStepIdx[op];
      for (unsigned i = 0, e = op->getNumResults(); i < e; ++i) {
        valueMap.map(op->getResult(i), step->getResult(firstIdx + i));
      }
    }

    // Handle iter_args with later uses
    for (auto iterArgPair : newIterArgs) {
      valueMap.map(std::get<0>(iterArgPair), std::get<1>(iterArgPair));
    }

    // Add values that need to be reregistered in the future
    for (auto *op : group) {
      if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
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
  }

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  for (int i = 0, vals = anchor->getNumOperands(); i < vals; ++i) {
    auto value = anchor->getOperand(i);
    termIterArgs.push_back(valueMap.lookup(value));
  }

  // Build the loopschedule.terminator with the condition produced by the
  // first step plus the iter_args and results.
  builder.setInsertionPointToEnd(&scheduleBlock);
  builder.create<LoopScheduleTerminatorOp>(sequentialCondResult, termIterArgs,
                                           termIterArgs);

  // Replace loop results with sequential results.
  for (size_t i = 0; i < loop.getNumResults(); ++i) {
    loop.getResult(i).replaceAllUsesWith(sequential.getResult(i));
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
LogicalResult SCFToLoopSchedulePass::createFuncLoopSchedule(FuncOp &funcOp,
                                                            Problem &problem) {
  auto *anchor = funcOp.getBody().back().getTerminator();

  auto opMap = getOperationCycleMap(problem);

  // auto outerLoop = loopNest.front();
  // auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(funcOp.getLoc(), funcOp);

  // Maintain mappings of values in the loop body and results of stages
  IRMapping valueMap;

  builder.setInsertionPointToStart(&funcOp.getBody().front());

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  unsigned endTime = 0;
  for (auto *op : problem.getOperations()) {
    if (isa<YieldOp, func::ReturnOp, memref::AllocaOp, arith::ConstantOp,
            memref::AllocOp, AllocInterface, IfOp>(op))
      continue;
    if (auto schedOp = dyn_cast<SchedulableInterface>(op)) {
      if (schedOp.isInitOp())
        continue;
    }
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
    if (startTime > endTime)
      endTime = *startTime;
  }

  auto usedByOperation = [&](Operation *op, Operation *maybeUser) {
    if (isa<MulIOp>(op)) {
      op->dump();
    }
    return llvm::any_of(maybeUser->getOperands(), [&](Value operand) {
      if (operand.getDefiningOp() == op)
        return true;

      // Forward values used for predicates as well
      if (predicateUse.contains(operand))
        return true;

      return false;
    });
  };

  auto hasLaterUse = [&](Operation *op, uint32_t resTime) {
    llvm::errs() << "op: \n";
    op->dump();
    llvm::errs() << "==============\n";
    llvm::errs() << "endTime: " << std::to_string(endTime) << "\n";
    for (uint32_t i = resTime + 1; i <= endTime; ++i) {
      llvm::errs() << "i: " << std::to_string(i) << "\n";
      if (startGroups.contains(i)) {
        auto startGroup = startGroups[i];
        for (auto *operation : startGroup) {
          // operation->dump();
          if (usedByOperation(op, operation))
            return true;
          auto wasInterrupted = operation->walk([&](Operation *inner) {
            if (usedByOperation(op, inner))
              return WalkResult::interrupt();
            return WalkResult::advance();
          });
          if (wasInterrupted.wasInterrupted())
            return true;
        }
      }
    }
    llvm::errs() << "no later use\n";
    return false;
  };

  // Must re-register return values of memories if they are used later
  for (auto *op : problem.getOperations()) {
    if (isa<LoopScheduleLoadOp>(op)) {
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

  SmallVector<SmallVector<Operation *>> scheduleGroups;
  auto totalLatency = problem.getStartTime(anchor).value();

  // Create the stages.
  Block &funcBlock = funcOp.getBody().front();
  auto *funcReturn = funcOp.getBody().back().getTerminator();
  builder.setInsertionPoint(funcReturn);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DenseMap<uint32_t, SmallVector<Value>> reregisterValues;

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
      SmallVector<Operation *, 10> users;
      users.append(op->getUsers().begin(), op->getUsers().end());
      for (auto res : op->getResults()) {
        auto predUsers = predicateUse.lookup(res);
        users.append(predUsers.begin(), predUsers.end());
      }
      for (auto *user : users) {
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

    for (auto val : reregisterValues[startTime]) {
      stepTypes.push_back(val.getType());
    }

    // Create the stage itself.
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
      OpBuilder::InsertionGuard g(builder);
      LoopScheduleIfOp ifOp;
      if (predicateMap.contains(op)) {
        Value cond = predicateMap.lookup(op);
        cond = valueMap.lookupOrDefault(cond);
        ifOp = builder.create<LoopScheduleIfOp>(op->getLoc(),
                                                op->getResultTypes(), cond);
        builder.setInsertionPointToStart(&ifOp.getBody().front());
      }
      auto *newOp = builder.clone(*op, valueMap);
      dependenceAnalysis->replaceOp(op, newOp);
      if (predicateMap.contains(op)) {
        if (!newOp->getResults().empty())
          builder.create<LoopScheduleYieldOp>(op->getLoc(),
                                              newOp->getResults());
        newOp = ifOp;
      }
      if (opsWithReturns.contains(op)) {
        stepTerminator->insertOperands(resultIndex, newOp->getResults());
        movedOps.emplace_back(op, newOp, resultIndex);
      }
      // All further uses in this step should used the cloned-version of values
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

    // Add the stage results to the value map for the original op.
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
      if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
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

std::unique_ptr<OperationPass<FuncOp>>
circt::createSCFToLoopSchedulePass(const SCFToLoopScheduleOptions &options) {
  return std::make_unique<SCFToLoopSchedulePass>(options);
}
