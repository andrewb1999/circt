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
                                           CyclicProblem &problem,
                                           Value condValue);
  LogicalResult createLoopScheduleSequential(scf::WhileOp &loop,
                                             Problem &problem,
                                             Value condValue);
  LogicalResult createFuncLoopSchedule(FuncOp &funcOp, Problem &problem);

  std::optional<LoopScheduleDependenceAnalysis> dependenceAnalysis;
  std::optional<OperatorLibraryAnalysis> operatorLibraryAnalysis;
  PredicateUse predicateUse;
  PredicateMap predicateMap;
};

/// Clone the `before` body's non-terminator ops into the start of the `after`
/// body so they participate in the scheduling problem. The `before` body
/// computes the loop condition from the iter_args; by inlining these ops into
/// the `after` body, the scheduler handles them as normal operations instead of
/// special-casing them. Returns the cloned condition value in the `after` body.
static Value inlineBeforeBodyOps(scf::WhileOp loop) {
  auto scfCond = cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
  Block *afterBody = loop.getAfterBody();
  IRMapping mapping;
  // before block args and after block args both represent the same iter_args.
  for (auto [beforeArg, afterArg] :
       llvm::zip(loop.getBeforeArguments(), loop.getAfterArguments()))
    mapping.map(beforeArg, afterArg);

  OpBuilder builder(loop.getContext());
  builder.setInsertionPointToStart(afterBody);
  for (auto &op : loop.getBeforeBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    builder.clone(op, mapping);
  }
  return mapping.lookup(scfCond.getCondition());
}

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

  // Map from while op to the cloned condition value in the after body.
  DenseMap<Operation *, Value> loopCondValues;

  // Schedule all pipelined loops first
  for (auto loop : llvm::make_early_inc_range(loops)) {
    // Inline the before-body condition ops into the after body so they
    // participate in the scheduling problem like any other operation.
    loopCondValues[loop] = inlineBeforeBodyOps(loop);

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
    if (failed(createLoopSchedulePipeline(loop, moduloProblem,
                                          loopCondValues[loop])))
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
    // Inline the before-body condition ops into the after body so they
    // participate in the scheduling problem like any other operation.
    loopCondValues[loop] = inlineBeforeBodyOps(loop);

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
    if (failed(createLoopScheduleSequential(loop, problem,
                                            loopCondValues[loop])))
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
                                                  CyclicProblem &problem,
                                                  Value condValue) {
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

  // Ensure the condition value gets registered as a stage result. Its only
  // consumer is the loopschedule.terminator (not yet created), so the forward-
  // piping loop above may not have added it.
  {
    Operation *condOp = condValue.getDefiningOp();
    unsigned condStartTime = *problem.getStartTime(condOp);
    for (unsigned i = registerValues.size(); i <= condStartTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());
    if (!llvm::is_contained(registerValues[condStartTime], condValue))
      registerValues[condStartTime].push_back(condValue);
    pipeTimes[condValue] = std::pair(condStartTime, condStartTime);
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
  // Track which stage result holds the condition value, set during stage
  // creation below once the condition op's stage is lowered.
  Value pipelineCondResult;

  // Create stages along with maps
  for (auto i : enumerate(startTimes)) {
    auto startTime = i.value();
    auto lastStage = i.index() == startTimes.size() - 1;
    auto group = startGroups[startTime];
    llvm::sort(group,
               [&](Operation *a, Operation *b) { return dom.dominates(a, b); });
    auto stageTypes = registerTypes[startTime];
    (void)lastStage;

    // Create the stage itself. The pipeline op no longer has a default
    // terminator (we'll create one after all stages are built), so insert at
    // end of the stages block.
    builder.setInsertionPointToEnd(&stagesBlock);
    auto startTimeAttr =
        builder.getIntegerAttr(builder.getIntegerType(64), startTime);
    auto stage = builder.create<LoopScheduleAtOp>(
        stageTypes, startTimeAttr);
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

    // If this stage contains the condition value, record its stage result
    // for the terminator.
    for (unsigned regIdx = 0; regIdx < registerValues[startTime].size();
         ++regIdx) {
      if (registerValues[startTime][regIdx] == condValue) {
        pipelineCondResult = stage.getResult(regIdx);
        break;
      }
    }
  }

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;

  for (auto it :
       llvm::enumerate(loop.getAfterBody()->getTerminator()->getOperands())) {
    unsigned i = it.index();
    Value value = it.value();
    unsigned lookupTime =
        std::min((unsigned)(stageValueMaps.size() - 1),
                 (unsigned)(pipeTimes[value].first + ii.getInt()));

    Value newValue = stageValueMaps[lookupTime].lookup(value);
    termIterArgs.push_back(newValue);
    termResults.push_back(newValue);

    // Emit an iter_arg_update inside the stage that produced the new value.
    // The LHS is the pipeline's iter-arg block argument for position `i`; the
    // RHS is the register-op operand that corresponds to `newValue` (the
    // stage result is not visible inside the stage itself).
    if (auto stage = newValue.getDefiningOp<LoopScheduleAtOp>()) {
      auto yieldOp = stage.getYieldOp();
      unsigned resultIdx = cast<OpResult>(newValue).getResultNumber();
      Value inside = yieldOp->getOperand(resultIdx);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(yieldOp);
      builder.create<LoopScheduleIterArgUpdateOp>(
          stage.getLoc(), pipeline.getStagesBlock().getArgument(i), inside);
    }
  }

  // Build the loopschedule.terminator with the condition and results.
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<LoopScheduleTerminatorOp>(pipelineCondResult, termResults,
                                           ValueRange{});

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
                                                    Problem &problem,
                                                    Value condValue) {
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
  }

  // Create the stages.
  builder.setInsertionPointToStart(&scheduleBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DenseMap<uint32_t, SmallVector<Value>> reregisterValues;

  // Track which frame result holds the condition value, set during frame
  // creation below once the condition op's frame is lowered.
  Value sequentialCondResult;
  Operation *condOp = condValue.getDefiningOp();
  DominanceInfo dom(getOperation());

  // === Partition buckets into phases using the SSA-dep rule. ===
  //
  // Walk buckets in startTimes order. A new phase starts when the current
  // bucket (or a recursively-traversed dependent op in the same bucket)
  // uses the SSA result of any LoopInterface op from an earlier bucket in
  // the current phase. Otherwise the bucket merges into the current phase.
  //
  // This keeps "cmpi + addi + launch" (common in AMC's sequential outer
  // loops) in one phase and only splits when a later bucket genuinely
  // consumes a launch's SSA result.
  SmallVector<SmallVector<unsigned>> phases;
  {
    DenseSet<Operation *> loopOpsInCurrentPhase;
    SmallVector<unsigned> currentPhase;
    auto bucketDependsOnLoopInPhase = [&](ArrayRef<Operation *> group) {
      if (loopOpsInCurrentPhase.empty())
        return false;
      // Check each op (and its children) in the bucket for any operand that
      // is a result of a LoopInterface op already in the current phase.
      for (auto *op : group) {
        bool dep = false;
        op->walk([&](Operation *inner) {
          for (Value operand : inner->getOperands()) {
            Operation *def = operand.getDefiningOp();
            if (def && loopOpsInCurrentPhase.count(def)) {
              dep = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (dep)
          return true;
      }
      return false;
    };
    for (auto t : startTimes) {
      auto &group = startGroups[t];
      bool splitHere = bucketDependsOnLoopInPhase(group) &&
                       !currentPhase.empty();
      if (splitHere) {
        phases.push_back(currentPhase);
        currentPhase.clear();
        loopOpsInCurrentPhase.clear();
      }
      currentPhase.push_back(t);
      for (auto *op : group)
        if (isa<LoopInterface>(op))
          loopOpsInCurrentPhase.insert(op);
    }
    if (!currentPhase.empty())
      phases.push_back(currentPhase);
  }

  // Per-frame bookkeeping: for each phase, we track the launch ops we created
  // (to thread handles via await / terminator-await).
  SmallVector<SmallVector<LoopScheduleLaunchOp>> phaseLaunches;
  phaseLaunches.resize(phases.size());

  // Map from bucket start-time to phase index. Lets the SSA-use checks below
  // distinguish "user in a later phase" (need to forward via await) from
  // "user in a later bucket within the same phase" (just map to cloned op).
  DenseMap<uint32_t, size_t> bucketTimeToPhase;
  for (auto phaseIdx : llvm::seq<size_t>(0, phases.size()))
    for (auto t : phases[phaseIdx])
      bucketTimeToPhase[t] = phaseIdx;

  auto isLoopTerminator = [loop](Operation *op) {
    return isa<YieldOp>(op) && op->getParentOp() == loop;
  };

  // Compute, for each original nested-loop op `lop`, the set of phase indices
  // whose static ops consume a result of `lop`. Plus a flag for "consumed by
  // the loop terminator". Used both to decide whether a forward is needed and
  // when to emit the corresponding `await`.
  auto computeLopUserPhases = [&](Operation *lop,
                                  DenseSet<size_t> &userPhases,
                                  bool &consumedByTerminator) {
    userPhases.clear();
    consumedByTerminator = false;
    for (auto res : lop->getResults()) {
      for (auto *user : res.getUsers()) {
        if (isLoopTerminator(user)) {
          consumedByTerminator = true;
          continue;
        }
        auto *userOrAncestor = loop.getAfter().findAncestorOpInRegion(*user);
        auto ut = problem.getStartTime(userOrAncestor);
        if (!ut.has_value())
          continue;
        auto it = bucketTimeToPhase.find(*ut);
        if (it != bucketTimeToPhase.end())
          userPhases.insert(it->second);
      }
    }
  };

  // Pending launches: launches whose handle is still live (not yet fully
  // awaited in a later phase) and whose results may still be needed for SSA
  // forwards or as a barrier. Each entry tracks:
  //   - origLoop: the original scf loop op
  //   - currentHandle: the most-recent SSA handle Value (initially the frame
  //     result; after being forwarded through subsequent frames, updated to
  //     that frame's forwarded result)
  //   - forwards: list of (resultIdx, origValue) pairs for SSA-forwarded
  //     results. origValue is the original scf loop's result that downstream
  //     phases want to consume.
  //   - remainingUserPhases: phase indices (>= the launch's phase) that still
  //     have a user that hasn't been awaited yet.
  //   - terminatorPending: true if the loop terminator consumes a result and
  //     has not yet been serviced.
  struct PendingLaunch {
    Operation *origLoop;
    Value currentHandle;
    SmallVector<std::pair<unsigned, Value>> forwards;
    DenseSet<size_t> remainingUserPhases;
    bool terminatorPending;
  };
  SmallVector<PendingLaunch> pendingLaunches;

  // Bodies-per-phase: whether THIS phase's body needs to forward a particular
  // pending launch's handle as a frame result (so a later phase can await it).
  // Populated at the start of each phase based on remainingUserPhases.
  // Tracks per-frame state for one phase; reset each iteration.

  for (auto phaseIdx : llvm::seq<size_t>(0, phases.size())) {
    auto &bucketTimes = phases[phaseIdx];
    uint32_t phaseBase = bucketTimes.front();

    // For each bucket we collect the static ops (non-LoopInterface) and the
    // nested LoopInterface ops separately.
    struct BucketContent {
      uint32_t offset; // bucketTime - phaseBase
      SmallVector<Operation *> staticOps;
      SmallVector<Operation *> loopOps;
    };
    SmallVector<BucketContent> buckets;
    for (auto t : bucketTimes) {
      BucketContent bc;
      bc.offset = t - phaseBase;
      for (auto *op : startGroups[t]) {
        if (isa<LoopInterface>(op))
          bc.loopOps.push_back(op);
        else
          bc.staticOps.push_back(op);
      }
      buckets.push_back(std::move(bc));
    }

    // --- Compute frame result types ---
    //
    // Contributions, in order:
    //  1) For each bucket (in offset order), for each static op in that
    //     bucket: op results that need to escape the frame (used by a later
    //     phase / loop-terminator / the seq-terminator condition), plus any
    //     iter-arg values that need to escape.
    //  2) Reregister values for this phase's base time.
    //  3) For each launch in any bucket: its handle value if a downstream
    //     phase will await it.

    // For each bucket, record which ops need export + the index-in-frame of
    // their first result.
    struct StaticExport {
      BucketContent *bucket;
      Operation *op;
      unsigned frameFirstIdx;
    };
    struct IterArgExport {
      BucketContent *bucket;
      Operation *op; // the op producing the iter-arg's new value
      Value iterArg;
      unsigned frameIdx;
    };
    SmallVector<StaticExport> staticExports;
    SmallVector<IterArgExport> iterArgExports;
    // Reregister exports (per-bucket-time).
    SmallVector<std::pair<uint32_t, Value>> reregExports;
    // Launches that need to produce a handle as a frame result.
    SmallVector<unsigned> launchHandleIdx; // bucket-flat index i -> frame result idx
    SmallVector<Type> stepTypes;

    bool phaseHasCondOp = false;
    for (auto &bc : buckets) {
      for (auto *op : bc.staticOps) {
        bool needsReturn = false;
        SmallVector<Operation *, 10> users;
        users.append(op->getUsers().begin(), op->getUsers().end());
        for (auto res : op->getResults()) {
          auto predUsers = predicateUse.lookup(res);
          users.append(predUsers.begin(), predUsers.end());
        }
        for (auto *user : users) {
          auto *userOrAncestor = loop.getAfter().findAncestorOpInRegion(*user);
          if (!userOrAncestor)
            continue;
          // The user (or its ancestor in the loop body) is inside this at's
          // region iff it's among the ops being cloned into this at.
          // Anything else — sibling launches at the same offset, ats at
          // later offsets, the loop terminator — escapes the at boundary
          // and must be reached via a yielded export.
          if (!llvm::is_contained(bc.staticOps, userOrAncestor)) {
            needsReturn = true;
            break;
          }
        }
        if (op == condOp) {
          needsReturn = true;
          phaseHasCondOp = true;
        }
        if (needsReturn) {
          StaticExport se{&bc, op, (unsigned)stepTypes.size()};
          staticExports.push_back(se);
          stepTypes.append(op->getResultTypes().begin(),
                           op->getResultTypes().end());
        }

        // Check iter-arg exports: if this op defines an operand of the loop's
        // terminator and that iter-arg has a later use, we need to forward
        // the updated value through the frame.
        for (auto &operand : anchor->getOpOperands()) {
          auto iterArgNum = operand.getOperandNumber();
          auto iterArg = loop.getAfterArguments()[iterArgNum];
          if (operand.get().getDefiningOp() == op &&
              valueHasLaterUse(iterArg, phaseBase + bc.offset)) {
            IterArgExport iae;
            iae.bucket = &bc;
            iae.op = op;
            iae.iterArg = iterArg;
            iae.frameIdx = stepTypes.size();
            iterArgExports.push_back(iae);
            stepTypes.push_back(iterArg.getType());
          }
        }
      }
    }

    // Reregister values are at the bucket-time they come due; anchored to the
    // frame base.
    for (auto &bc : buckets) {
      uint32_t bt = phaseBase + bc.offset;
      for (auto val : reregisterValues[bt]) {
        reregExports.emplace_back(bt, val);
        stepTypes.push_back(val.getType());
      }
    }

    // Determine which launches need their handle exposed as a frame result.
    // A launch's handle is exposed if a later phase awaits it.
    SmallVector<std::pair<BucketContent *, Operation *>>
        launchesInPhase; // (bucket, original loop op)
    for (auto &bc : buckets)
      for (auto *lop : bc.loopOps)
        launchesInPhase.emplace_back(&bc, lop);

    // For each launch in this phase, precompute the set of user phases so we
    // know whether to export the handle as a frame result. We also need this
    // when we emit the launch body yield.
    SmallVector<DenseSet<size_t>> launchUserPhasesInPhase;
    SmallVector<bool> launchTerminatorConsumedInPhase;
    for (auto &bc_lop : launchesInPhase) {
      DenseSet<size_t> userPhases;
      bool terminatorConsumed = false;
      computeLopUserPhases(bc_lop.second, userPhases, terminatorConsumed);
      launchUserPhasesInPhase.push_back(userPhases);
      launchTerminatorConsumedInPhase.push_back(terminatorConsumed);
    }

    // Always reserve a frame-result slot for every launch created in this
    // phase. Each launch's handle must be awaited exactly once downstream;
    // we materialize it as a frame result so a later phase's await region
    // (or the sequential terminator's await list) can consume it.
    SmallVector<std::optional<unsigned>> launchHandleIdxOpt;
    for (auto [i, bc_lop] : llvm::enumerate(launchesInPhase)) {
      (void)i, (void)bc_lop;
      launchHandleIdx.push_back(stepTypes.size());
      launchHandleIdxOpt.push_back(stepTypes.size());
      stepTypes.push_back(HandleType::get(builder.getContext()));
    }

    // For each pending launch (from earlier phases), decide:
    //   - awaitHere: if it has any SSA user in THIS phase → emit
    //     `await %h -> (T...)` in the await region. This services the
    //     launch (consumes the handle).
    //   - else: forward the handle through this frame as a frame result
    //     (`forwardHere = true` — unconditionally), so a later phase or
    //     the sequential terminator's await list can consume it.
    struct PendingService {
      size_t pendingIdx; // index into pendingLaunches
      bool awaitHere;    // emit await here; consumes the launch
      bool forwardHere;  // forward handle via body yield for future use
      // Forwards to emit on the await (only if awaitHere):
      SmallVector<std::pair<unsigned, Value>> forwards;
      // Frame result index assigned for forward (when forwardHere).
      std::optional<unsigned> forwardFrameIdx;
    };
    // Always await prior-phase launches in the immediately following phase.
    // This enforces the launch's ordering semantics (the user said: "anything
    // that depends on [a launch's] results should be put into a second frame
    // that awaits the loop", and for pure-memref ordering without SSA deps,
    // the barrier is required for correctness — otherwise launches would run
    // concurrently). Value forwards (for SSA consumers) are attached to the
    // await so a single op does both the barrier and the value-forwarding.
    SmallVector<PendingService> pendingServices;
    for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
      PendingService ps;
      ps.pendingIdx = pi;
      ps.awaitHere = true;
      ps.forwardHere = false;
      for (auto &fwd : pl.forwards) {
        // Forward the value through the await if it has a user in THIS phase.
        bool useHere = false;
        for (auto *user : fwd.second.getUsers()) {
          if (isLoopTerminator(user))
            continue;
          auto *userOrAncestor =
              loop.getAfter().findAncestorOpInRegion(*user);
          auto ut = problem.getStartTime(userOrAncestor);
          if (!ut.has_value())
            continue;
          auto it = bucketTimeToPhase.find(*ut);
          if (it != bucketTimeToPhase.end() && it->second == phaseIdx) {
            useHere = true;
            break;
          }
        }
        if (useHere)
          ps.forwards.push_back(fwd);
      }
      pendingServices.push_back(ps);
    }

    // --- Build frame ---
    auto frame = builder.create<LoopScheduleFrameOp>(stepTypes);

    // Await region: emit per-pending-launch awaits. Collect the yielded
    // values (these become the body block args).
    SmallVector<Value> awaitYieldValues;
    SmallVector<Type> bodyBlockArgTypes;
    // For each PendingService with awaitHere+forwards, track the first
    // body-block-arg index (so we can map origValues → bodyBlockArgs
    // after the body block is created).
    struct ServicedForwardInfo {
      size_t serviceIdx; // index into pendingServices
      unsigned firstBodyArgIdx;
    };
    SmallVector<ServicedForwardInfo> servicedForwards;
    {
      Block &awaitBlock = frame.getAwaitRegion().emplaceBlock();
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&awaitBlock);
      for (auto [si, ps] : llvm::enumerate(pendingServices)) {
        if (!ps.awaitHere)
          continue;
        auto &pl = pendingLaunches[ps.pendingIdx];
        SmallVector<Type> forwardTypes;
        for (auto &fwd : ps.forwards)
          forwardTypes.push_back(fwd.second.getType());
        auto awaitOp = builder.create<LoopScheduleAwaitOp>(
            pl.currentHandle.getLoc(), forwardTypes,
            ValueRange{pl.currentHandle});
        if (!ps.forwards.empty()) {
          ServicedForwardInfo sfi;
          sfi.serviceIdx = si;
          sfi.firstBodyArgIdx = bodyBlockArgTypes.size();
          servicedForwards.push_back(sfi);
          for (auto r : awaitOp.getResults()) {
            awaitYieldValues.push_back(r);
            bodyBlockArgTypes.push_back(r.getType());
          }
        }
      }
      builder.create<LoopScheduleYieldOp>(awaitYieldValues);
    }

    // Body region with yield placeholder. Add block args up front so body
    // ops can reference them via valueMap.
    Block &bodyBlock = frame.getBodyRegion().emplaceBlock();
    for (auto t : bodyBlockArgTypes)
      bodyBlock.addArgument(t, frame.getLoc());
    // Map the forwarded original values to the new body block args so
    // subsequent clones in this phase pick them up.
    for (auto &sfi : servicedForwards) {
      auto &ps = pendingServices[sfi.serviceIdx];
      for (auto [i, fwd] : llvm::enumerate(ps.forwards)) {
        Value bodyArg = bodyBlock.getArgument(sfi.firstBodyArgIdx + i);
        valueMap.map(fwd.second, bodyArg);
      }
    }
    LoopScheduleYieldOp bodyYield;
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&bodyBlock);
      bodyYield = builder.create<LoopScheduleYieldOp>();
    }

    // Body yield operands will be filled as we go. Build up a vector of
    // values of length stepTypes.size().
    SmallVector<Value> bodyYieldOperands(stepTypes.size(), Value());

    // Fill in forwarded handles (for pending launches that aren't awaited
    // here). The handle value lives outside the frame so we can't reference
    // it inside the body — but we want to forward it as a frame result.
    // To do that we also need a way to reference the handle inside the body.
    //
    // Trick: the await region can also yield handles, which then appear as
    // body block args. But that would "await" the handle, which is not what
    // we want for forwarding. Instead we use a separate mechanism: emit an
    // empty `loopschedule.at 0` whose yield gets the handle via... no, at
    // ops can't reference outside values that aren't captured through frame
    // block args either.
    //
    // Actually, the frame body is a plain region (not isolated from above),
    // so inside the body we CAN reference SSA values defined in the
    // enclosing scope. So we can directly yield `pl.currentHandle` from the
    // body yield. That's the simplest approach.
    for (auto &ps : pendingServices) {
      if (!ps.forwardHere)
        continue;
      auto &pl = pendingLaunches[ps.pendingIdx];
      bodyYieldOperands[*ps.forwardFrameIdx] = pl.currentHandle;
    }

    // Emit one `at offset` per bucket with static ops.
    DenseMap<uint32_t, LoopScheduleAtOp> atByOffset;

    // Emit ats in offset order, collecting results.
    for (auto &bc : buckets) {
      if (bc.staticOps.empty())
        continue;

      // Determine this at's result types: union of escaping ops' results and
      // iter-arg exports and reregister exports that reach from this
      // bucket's offset.
      SmallVector<Type> atTypes;
      // Track per-export the at-result index.
      struct AtSlot {
        enum { Static, IterArg, Rereg } kind;
        unsigned frameIdx;
        unsigned atIdx;
        // For Static: StaticExport*, iterArg: IterArgExport*, rereg: pair idx.
        StaticExport *se = nullptr;
        IterArgExport *iae = nullptr;
        Value reregVal;
      };
      SmallVector<AtSlot> slots;
      for (auto &se : staticExports) {
        if (se.bucket != &bc)
          continue;
        AtSlot s;
        s.kind = AtSlot::Static;
        s.frameIdx = se.frameFirstIdx;
        s.atIdx = atTypes.size();
        s.se = &se;
        for (auto t : se.op->getResultTypes())
          atTypes.push_back(t);
        slots.push_back(s);
      }
      for (auto &iae : iterArgExports) {
        if (iae.bucket != &bc)
          continue;
        AtSlot s;
        s.kind = AtSlot::IterArg;
        s.frameIdx = iae.frameIdx;
        s.atIdx = atTypes.size();
        s.iae = &iae;
        atTypes.push_back(iae.iterArg.getType());
        slots.push_back(s);
      }
      uint32_t bt = phaseBase + bc.offset;
      for (auto &re : reregExports) {
        if (re.first != bt)
          continue;
        // frame index of this rereg export
        unsigned rrFrameIdx = 0;
        // Walk reregExports to find re's frame index.
        {
          // reregExports frame indices are assigned after staticExports and
          // iterArgExports. Compute base = static-exports-count + iter-arg-exports-count,
          // then index within reregExports.
          unsigned base = 0;
          for (auto &s : staticExports)
            base += s.op->getNumResults();
          for (auto &iae : iterArgExports)
            (void)iae, base += 1;
          unsigned idx = 0;
          for (auto &re2 : reregExports) {
            if (&re2 == &re) {
              rrFrameIdx = base + idx;
              break;
            }
            ++idx;
          }
        }
        AtSlot s;
        s.kind = AtSlot::Rereg;
        s.frameIdx = rrFrameIdx;
        s.atIdx = atTypes.size();
        s.reregVal = re.second;
        atTypes.push_back(re.second.getType());
        slots.push_back(s);
      }

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(bodyYield);
      auto atOp = builder.create<LoopScheduleAtOp>(
          atTypes, builder.getI64IntegerAttr(bc.offset));
      atByOffset[bc.offset] = atOp;
      auto &atBlock = atOp.getBodyBlock();
      auto *atTerm = atBlock.getTerminator();
      builder.setInsertionPointToStart(&atBlock);

      // Sort static ops by dominance.
      auto staticOps = bc.staticOps;
      llvm::sort(staticOps, [&](Operation *a, Operation *b) {
        return dom.dominates(a, b);
      });

      // Clone each static op.
      DenseMap<Operation *, Operation *> oldToNew;
      for (auto *op : staticOps) {
        OpBuilder::InsertionGuard g2(builder);
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
        oldToNew[op] = newOp;
        // valueMap for in-at references.
        for (auto result : op->getResults())
          valueMap.map(result, newOp->getResult(result.getResultNumber()));
      }

      // Build at yield operands in slot order.
      SmallVector<Value> atYieldOperands(atTypes.size(), Value());
      for (auto &s : slots) {
        if (s.kind == AtSlot::Static) {
          auto *newOp = oldToNew.lookup(s.se->op);
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            atYieldOperands[s.atIdx + i] = newOp->getResult(i);
        } else if (s.kind == AtSlot::IterArg) {
          Value mapped = valueMap.lookup(s.iae->iterArg);
          atYieldOperands[s.atIdx] = mapped;
        } else {
          atYieldOperands[s.atIdx] = valueMap.lookup(s.reregVal);
        }
      }
      atTerm->setOperands(atYieldOperands);

      // Emit iter_arg_update ops inside the at for any iter-args this at
      // produces.
      for (auto &s : slots) {
        if (s.kind != AtSlot::IterArg)
          continue;
        OpBuilder::InsertionGuard ig(builder);
        builder.setInsertionPoint(atTerm);
        Value insideAt = atTerm->getOperand(s.atIdx);
        unsigned argIdx =
            cast<BlockArgument>(s.iae->iterArg).getArgNumber();
        builder.create<LoopScheduleIterArgUpdateOp>(
            atOp.getLoc(), sequential.getScheduleBlock().getArgument(argIdx),
            insideAt);
      }

      // Map each static-export's original op result to the at-op result
      // (visible to later ops in the same phase through valueMap); also use
      // the frame result for this op below.
      for (auto &s : slots) {
        if (s.kind != AtSlot::Static)
          continue;
        for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i) {
          // Update valueMap to point to the at result for intra-phase uses.
          valueMap.map(s.se->op->getResult(i), atOp.getResult(s.atIdx + i));
        }
      }
      // Reregister values: map to at result within this phase.
      for (auto &s : slots) {
        if (s.kind != AtSlot::Rereg)
          continue;
        valueMap.map(s.reregVal, atOp.getResult(s.atIdx));
      }

      // Forward at results to the frame body yield at the matching frameIdx.
      for (auto &s : slots) {
        if (s.kind == AtSlot::Static) {
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            bodyYieldOperands[s.frameIdx + i] = atOp.getResult(s.atIdx + i);
        } else if (s.kind == AtSlot::IterArg) {
          bodyYieldOperands[s.frameIdx] = atOp.getResult(s.atIdx);
        } else {
          bodyYieldOperands[s.frameIdx] = atOp.getResult(s.atIdx);
        }
      }
    }

    // Emit one `launch at offset` per LoopInterface op in this phase.
    SmallVector<LoopScheduleLaunchOp> createdLaunches;
    SmallVector<Operation *> createdLaunchClonedLops;
    for (auto [launchIdx, bc_lop] : llvm::enumerate(launchesInPhase)) {
      BucketContent *bc = bc_lop.first;
      Operation *lop = bc_lop.second;
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(bodyYield);
      auto launch = builder.create<LoopScheduleLaunchOp>(
          lop->getLoc(),
          HandleType::get(builder.getContext()),
          builder.getI64IntegerAttr(bc->offset));
      Block &launchBlock = launch.getBody().emplaceBlock();
      {
        OpBuilder::InsertionGuard gg(builder);
        builder.setInsertionPointToEnd(&launchBlock);
        builder.create<LoopScheduleYieldOp>();
      }
      auto *launchYield = launchBlock.getTerminator();
      builder.setInsertionPointToStart(&launchBlock);

      // Clone the nested loop op into the launch body.
      auto *newOp = builder.clone(*lop, valueMap);
      dependenceAnalysis->replaceOp(lop, newOp);
      if (auto opr = problem.getLinkedOperatorType(lop)) {
        unsigned lat = problem.getLatency(*opr).value_or(1);
        if (lat > 1)
          newOp->setAttr("loopschedule.cycle_latency",
                         builder.getI64IntegerAttr(lat));
      }
      // Walk nested ops into dependence analysis (as the old path did).
      std::queue<Operation *> oldOps;
      lop->walk([&](Operation *op) { oldOps.push(op); });
      if (isa<LoopInterface>(newOp)) {
        newOp->walk([&](Operation *op) {
          Operation *oldOp = oldOps.front();
          dependenceAnalysis->replaceOp(oldOp, op);
          oldOps.pop();
        });
      }

      // The launch body yield forwards the nested loop's results — this is
      // required by the dialect docs even though the launch SSA result is
      // just a handle.
      if (newOp->getNumResults() > 0) {
        launchYield->setOperands(newOp->getResults());
      }

      // Map same-phase uses directly to the cloned op's results so in-phase
      // higher-offset ats see them through valueMap.
      for (auto [orig, clone] :
           llvm::zip(lop->getResults(), newOp->getResults()))
        valueMap.map(orig, clone);

      createdLaunches.push_back(launch);
      createdLaunchClonedLops.push_back(newOp);
      phaseLaunches[phaseIdx].push_back(launch);
      if (launchHandleIdxOpt[launchIdx].has_value()) {
        bodyYieldOperands[*launchHandleIdxOpt[launchIdx]] = launch.getResult();
      }
    }

    bodyYield->setOperands(bodyYieldOperands);

    // Reorder frame body children by offset so ats/launches appear in
    // monotonically non-decreasing order. The emitter produced them in two
    // batches (ats first, then launches); stable-sort by offset here so the
    // frame verifier's monotonic-offset invariant holds.
    {
      SmallVector<Operation *> children;
      for (Operation &op : bodyBlock.without_terminator())
        children.push_back(&op);
      std::stable_sort(children.begin(), children.end(),
                       [](Operation *a, Operation *b) {
                         auto off = [](Operation *op) -> uint64_t {
                           if (auto at = dyn_cast<LoopScheduleAtOp>(op))
                             return at.getOffset();
                           return cast<LoopScheduleLaunchOp>(op).getOffset();
                         };
                         return off(a) < off(b);
                       });
      for (Operation *op : children)
        op->moveBefore(bodyYield);
    }

    // After the frame: update valueMap to point to frame results for exports
    // that escape the phase (so subsequent phases see frame boundaries).
    for (auto &se : staticExports) {
      for (unsigned i = 0, e = se.op->getNumResults(); i < e; ++i) {
        valueMap.map(se.op->getResult(i), frame->getResult(se.frameFirstIdx + i));
      }
      if (phaseHasCondOp && se.op == condOp) {
        unsigned condResNum = cast<OpResult>(condValue).getResultNumber();
        sequentialCondResult = frame->getResult(se.frameFirstIdx + condResNum);
      }
    }
    for (auto &iae : iterArgExports) {
      valueMap.map(iae.iterArg, frame->getResult(iae.frameIdx));
    }
    {
      unsigned base = 0;
      for (auto &s : staticExports)
        base += s.op->getNumResults();
      for (auto &iae : iterArgExports)
        (void)iae, base += 1;
      unsigned idx = 0;
      for (auto &re : reregExports) {
        valueMap.map(re.second, frame->getResult(base + idx));
        ++idx;
      }
    }

    // --- Update pendingLaunches after this frame is built ---
    //
    // Build a new pendingLaunches list for the next phase:
    //   - From prior pendingServices: retain any that are NOT awaited here
    //     AND still have remaining use (forwarded or not), updating
    //     currentHandle to the new frame result.
    //   - Add newly-created launches if they have cross-phase users or
    //     terminator consumers.
    SmallVector<PendingLaunch> nextPending;
    for (auto &ps : pendingServices) {
      if (ps.awaitHere)
        continue;
      auto pl = pendingLaunches[ps.pendingIdx];
      pl.currentHandle = frame->getResult(*ps.forwardFrameIdx);
      // Drop this phase from remaining (no op since we only track future
      // phases).
      pl.remainingUserPhases.erase(phaseIdx);
      // Keep the pending launch alive — even without any SSA user, the
      // handle will be consumed either in a later phase (SSA forward or
      // void await) or via the sequential terminator's await list.
      nextPending.push_back(pl);
    }
    for (auto [launchIdx, bc_lop] : llvm::enumerate(launchesInPhase)) {
      PendingLaunch pl;
      pl.origLoop = bc_lop.second;
      pl.currentHandle =
          frame->getResult(*launchHandleIdxOpt[launchIdx]);
      // Every result is a candidate forward (only emit in the await if a
      // downstream phase's user consumes it).
      for (auto res : bc_lop.second->getResults())
        pl.forwards.push_back({res.getResultNumber(), res});
      for (auto p : launchUserPhasesInPhase[launchIdx])
        if (p > phaseIdx)
          pl.remainingUserPhases.insert(p);
      pl.terminatorPending = launchTerminatorConsumedInPhase[launchIdx];
      nextPending.push_back(pl);
    }
    pendingLaunches = std::move(nextPending);

    // Post-phase: compute values to reregister for future bucket times.
    for (auto &bc : buckets) {
      uint32_t bt = phaseBase + bc.offset;
      for (auto *op : bc.staticOps) {
        if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
          if (hasLaterUse(op, bt + 1))
            reregisterValues[bt + 1].push_back(load.getResult());
        } else if (auto load = dyn_cast<LoadInterface>(op)) {
          auto latency = load.getLatency();
          if (hasLaterUse(op, bt + latency))
            reregisterValues[bt + latency].push_back(load.getResult());
        }
      }
    }
  }

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  for (int i = 0, vals = anchor->getNumOperands(); i < vals; ++i) {
    auto value = anchor->getOperand(i);
    Value newValue = valueMap.lookup(value);
    termIterArgs.push_back(newValue);

    // Emit an iter_arg_update inside the `at` that produced the new value.
    // The LHS is the sequential's iter-arg block argument for position `i`;
    // the RHS is the at-op's yield operand corresponding to `newValue`.
    if (auto frameOp = newValue.getDefiningOp<LoopScheduleFrameOp>()) {
      auto bodyYield = frameOp.getBodyYield();
      unsigned frameResultIdx = cast<OpResult>(newValue).getResultNumber();
      Value insideFrame = bodyYield->getOperand(frameResultIdx);
      if (auto at = insideFrame.getDefiningOp<LoopScheduleAtOp>()) {
        auto atYield = at.getYieldOp();
        unsigned atResultIdx = cast<OpResult>(insideFrame).getResultNumber();
        Value insideAt = atYield->getOperand(atResultIdx);
        // Avoid duplicates: if an iter_arg_update already exists in this at
        // for this iter-arg, skip.
        bool alreadyEmitted = false;
        at.getBodyBlock().walk([&](LoopScheduleIterArgUpdateOp u) {
          if (u.getIterArg() ==
              sequential.getScheduleBlock().getArgument(i)) {
            alreadyEmitted = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (!alreadyEmitted) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(atYield);
          builder.create<LoopScheduleIterArgUpdateOp>(
              at.getLoc(), sequential.getScheduleBlock().getArgument(i),
              insideAt);
        }
      }
    }
  }

  // Build the loopschedule.terminator with the condition produced by the
  // first step plus the loop results. Any launch handles left "dangling" in
  // the last phase are awaited at the iteration boundary via await(...).
  SmallVector<Value> termAwaitHandles;
  for (auto &pl : pendingLaunches)
    termAwaitHandles.push_back(pl.currentHandle);
  builder.setInsertionPointToEnd(&scheduleBlock);
  builder.create<LoopScheduleTerminatorOp>(sequentialCondResult, termIterArgs,
                                           termAwaitHandles);

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
    for (uint32_t i = resTime + 1; i <= endTime; ++i) {
      if (startGroups.contains(i)) {
        auto startGroup = startGroups[i];
        for (auto *operation : startGroup) {
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

  auto isFuncTerminator = [funcOp](Operation *op) {
    return isa<func::ReturnOp>(op) && op->getParentOp() == funcOp;
  };

  // === Partition buckets into phases using the close-after-launch-bucket
  // rule. This matches the user-provided target shape for vadd-affine. ===
  SmallVector<SmallVector<unsigned>> phases;
  {
    SmallVector<unsigned> currentPhase;
    for (auto t : startTimes) {
      currentPhase.push_back(t);
      bool hasLaunch = false;
      for (auto *op : startGroups[t])
        if (isa<LoopInterface>(op)) {
          hasLaunch = true;
          break;
        }
      if (hasLaunch) {
        phases.push_back(currentPhase);
        currentPhase.clear();
      }
    }
    if (!currentPhase.empty())
      phases.push_back(currentPhase);
  }

  DominanceInfo dom(getOperation());

  // Bucket-time → phase-index map for the in-phase SSA check on launched
  // loops (mirrors the sequential-level helper at line ~1292).
  DenseMap<uint32_t, size_t> bucketTimeToPhase;
  for (auto phaseIdx : llvm::seq<size_t>(0, phases.size()))
    for (auto t : phases[phaseIdx])
      bucketTimeToPhase[t] = phaseIdx;

  // For each nested loop op, compute phases whose static ops consume a
  // result. Also track if the function terminator (return) consumes it.
  auto computeLopUserPhases = [&](Operation *lop,
                                  DenseSet<size_t> &userPhases,
                                  bool &consumedByTerminator) {
    userPhases.clear();
    consumedByTerminator = false;
    for (auto res : lop->getResults()) {
      for (auto *user : res.getUsers()) {
        if (isFuncTerminator(user)) {
          consumedByTerminator = true;
          continue;
        }
        auto userStart = opOrParentStartTime(problem, user);
        if (userStart < 0)
          continue;
        auto it = bucketTimeToPhase.find((uint32_t)userStart);
        if (it != bucketTimeToPhase.end())
          userPhases.insert(it->second);
      }
    }
  };

  // Pending launches alive across phases. See sequential version's
  // PendingLaunch for semantics.
  struct PendingLaunch {
    Operation *origLoop;
    Value currentHandle;
    SmallVector<std::pair<unsigned, Value>> forwards;
    DenseSet<size_t> remainingUserPhases;
    bool terminatorPending;
  };
  SmallVector<PendingLaunch> pendingLaunches;

  for (auto phaseIdx : llvm::seq<size_t>(0, phases.size())) {
    auto &bucketTimes = phases[phaseIdx];
    uint32_t phaseBase = bucketTimes.front();

    struct BucketContent {
      uint32_t offset;
      SmallVector<Operation *> staticOps;
      SmallVector<Operation *> loopOps;
    };
    SmallVector<BucketContent> buckets;
    for (auto t : bucketTimes) {
      BucketContent bc;
      bc.offset = t - phaseBase;
      for (auto *op : startGroups[t]) {
        if (isa<LoopInterface>(op))
          bc.loopOps.push_back(op);
        else
          bc.staticOps.push_back(op);
      }
      buckets.push_back(std::move(bc));
    }

    struct StaticExport {
      BucketContent *bucket;
      Operation *op;
      unsigned frameFirstIdx;
    };
    SmallVector<StaticExport> staticExports;
    SmallVector<std::pair<uint32_t, Value>> reregExports;
    SmallVector<unsigned> launchHandleIdx;
    SmallVector<Type> stepTypes;

    for (auto &bc : buckets) {
      for (auto *op : bc.staticOps) {
        bool needsReturn = false;
        SmallVector<Operation *, 10> users;
        users.append(op->getUsers().begin(), op->getUsers().end());
        for (auto res : op->getResults()) {
          auto predUsers = predicateUse.lookup(res);
          users.append(predUsers.begin(), predUsers.end());
        }
        for (auto *user : users) {
          auto *userOrAncestor =
              funcOp.getBody().findAncestorOpInRegion(*user);
          if (!userOrAncestor)
            continue;
          // The user (or its ancestor in the func body) is inside this at's
          // region iff it's among the ops being cloned into this at.
          // Anything else escapes the at boundary and must be yielded.
          if (!llvm::is_contained(bc.staticOps, userOrAncestor)) {
            needsReturn = true;
            break;
          }
        }
        if (needsReturn) {
          StaticExport se{&bc, op, (unsigned)stepTypes.size()};
          staticExports.push_back(se);
          stepTypes.append(op->getResultTypes().begin(),
                           op->getResultTypes().end());
        }
      }
    }
    for (auto &bc : buckets) {
      uint32_t bt = phaseBase + bc.offset;
      for (auto val : reregisterValues[bt]) {
        reregExports.emplace_back(bt, val);
        stepTypes.push_back(val.getType());
      }
    }

    SmallVector<std::pair<BucketContent *, Operation *>> launchesInPhase;
    for (auto &bc : buckets)
      for (auto *lop : bc.loopOps)
        launchesInPhase.emplace_back(&bc, lop);

    // Per-launch user-phase info.
    SmallVector<DenseSet<size_t>> launchUserPhasesInPhase;
    SmallVector<bool> launchTerminatorConsumedInPhase;
    for (auto &bl : launchesInPhase) {
      DenseSet<size_t> userPhases;
      bool terminatorConsumed = false;
      computeLopUserPhases(bl.second, userPhases, terminatorConsumed);
      launchUserPhasesInPhase.push_back(userPhases);
      launchTerminatorConsumedInPhase.push_back(terminatorConsumed);
    }

    // Always reserve a frame-result slot for every launch.
    SmallVector<std::optional<unsigned>> launchHandleIdxOpt;
    for (auto &bl : launchesInPhase) {
      (void)bl;
      launchHandleIdx.push_back(stepTypes.size());
      launchHandleIdxOpt.push_back(stepTypes.size());
      stepTypes.push_back(HandleType::get(builder.getContext()));
    }

    // Pending-launch services (same as sequential).
    struct PendingService {
      size_t pendingIdx;
      bool awaitHere;
      bool forwardHere;
      SmallVector<std::pair<unsigned, Value>> forwards;
      std::optional<unsigned> forwardFrameIdx;
    };
    // Always await prior-phase launches in the immediately following phase
    // to enforce launch ordering (a launch's body runs asynchronously; the
    // next frame's await is what prevents the next launch from racing).
    SmallVector<PendingService> pendingServices;
    for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
      PendingService ps;
      ps.pendingIdx = pi;
      ps.awaitHere = true;
      ps.forwardHere = false;
      for (auto &fwd : pl.forwards) {
        bool useHere = false;
        for (auto *user : fwd.second.getUsers()) {
          if (isFuncTerminator(user))
            continue;
          auto userStart = opOrParentStartTime(problem, user);
          if (userStart < 0)
            continue;
          auto it = bucketTimeToPhase.find((uint32_t)userStart);
          if (it != bucketTimeToPhase.end() && it->second == phaseIdx) {
            useHere = true;
            break;
          }
        }
        if (useHere)
          ps.forwards.push_back(fwd);
      }
      pendingServices.push_back(ps);
    }

    auto frame = builder.create<LoopScheduleFrameOp>(stepTypes);

    // Await region + body block args.
    SmallVector<Value> awaitYieldValues;
    SmallVector<Type> bodyBlockArgTypes;
    struct ServicedForwardInfo {
      size_t serviceIdx;
      unsigned firstBodyArgIdx;
    };
    SmallVector<ServicedForwardInfo> servicedForwards;
    {
      Block &awaitBlock = frame.getAwaitRegion().emplaceBlock();
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&awaitBlock);
      for (auto [si, ps] : llvm::enumerate(pendingServices)) {
        if (!ps.awaitHere)
          continue;
        auto &pl = pendingLaunches[ps.pendingIdx];
        SmallVector<Type> forwardTypes;
        for (auto &fwd : ps.forwards)
          forwardTypes.push_back(fwd.second.getType());
        auto awaitOp = builder.create<LoopScheduleAwaitOp>(
            pl.currentHandle.getLoc(), forwardTypes,
            ValueRange{pl.currentHandle});
        if (!ps.forwards.empty()) {
          ServicedForwardInfo sfi;
          sfi.serviceIdx = si;
          sfi.firstBodyArgIdx = bodyBlockArgTypes.size();
          servicedForwards.push_back(sfi);
          for (auto r : awaitOp.getResults()) {
            awaitYieldValues.push_back(r);
            bodyBlockArgTypes.push_back(r.getType());
          }
        }
      }
      builder.create<LoopScheduleYieldOp>(awaitYieldValues);
    }

    Block &bodyBlock = frame.getBodyRegion().emplaceBlock();
    for (auto t : bodyBlockArgTypes)
      bodyBlock.addArgument(t, frame.getLoc());
    for (auto &sfi : servicedForwards) {
      auto &ps = pendingServices[sfi.serviceIdx];
      for (auto [i, fwd] : llvm::enumerate(ps.forwards)) {
        Value bodyArg = bodyBlock.getArgument(sfi.firstBodyArgIdx + i);
        valueMap.map(fwd.second, bodyArg);
      }
    }
    LoopScheduleYieldOp bodyYield;
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&bodyBlock);
      bodyYield = builder.create<LoopScheduleYieldOp>();
    }
    SmallVector<Value> bodyYieldOperands(stepTypes.size(), Value());
    // Forward unserviced pending launches' handles.
    for (auto &ps : pendingServices) {
      if (!ps.forwardHere)
        continue;
      auto &pl = pendingLaunches[ps.pendingIdx];
      bodyYieldOperands[*ps.forwardFrameIdx] = pl.currentHandle;
    }

    for (auto &bc : buckets) {
      if (bc.staticOps.empty())
        continue;

      struct AtSlot {
        enum { Static, Rereg } kind;
        unsigned frameIdx;
        unsigned atIdx;
        StaticExport *se = nullptr;
        Value reregVal;
      };
      SmallVector<AtSlot> slots;
      SmallVector<Type> atTypes;
      for (auto &se : staticExports) {
        if (se.bucket != &bc)
          continue;
        AtSlot s;
        s.kind = AtSlot::Static;
        s.frameIdx = se.frameFirstIdx;
        s.atIdx = atTypes.size();
        s.se = &se;
        for (auto t : se.op->getResultTypes())
          atTypes.push_back(t);
        slots.push_back(s);
      }
      uint32_t bt = phaseBase + bc.offset;
      for (auto &re : reregExports) {
        if (re.first != bt)
          continue;
        unsigned rrFrameIdx = 0;
        unsigned base = 0;
        for (auto &s : staticExports)
          base += s.op->getNumResults();
        unsigned idx = 0;
        for (auto &re2 : reregExports) {
          if (&re2 == &re) {
            rrFrameIdx = base + idx;
            break;
          }
          ++idx;
        }
        AtSlot s;
        s.kind = AtSlot::Rereg;
        s.frameIdx = rrFrameIdx;
        s.atIdx = atTypes.size();
        s.reregVal = re.second;
        atTypes.push_back(re.second.getType());
        slots.push_back(s);
      }

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(bodyYield);
      auto atOp = builder.create<LoopScheduleAtOp>(
          atTypes, builder.getI64IntegerAttr(bc.offset));
      auto &atBlock = atOp.getBodyBlock();
      auto *atTerm = atBlock.getTerminator();
      builder.setInsertionPointToStart(&atBlock);

      auto staticOps = bc.staticOps;
      llvm::sort(staticOps, [&](Operation *a, Operation *b) {
        return dom.dominates(a, b);
      });

      DenseMap<Operation *, Operation *> oldToNew;
      for (auto *op : staticOps) {
        OpBuilder::InsertionGuard g2(builder);
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
        oldToNew[op] = newOp;
        for (auto result : op->getResults())
          valueMap.map(result, newOp->getResult(result.getResultNumber()));
      }

      SmallVector<Value> atYieldOperands(atTypes.size(), Value());
      for (auto &s : slots) {
        if (s.kind == AtSlot::Static) {
          auto *newOp = oldToNew.lookup(s.se->op);
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            atYieldOperands[s.atIdx + i] = newOp->getResult(i);
        } else {
          atYieldOperands[s.atIdx] = valueMap.lookup(s.reregVal);
        }
      }
      atTerm->setOperands(atYieldOperands);

      for (auto &s : slots) {
        if (s.kind == AtSlot::Static) {
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            valueMap.map(s.se->op->getResult(i), atOp.getResult(s.atIdx + i));
        } else {
          valueMap.map(s.reregVal, atOp.getResult(s.atIdx));
        }
      }
      for (auto &s : slots) {
        if (s.kind == AtSlot::Static) {
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            bodyYieldOperands[s.frameIdx + i] = atOp.getResult(s.atIdx + i);
        } else {
          bodyYieldOperands[s.frameIdx] = atOp.getResult(s.atIdx);
        }
      }
    }

    for (auto [launchIdx, bl] : llvm::enumerate(launchesInPhase)) {
      BucketContent *bc = bl.first;
      Operation *lop = bl.second;
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(bodyYield);
      auto launch = builder.create<LoopScheduleLaunchOp>(
          lop->getLoc(),
          HandleType::get(builder.getContext()),
          builder.getI64IntegerAttr(bc->offset));
      Block &launchBlock = launch.getBody().emplaceBlock();
      {
        OpBuilder::InsertionGuard gg(builder);
        builder.setInsertionPointToEnd(&launchBlock);
        builder.create<LoopScheduleYieldOp>();
      }
      auto *launchYield = launchBlock.getTerminator();
      builder.setInsertionPointToStart(&launchBlock);

      auto *newOp = builder.clone(*lop, valueMap);
      dependenceAnalysis->replaceOp(lop, newOp);

      std::queue<Operation *> oldOps;
      lop->walk([&](Operation *op) { oldOps.push(op); });
      if (isa<LoopInterface>(newOp)) {
        newOp->walk([&](Operation *op) {
          Operation *oldOp = oldOps.front();
          dependenceAnalysis->replaceOp(oldOp, op);
          oldOps.pop();
        });
      }

      // Forward the nested loop's results via the launch body yield
      // (dialect docs require this even if the launch's own result is just
      // a handle).
      if (newOp->getNumResults() > 0)
        launchYield->setOperands(newOp->getResults());

      for (auto [orig, clone] :
           llvm::zip(lop->getResults(), newOp->getResults()))
        valueMap.map(orig, clone);

      bodyYieldOperands[*launchHandleIdxOpt[launchIdx]] = launch.getResult();
    }

    bodyYield->setOperands(bodyYieldOperands);

    // Reorder at/launch children by offset to satisfy the frame op's
    // monotonic-offset verifier invariant (see sequential-emitter sort).
    {
      SmallVector<Operation *> children;
      for (Operation &op : bodyBlock.without_terminator())
        children.push_back(&op);
      std::stable_sort(children.begin(), children.end(),
                       [](Operation *a, Operation *b) {
                         auto off = [](Operation *op) -> uint64_t {
                           if (auto at = dyn_cast<LoopScheduleAtOp>(op))
                             return at.getOffset();
                           return cast<LoopScheduleLaunchOp>(op).getOffset();
                         };
                         return off(a) < off(b);
                       });
      for (Operation *op : children)
        op->moveBefore(bodyYield);
    }

    for (auto &se : staticExports) {
      for (unsigned i = 0, e = se.op->getNumResults(); i < e; ++i)
        valueMap.map(se.op->getResult(i),
                     frame->getResult(se.frameFirstIdx + i));
    }
    {
      unsigned base = 0;
      for (auto &s : staticExports)
        base += s.op->getNumResults();
      unsigned idx = 0;
      for (auto &re : reregExports) {
        valueMap.map(re.second, frame->getResult(base + idx));
        ++idx;
      }
    }

    // Update pendingLaunches for next phase.
    SmallVector<PendingLaunch> nextPending;
    for (auto &ps : pendingServices) {
      if (ps.awaitHere)
        continue;
      auto pl = pendingLaunches[ps.pendingIdx];
      pl.currentHandle = frame->getResult(*ps.forwardFrameIdx);
      pl.remainingUserPhases.erase(phaseIdx);
      nextPending.push_back(pl);
    }
    for (auto [launchIdx, bl] : llvm::enumerate(launchesInPhase)) {
      PendingLaunch pl;
      pl.origLoop = bl.second;
      pl.currentHandle = frame->getResult(*launchHandleIdxOpt[launchIdx]);
      for (auto res : bl.second->getResults())
        pl.forwards.push_back({res.getResultNumber(), res});
      for (auto p : launchUserPhasesInPhase[launchIdx])
        if (p > phaseIdx)
          pl.remainingUserPhases.insert(p);
      pl.terminatorPending = launchTerminatorConsumedInPhase[launchIdx];
      nextPending.push_back(pl);
    }
    pendingLaunches = std::move(nextPending);

    for (auto &bc : buckets) {
      uint32_t bt = phaseBase + bc.offset;
      for (auto *op : bc.staticOps) {
        if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
          if (hasLaterUse(op, bt + 1))
            reregisterValues[bt + 1].push_back(load.getResult());
        } else if (auto load = dyn_cast<LoadInterface>(op)) {
          auto latency = load.getLatency();
          if (hasLaterUse(op, bt + latency))
            reregisterValues[bt + latency].push_back(load.getResult());
        }
      }
    }
  }

  // If there are pending launches (their handles live as of the last phase's
  // frame results), emit a trailing "barrier" frame that awaits them. This
  // keeps the launch verifier satisfied (every launch must reach an await
  // terminal) and gives the top-level function a clean completion point.
  if (!pendingLaunches.empty()) {
    // Collect per-launch forwards whose original value is consumed by the
    // func terminator (i.e. returned). These need to flow out of the
    // barrier frame as frame SSA results so func.return can reach them.
    struct ReturnForward {
      size_t pendingIdx;    // index into pendingLaunches
      unsigned forwardIdx;  // index into pl.forwards
      Type resultType;
    };
    SmallVector<ReturnForward> returnForwards;
    SmallVector<Type> barrierResultTypes;
    for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
      for (auto [fi, fwd] : llvm::enumerate(pl.forwards)) {
        for (auto *user : fwd.second.getUsers()) {
          if (isFuncTerminator(user)) {
            returnForwards.push_back({pi, (unsigned)fi, fwd.second.getType()});
            barrierResultTypes.push_back(fwd.second.getType());
            break;
          }
        }
      }
    }

    builder.setInsertionPoint(funcReturn);
    auto barrier = builder.create<LoopScheduleFrameOp>(barrierResultTypes);

    // Await region: await each pending launch's handle; for launches with
    // return-forwards, await with the forward's result type so the value
    // crosses into the body-block-arg.
    {
      Block &awaitBlock = barrier.getAwaitRegion().emplaceBlock();
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&awaitBlock);

      SmallVector<Value> awaitYieldOperands;
      // Per-pending-launch: await once; if it has return-forwards, include
      // their types in the await's result types.
      for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
        SmallVector<Type> awaitRetTypes;
        SmallVector<const ReturnForward *> forwardsForThis;
        for (auto &rf : returnForwards) {
          if (rf.pendingIdx == pi) {
            awaitRetTypes.push_back(rf.resultType);
            forwardsForThis.push_back(&rf);
          }
        }
        auto awaitOp = builder.create<LoopScheduleAwaitOp>(
            pl.currentHandle.getLoc(), awaitRetTypes,
            ValueRange{pl.currentHandle});
        for (auto [i, rf] : llvm::enumerate(forwardsForThis))
          awaitYieldOperands.push_back(awaitOp.getResult(i));
      }
      builder.create<LoopScheduleYieldOp>(funcOp.getLoc(), awaitYieldOperands);
    }

    // Body region: block args come from the await-region yield. The body
    // forwards them as frame results so func.return can pick them up.
    {
      Block &bodyBlock = barrier.getBodyRegion().emplaceBlock();
      for (Type t : barrierResultTypes)
        bodyBlock.addArgument(t, funcOp.getLoc());
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&bodyBlock);
      builder.create<LoopScheduleYieldOp>(
          funcOp.getLoc(),
          SmallVector<Value>(bodyBlock.getArguments().begin(),
                             bodyBlock.getArguments().end()));
    }

    // Rewrite func.return uses: each return-forwarded original value becomes
    // the corresponding frame SSA result.
    for (auto [i, rf] : llvm::enumerate(returnForwards)) {
      Value origValue = pendingLaunches[rf.pendingIdx].forwards[rf.forwardIdx].second;
      origValue.replaceAllUsesWith(barrier.getResult(i));
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

  std::function<bool(Operation *)> inTopLevelFrameOp = [&](Operation *op) {
    auto parent = op->getParentOfType<LoopScheduleFrameOp>();
    if (!parent)
      return false;

    if (isa<func::FuncOp>(parent->getParentOp()))
      return true;

    return inTopLevelFrameOp(parent);
  };

  // Remove the loop nest from the IR.
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](Operation *op) {
    if ((isa<LoopScheduleFrameOp>(op) && isa<FuncOp>(op->getParentOp())) ||
        inTopLevelFrameOp(op) ||
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
