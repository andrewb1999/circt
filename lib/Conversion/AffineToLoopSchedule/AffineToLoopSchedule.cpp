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
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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
  void runOnOperation() override;

private:
  ModuloProblem getModuloProblem(affine::AffineForOp forOp);
  SharedOperatorsProblem getSharedOperatorsProblem(affine::AffineForOp forOp);
  SharedOperatorsProblem getSharedOperatorsProblem(func::FuncOp funcOp);
  LogicalResult lowerAffineStructures();
  LogicalResult unrollSubLoops(AffineForOp &forOp);
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

ModuloProblem
AffineToLoopSchedule::getModuloProblem(affine::AffineForOp forOp) {
  // Create a modulo scheduling problem.
  ModuloProblem problem = ModuloProblem::get(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences =
        dependenceAnalysis->getDependences(op);
    if (dependences.empty())
      return;

    for (MemoryDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      unsigned distance = *memoryDep.dependenceComponents.back().lb;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](Operation *op) {
    if (op == anchor || !problem.hasOperation(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Handle explicitly computed loop-carried values, i.e. excluding the
  // induction variable. Insert inter-iteration dependences from the definers of
  // "iter_args" to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = forOp.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      Operation *iterArgDefiner = anchor->getOperand(i).getDefiningOp();
      // If it's not an operation, we don't need to model the dependence.
      if (!iterArgDefiner)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Problem::Dependence dep(iterArgDefiner, iterArgUser);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;

        // Values always flow between subsequent iterations.
        problem.setDistance(dep, 1);
      }
    }
  }

  return problem;
}

SharedOperatorsProblem
AffineToLoopSchedule::getSharedOperatorsProblem(affine::AffineForOp forOp) {
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(forOp);

  // Insert memory dependences into the problem.
  assert(forOp.getLoopRegions().size() == 1);
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopInterface>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    if (auto loop = dyn_cast<LoopInterface>(op)) {
      loop.getBodyBlock()->walk([&](Operation *innerOp) {
        for (auto &operand : innerOp->getOpOperands()) {
          auto *definingOp = operand.get().getDefiningOp();
          if (definingOp && definingOp->getParentOp() == forOp) {
            Problem::Dependence dep(definingOp, op);
            auto depInserted = problem.insertDependence(dep);
            assert(succeeded(depInserted));
            (void)depInserted;
          }
        }
      });
    }

    ArrayRef<MemoryDependence> dependences =
        dependenceAnalysis->getDependences(op);
    if (dependences.empty())
      return;

    for (const MemoryDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      assert(memoryDep.source != nullptr);
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Do not consider inter-iteration deps for seq loops
      auto distance = memoryDep.dependenceComponents.back().lb;
      if (distance.has_value())
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // DenseMap<Operation *, SmallVector<LoopInterface>> memOps;
  // for (auto loop : forOp.getOps<LoopInterface>()) {
  //   loop.getBodyBlock()->walk([&](Operation *op) {
  //     if (isa<AffineLoadOp, AffineStoreOp, memref::LoadOp, memref::StoreOp,
  //             LoadInterface, StoreInterface>(op)) {
  //       memOps[op].push_back(loop);
  //     }
  //   });
  // }

  // for (auto loop : forOp.getOps<LoopInterface>()) {
  //   for (auto it : memOps) {
  //     auto *memOp = it.getFirst();
  //     auto dependences = dependenceAnalysis->getDependences(memOp);
  //     for (const MemoryDependence &memoryDep : dependences) {
  //       if (!hasDependence(memoryDep.dependenceType))
  //         continue;
  //       for (auto otherLoop : memOps.lookup(memoryDep.source)) {
  //         if (loop == otherLoop || !loop->isBeforeInBlock(otherLoop))
  //           continue;
  //         for (auto comp : memoryDep.dependenceComponents) {
  //           if (!comp.lb.has_value() || comp.lb == 0)
  //             continue;
  //           llvm::errs() << "op: ";
  //           memOp->dump();
  //           llvm::errs() << "otherOp: ";
  //           memoryDep.source->dump();
  //           Problem::Dependence dep(loop, otherLoop);
  //           auto depInserted = problem.insertDependence(dep);
  //           assert(succeeded(depInserted));
  //         }
  //       }
  //     }
  //   }
  // }

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  assert(forOp.getLoopRegions().size() == 1);
  auto *anchor = forOp.getLoopRegions().front()->back().getTerminator();
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;
    if (!isa<AffineStoreOp, memref::StoreOp, StoreInterface>(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

SharedOperatorsProblem
AffineToLoopSchedule::getSharedOperatorsProblem(func::FuncOp funcOp) {
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(funcOp);

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
  //     // llvm::errs() << "deps: ";
  //     // memoryDep.source->dump();
  //   }
  // }

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences =
        dependenceAnalysis->getDependences(op);
    if (dependences.empty())
      return;

    // op->dump();

    for (const MemoryDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;
      if (!funcOp->isAncestor(memoryDep.source))
        continue;
      if (memoryDep.dependenceComponents.back().lb.has_value())
        continue;
      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // DenseMap<Operation *, SmallVector<LoopInterface>> memOps;
  // for (auto loop : funcOp.getOps<LoopInterface>()) {
  //   loop.getBodyBlock()->walk([&](Operation *op) {
  //     if (isa<AffineLoadOp, AffineStoreOp, memref::LoadOp, memref::StoreOp,
  //             LoadInterface, StoreInterface>(op)) {
  //       memOps[op].push_back(loop);
  //     }
  //   });
  // }

  // for (auto loop : funcOp.getOps<LoopInterface>()) {
  //   for (auto it : memOps) {
  //     auto *memOp = it.getFirst();
  //     auto dependences = dependenceAnalysis->getDependences(memOp);
  //     // memOp->dump();
  //     for (const MemoryDependence &memoryDep : dependences) {
  //       // llvm::errs() << "dep: \n";
  //       // memoryDep.source->dump();
  //       if (!hasDependence(memoryDep.dependenceType))
  //         continue;
  //       for (auto otherLoop : memOps.lookup(memoryDep.source)) {
  //         if (loop == otherLoop || !loop->isBeforeInBlock(otherLoop))
  //           continue;
  //         // llvm::errs() << "loop dep\n";
  //         // loop.dump();
  //         // otherLoop.dump();
  //         Problem::Dependence dep(loop, otherLoop);
  //         auto depInserted = problem.insertDependence(dep);
  //         assert(succeeded(depInserted));
  //       }
  //     }
  //   }
  // }

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = funcOp.getBody().back().getTerminator();
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
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

Value getMemref(Operation *op) {
  Value memref = isa<AffineStoreOp>(*op)  ? cast<AffineStoreOp>(*op).getMemRef()
                 : isa<AffineLoadOp>(*op) ? cast<AffineLoadOp>(*op).getMemRef()
                 : isa<memref::StoreOp>(*op)
                     ? cast<memref::StoreOp>(*op).getMemRef()
                     : cast<memref::LoadOp>(*op).getMemRef();
  return memref;
}

bool oneIsStore(Operation *op, Operation *otherOp) {
  auto firstIsStore = isa<AffineStoreOp, memref::StoreOp, StoreInterface>(*op);
  auto secondIsStore =
      isa<AffineStoreOp, memref::StoreOp, StoreInterface>(*otherOp);
  return firstIsStore || secondIsStore;
}

bool hasMemoryDependence(Operation *op, Operation *otherOp) {
  if (isa<LoadInterface, StoreInterface>(op)) {
    if (!isa<LoadInterface, StoreInterface>(otherOp)) {
      return false;
    }

    if (auto load = dyn_cast<LoadInterface>(op)) {
      return load.hasDependence(otherOp);
    }

    auto store = dyn_cast<StoreInterface>(op);
    return store.hasDependence(otherOp);
  }

  if (isa<LoadInterface, StoreInterface>(otherOp)) {
    return false;
  }

  auto memref = getMemref(op);
  auto otherMemref = getMemref(otherOp);
  return memref == otherMemref && oneIsStore(op, otherOp);
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
  if (failed(lowerAffineStructures()))
    return signalPassFailure();

  // Schedule all pipelined loops first
  for (auto loop : llvm::make_early_inc_range(loops)) {

    // Populate the target operator types.
    ModuloProblem moduloProblem = getModuloProblem(loop);

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
    auto problem = getSharedOperatorsProblem(loop);

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
  auto problem = getSharedOperatorsProblem(funcOp);

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

class SchedulableAffineReadInterfaceLowering : public mlir::RewritePattern {
public:
  SchedulableAffineReadInterfaceLowering(
      MLIRContext *context, MemoryDependenceAnalysis &dependenceAnalysis)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(*op)) {
      // Expand affine map from 'affineWriteOpInterface'.
      SmallVector<Value, 8> indices(readOp.getMapOperands());
      auto maybeExpandedMap = expandAffineMap(rewriter, readOp.getLoc(),
                                              readOp.getAffineMap(), indices);
      if (!maybeExpandedMap.has_value())
        return failure();

      if (auto schedulableOp =
              dyn_cast<loopschedule::SchedulableAffineInterface>(*op)) {
        // Build memref.store valueToStore, memref[expandedMap.results].
        auto *newOp =
            schedulableOp.createNonAffineOp(rewriter, *maybeExpandedMap);
        rewriter.replaceOp(op, newOp->getResults());

        dependenceAnalysis.replaceOp(op, newOp);

        return success();
      }
    }
    return failure();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

class SchedulableAffineWriteInterfaceLowering : public mlir::RewritePattern {
public:
  SchedulableAffineWriteInterfaceLowering(
      MLIRContext *context, MemoryDependenceAnalysis &dependenceAnalysis)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto writeOp = dyn_cast<AffineWriteOpInterface>(*op)) {
      // Expand affine map from 'affineWriteOpInterface'.
      SmallVector<Value, 8> indices(writeOp.getMapOperands());
      auto maybeExpandedMap = expandAffineMap(rewriter, writeOp.getLoc(),
                                              writeOp.getAffineMap(), indices);
      if (!maybeExpandedMap.has_value())
        return failure();

      if (auto schedulableOp =
              dyn_cast<loopschedule::SchedulableAffineInterface>(*op)) {
        // Build memref.store valueToStore, memref[expandedMap.results].
        auto *newOp =
            schedulableOp.createNonAffineOp(rewriter, *maybeExpandedMap);
        rewriter.replaceOp(op, newOp->getResults());

        dependenceAnalysis.replaceOp(op, newOp);

        return success();
      }
    }
    return failure();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

struct MulStrengthReduction : OpConversionPattern<MulIOp> {
  using OpConversionPattern<MulIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *lhsDef = op.getLhs().getDefiningOp();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto log = val.getValue().exactLogBase2();
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), log);
        auto shift = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(),
                                                   shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

struct RemUIStrengthReduction : OpConversionPattern<RemUIOp> {
  using OpConversionPattern<RemUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *lhsDef = op.getLhs().getDefiningOp();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto shifted = val.getValue() - 1;
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), shifted);
        auto shift = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(),
                                                   shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

struct RemSIStrengthReduction : OpConversionPattern<RemSIOp> {
  using OpConversionPattern<RemSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, op.getLhs(), op.getRhs());

    return success();
  }
};

struct DivSIStrengthReduction : OpConversionPattern<DivSIOp> {
  using OpConversionPattern<DivSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *lhsDef = op.getLhs().getDefiningOp();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto log = val.getValue().exactLogBase2();
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), log);
        auto shift = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, op.getLhs(),
                                                    shift.getResult());
        return success();
      }
    }

    return failure();
  }
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

static bool schedulableAffineInterfaceLegalityCallback(Operation *op) {
  return !(
      isa<loopschedule::SchedulableAffineInterface>(*op) &&
      (isa<AffineReadOpInterface>(*op) || isa<AffineWriteOpInterface>(*op)));
}

static bool mulLegalityCallback(Operation *op) {
  if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    auto *rhsDef = mulOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool DivSIOpLegalityCallback(Operation *op) {
  if (auto mulOp = dyn_cast<arith::DivSIOp>(op)) {
    auto *rhsDef = mulOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool remUILegalityCallback(Operation *op) {
  if (auto remOp = dyn_cast<arith::RemUIOp>(op)) {
    auto *rhsDef = remOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool remSILegalityCallback(Operation *op) {
  if (auto remOp = dyn_cast<arith::RemSIOp>(op)) {
    auto *rhsDef = remOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto rhsValue = cast<IntegerAttr>(constOp.getValue());
      if (rhsValue.getValue().exactLogBase2() != -1) {
        if (rhsValue.getInt() >= 0)
          return false;
      }
    }
  }
  return true;
}

/// After analyzing memory dependences, and before creating the schedule, we
/// want to materialize affine operations with arithmetic, scf, and memref
/// operations, which make the condition computation of addresses, etc.
/// explicit. This is important so the schedule can consider potentially complex
/// computations in the condition of ifs, or the addresses of loads and stores.
/// The dependence analysis will be updated so the dependences from the affine
/// loads and stores are now on the memref loads and stores.
LogicalResult AffineToLoopSchedule::lowerAffineStructures() {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, ArithDialect, MemRefDialect,
                         SCFDialect>();
  target.addIllegalOp<AffineIfOp, AffineLoadOp, AffineStoreOp>();
  target.markUnknownOpDynamicallyLegal(
      schedulableAffineInterfaceLegalityCallback);
  target.addDynamicallyLegalOp<IfOp>(ifOpLegalityCallback);
  target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  RewritePatternSet patterns(context);
  patterns.add<AffineLoadLowering>(context, *dependenceAnalysis);
  patterns.add<AffineStoreLowering>(context, *dependenceAnalysis);
  patterns.add<IfOpHoisting>(context);
  patterns.add<SchedulableAffineReadInterfaceLowering>(context,
                                                       *dependenceAnalysis);
  patterns.add<SchedulableAffineWriteInterfaceLowering>(context,
                                                        *dependenceAnalysis);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  patterns.clear();
  populateAffineToStdConversionPatterns(patterns);
  target.addIllegalOp<AffineApplyOp>();
  patterns.add<MulStrengthReduction>(context);
  patterns.add<DivSIStrengthReduction>(context);
  patterns.add<RemUIStrengthReduction>(context);
  patterns.add<RemSIStrengthReduction>(context);

  target.addDynamicallyLegalOp<MulIOp>(mulLegalityCallback);
  target.addDynamicallyLegalOp<DivSIOp>(DivSIOpLegalityCallback);
  target.addDynamicallyLegalOp<RemUIOp>(remUILegalityCallback);
  target.addDynamicallyLegalOp<RemSIOp>(remSILegalityCallback);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  // Loop invariant code motion to hoist produced constants out of loop
  op->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

  mlir::PassManager pm(context);
  pm.addPass(arith::createIntRangeOptimizationsPass());
  if (failed(pm.run(op)))
    signalPassFailure();

  return success();
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
              YieldOp, func::ReturnOp>([&](Operation *combOp) {
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
            if (isa<AffineLoadOp, AffineStoreOp, memref::LoadOp,
                    memref::StoreOp>(op)) {
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
        .Case<memref::StoreOp, AffineStoreOp>([&](Operation *memOp) {
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
        .Case<memref::LoadOp, AffineLoadOp>([&](Operation *memOp) {
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

/// Create the pipeline op for a loop nest.
LogicalResult
AffineToLoopSchedule::createLoopSchedulePipeline(AffineForOp &loop,
                                                 ModuloProblem &problem) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  // loop.dump();

  builder.setInsertionPointToStart(
      &loop->getParentOfType<FuncOp>().getBody().front());

  // Create Values for the loop's lower and upper bounds.
  Value lowerBound = lowerAffineLowerBound(loop, builder);
  Value upperBound = lowerAffineUpperBound(loop, builder);
  int64_t stepValue = loop.getStep().getSExtValue();
  auto step = builder.create<arith::ConstantOp>(
      IntegerAttr::get(builder.getIndexType(), stepValue));

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

      // Make sure a stage exists for every time between startTime and
      // pipeEndTime
      // for (unsigned i = startTime; i < pipeEndTime; ++i)
      //   newStartTimes.insert(i);

      // Insert the range of pipeline stages the value needs to be valid for
      for (auto res : op->getResults())
        pipeTimes[res] = std::pair(startTime, pipeEndTime);

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
  Value lowerBound = lowerAffineLowerBound(loop, builder);
  Value upperBound = lowerAffineUpperBound(loop, builder);
  int64_t stepValue = loop.getStep().getSExtValue();
  auto incr = builder.create<arith::ConstantOp>(
      IntegerAttr::get(builder.getIndexType(), stepValue));

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
      auto resTime = *startTime + *load.getLatency();
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
        if (hasLaterUse(op, startTime + *load.getLatency())) {
          auto resTime = startTime + *load.getLatency();
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
      auto *newOp = builder.clone(*op, valueMap);
      dependenceAnalysis->replaceOp(op, newOp);
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

  // Update return with correct values
  auto *returnOp = funcOp.getBody().back().getTerminator();
  int numOperands = returnOp->getNumOperands();
  for (int i = 0; i < numOperands; ++i) {
    auto operand = returnOp->getOperand(i);
    auto newValue = valueMap.lookup(operand);
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

std::unique_ptr<mlir::Pass> circt::createAffineToLoopSchedule() {
  return std::make_unique<AffineToLoopSchedule>();
}
