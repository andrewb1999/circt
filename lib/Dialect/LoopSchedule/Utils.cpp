//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace circt;
using namespace circt::loopschedule;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace circt {

namespace loopschedule {

Value getMemref(Operation *op) {
  Value memref =
      isa<AffineStoreOp>(*op)     ? cast<AffineStoreOp>(*op).getMemRef()
      : isa<AffineLoadOp>(*op)    ? cast<AffineLoadOp>(*op).getMemRef()
      : isa<memref::StoreOp>(*op) ? cast<memref::StoreOp>(*op).getMemRef()
      : isa<memref::LoadOp>(*op)  ? cast<memref::LoadOp>(*op).getMemRef()
      : isa<LoopScheduleLoadOp>(*op)
          ? cast<LoopScheduleLoadOp>(*op).getMemRef()
          : cast<LoopScheduleStoreOp>(*op).getMemRef();
  return memref;
}

bool oneIsStore(Operation *op, Operation *otherOp) {
  auto firstIsStore = isa<AffineStoreOp, memref::StoreOp, StoreInterface>(*op);
  auto secondIsStore =
      isa<AffineStoreOp, memref::StoreOp, StoreInterface>(*otherOp);
  return firstIsStore || secondIsStore;
}

bool hasLoopScheduleDependence(Operation *op, Operation *otherOp) {
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

ModuloProblem
getModuloProblem(scf::ForOp forOp,
                 LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  // Create a modulo scheduling problem.
  ModuloProblem problem = ModuloProblem::get(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (LoopScheduleDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      unsigned distance = memoryDep.distance;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  problem.insertOperation(anchor);
  forOp.getBody()->walk([&](Operation *op) {
    if (op == anchor || !problem.hasOperation(op))
      return;
    Dependence dep(op, anchor);
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
        Dependence dep(iterArgDefiner, iterArgUser);
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
getSharedOperatorsProblem(scf::ForOp forOp,
                          LoopScheduleDependenceAnalysis &dependenceAnalysis) {
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
            Dependence dep(definingOp, op);
            auto depInserted = problem.insertDependence(dep);
            assert(succeeded(depInserted));
            (void)depInserted;
          }
        }
      });
    }

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (const LoopScheduleDependence &memoryDep : dependences) {
      assert(memoryDep.source != nullptr);
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Do not consider inter-iteration deps for seq loops
      auto distance = memoryDep.distance;
      if (distance > 0)
        continue;

      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  assert(forOp.getLoopRegions().size() == 1);
  auto *anchor = forOp.getLoopRegions().front()->back().getTerminator();
  problem.insertOperation(anchor);
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
        !problem.hasOperation(op))
      return;
    if (!isa<AffineStoreOp, memref::StoreOp, StoreInterface>(op))
      return;
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

SharedOperatorsProblem
getSharedOperatorsProblem(func::FuncOp funcOp,
                          LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(funcOp);

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (const LoopScheduleDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!funcOp->isAncestor(memoryDep.source))
        continue;
      if (memoryDep.distance > 0)
        continue;
      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = funcOp.getBody().back().getTerminator();
  problem.insertOperation(anchor);
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
        !problem.hasOperation(op))
      return;
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

LogicalResult unrollSubLoops(scf::ForOp &forOp) {
  auto result = forOp.getBody()->walk<WalkOrder::PostOrder>([](scf::ForOp op) {
    std::optional<int64_t> lbCstOp = getConstantIntValue(op.getLowerBound());
    std::optional<int64_t> ubCstOp = getConstantIntValue(op.getUpperBound());
    std::optional<int64_t> stepCstOp = getConstantIntValue(op.getStep());
    if (!lbCstOp || !ubCstOp || !stepCstOp) {
      return WalkResult::interrupt();
    }
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);
    if (loopUnrollByFactor(op, tripCount).failed())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    forOp.emitOpError("Could not unroll sub loops");
    return failure();
  }

  return success();
}

} // namespace loopschedule
} // namespace circt
