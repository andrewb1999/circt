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
#include "mlir/Dialect/Math/IR/Math.h"
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
#include "mlir/Support/WalkResult.h"
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

SmallVector<LoopScheduleAtOp> getAtOpsInOrder(Region &region) {
  SmallVector<LoopScheduleAtOp> result;
  if (region.empty())
    return result;
  for (auto at : region.front().getOps<LoopScheduleAtOp>())
    result.push_back(at);
  return result;
}

SmallVector<LoopScheduleLaunchOp> getLaunchOpsInOrder(LoopScheduleFrameOp frame) {
  SmallVector<LoopScheduleLaunchOp> result;
  // In the post-normalization shape consumed by dissolveLaunchesAndAwaits,
  // launches are peers of at-ops directly in the frame body.
  for (auto launch : frame.getBodyBlock().getOps<LoopScheduleLaunchOp>())
    result.push_back(launch);
  // Also walk ats (the new-shape case, in case normalization hasn't
  // run yet): launches may live inside their parent at.
  for (auto at : frame.getBodyBlock().getOps<LoopScheduleAtOp>())
    for (auto launch : at.getBodyBlock().getOps<LoopScheduleLaunchOp>())
      result.push_back(launch);
  return result;
}

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
getModuloProblem(scf::WhileOp whileOp,
                 LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  // Create a modulo scheduling problem.
  ModuloProblem problem(whileOp);

  // Insert memory dependences into the problem.
  whileOp.getAfterBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (LoopScheduleDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!whileOp->isAncestor(memoryDep.source))
        continue;

      unsigned distance = memoryDep.distance;
      // if (distance > 0)
      //   continue;
      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      if (isa<loopschedule::LoopScheduleStoreOp, StoreInterface,
              memref::StoreOp>(memoryDep.source)) {
        problem.setSrcAsStore(dep, true);
      }
      auto depInserted = problem.insertDependence(dep);

      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      // unsigned distance = memoryDep.distance;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = whileOp.getAfterBody()->getTerminator();
  problem.insertOperation(anchor);
  whileOp.getAfterBody()->walk([&](Operation *op) {
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
    auto iterArgs = whileOp.getRegionIterArgs();
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

ChainingModuloProblem
getChainingModuloProblem(scf::WhileOp whileOp,
                         LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  // Create a modulo scheduling problem.
  ChainingModuloProblem problem(whileOp);

  // Insert memory dependences into the problem.
  whileOp.getAfterBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (LoopScheduleDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!whileOp->isAncestor(memoryDep.source))
        continue;

      unsigned distance = memoryDep.distance;
      // if (distance > 0)
      //   continue;
      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      if (isa<loopschedule::LoopScheduleStoreOp, StoreInterface,
              memref::StoreOp>(memoryDep.source)) {
        problem.setSrcAsStore(dep, true);
      }
      auto depInserted = problem.insertDependence(dep);

      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      // unsigned distance = memoryDep.distance;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = whileOp.getAfterBody()->getTerminator();
  problem.insertOperation(anchor);
  whileOp.getAfterBody()->walk([&](Operation *op) {
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
    auto iterArgs = whileOp.getAfterArguments();
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
getSharedOperatorsProblem(scf::WhileOp whileOp,
                          LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  SharedOperatorsProblem problem(whileOp);

  // Insert memory dependences into the problem.
  assert(whileOp.getLoopRegions().size() == 1);
  whileOp.getAfter().walk([&](Operation *op) {
    if (op->getParentOfType<LoopInterface>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    if (auto loop = dyn_cast<LoopInterface>(op)) {
      loop.getBodyBlock()->walk([&](Operation *innerOp) {
        for (auto &operand : innerOp->getOpOperands()) {
          auto *definingOp = operand.get().getDefiningOp();
          if (definingOp && definingOp->getParentOp() == whileOp) {
            Problem::Dependence dep(definingOp, op);
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
      if (!whileOp->isAncestor(memoryDep.source))
        continue;

      // Do not consider inter-iteration deps for seq loops
      auto distance = memoryDep.distance;
      if (distance > 0)
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  assert(whileOp.getLoopRegions().size() == 1);
  auto *anchor = whileOp.getAfter().back().getTerminator();
  problem.insertOperation(anchor);
  whileOp.getAfter().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
        !problem.hasOperation(op))
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

ChainingSharedOperatorsProblem getChainingSharedOperatorsProblem(
    scf::WhileOp whileOp, LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  ChainingSharedOperatorsProblem problem(whileOp);

  // Insert memory dependences into the problem.
  whileOp.getAfterBody()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopInterface>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    if (auto loop = dyn_cast<LoopInterface>(op)) {
      loop.getBodyBlock()->walk([&](Operation *innerOp) {
        for (auto &operand : innerOp->getOpOperands()) {
          auto *definingOp = operand.get().getDefiningOp();
          if (definingOp && definingOp->getParentOp() == whileOp) {
            Problem::Dependence dep(definingOp, op);
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
      if (!whileOp->isAncestor(memoryDep.source))
        continue;

      // Do not consider inter-iteration deps for seq loops
      auto distance = memoryDep.distance;
      if (distance > 0)
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = whileOp.getAfter().back().getTerminator();
  problem.insertOperation(anchor);
  whileOp.getAfter().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
        !problem.hasOperation(op))
      return;

    for (OpOperand &operand : anchor->getOpOperands()) {
      Value v = operand.get();
      Operation *argProducer = v.getDefiningOp();
      if (op != argProducer) {
        Value arg = whileOp.getAfterArguments()[operand.getOperandNumber()];
        if (auto loop = dyn_cast<LoopInterface>(op)) {
          loop.getBodyBlock()->walk([&](Operation *innerOp) {
            for (OpOperand &otherOperand : innerOp->getOpOperands()) {
              if (otherOperand.get() == arg) {
                Problem::Dependence dep(op, argProducer);
                auto depInserted = problem.insertDependence(dep);
                assert(succeeded(depInserted));
                (void)depInserted;
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
        } else {
          for (OpOperand &otherOperand : op->getOpOperands()) {
            if (otherOperand.get() == arg) {
              Problem::Dependence dep(op, argProducer);
              auto depInserted = problem.insertDependence(dep);
              assert(succeeded(depInserted));
              (void)depInserted;
              break;
            }
          }
        }
      }
    }

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
getSharedOperatorsProblem(func::FuncOp funcOp,
                          LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  SharedOperatorsProblem problem(funcOp);

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    // Add dependencies for ops contained within loops
    if (isa<LoopSchedulePipelineOp>(op) || isa<LoopScheduleSequentialOp>(op)) {
      op->walk([&](Operation *innerOp) {
        for (auto operand : innerOp->getOperands()) {
          if (isa<BlockArgument>(operand)) {
            continue;
          }

          if (problem.hasOperation(operand.getDefiningOp())) {
            Problem::Dependence dep(operand.getDefiningOp(), op);
            auto depInserted = problem.insertDependence(dep);
            assert(succeeded(depInserted));
          }
        }
      });
    }

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
      Problem::Dependence dep(memoryDep.source, op);
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
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

ChainingModuloProblem
getChainingModuloProblem(
    func::FuncOp funcOp,
    LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  ChainingModuloProblem problem(funcOp);

  // Insert ops + dependences for the func body. Skip ops that already live
  // inside an inner LoopSchedule container (sequential / pipeline / frame)
  // — those have been scheduled as launches already.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependencies =
        dependenceAnalysis.getDependencies(op);
    if (dependencies.empty())
      return;

    for (const LoopScheduleDependence &memoryDep : dependencies) {
      if (!funcOp->isAncestor(memoryDep.source))
        continue;
      Problem::Dependence dep(memoryDep.source, op);
      if (isa<loopschedule::LoopScheduleStoreOp, StoreInterface,
              memref::StoreOp>(memoryDep.source))
        problem.setSrcAsStore(dep, true);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
      // Cross-transaction memory dependences become distance>0 backedges
      // in the modulo problem (ChainingModulo's whole point — bound II by
      // both resource overlap and any real cyclic edges).
      if (memoryDep.distance > 0)
        problem.setDistance(dep, memoryDep.distance);
    }
  });

  // Anchor: terminator must be scheduled after every other op.
  auto *anchor = funcOp.getBody().back().getTerminator();
  problem.insertOperation(anchor);
  funcOp.getBody().walk([&](Operation *op) {
    if (op == anchor || !problem.hasOperation(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

ChainingSharedOperatorsProblem getChainingSharedOperatorsProblem(
    func::FuncOp funcOp, LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  ChainingSharedOperatorsProblem problem(funcOp);

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    // Add dependencies for ops contained within loops
    if (isa<LoopSchedulePipelineOp>(op) || isa<LoopScheduleSequentialOp>(op)) {
      op->walk([&](Operation *innerOp) {
        for (auto operand : innerOp->getOperands()) {
          if (isa<BlockArgument>(operand)) {
            continue;
          }

          if (problem.hasOperation(operand.getDefiningOp())) {
            Problem::Dependence dep(operand.getDefiningOp(), op);
            auto depInserted = problem.insertDependence(dep);
            assert(succeeded(depInserted));
          }
        }
      });
    }

    ArrayRef<LoopScheduleDependence> dependencies =
        dependenceAnalysis.getDependencies(op);
    if (dependencies.empty())
      return;

    for (const LoopScheduleDependence &memoryDep : dependencies) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!funcOp->isAncestor(memoryDep.source))
        continue;
      if (memoryDep.distance > 0)
        continue;
      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
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
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

namespace {
struct IfOpTypes {
  IfOpTypes(scf::IfOp ifOp, bool inThen) : ifOp(ifOp), inThen(inThen) {}

  scf::IfOp ifOp;
  bool inThen;
  llvm::StringMap<SmallVector<std::string>> thenTypes;
  llvm::StringMap<SmallVector<std::string>> elseTypes;
};
} // namespace

static std::map<Operation *, std::string> uniqueName;
static int ifCounter = 0;

static std::string getUnqiueName(Operation *op) {
  if (uniqueName.count(op) > 0)
    return uniqueName[op];
  auto name = "if" + std::to_string(ifCounter);
  uniqueName.insert(std::pair(op, name));
  ifCounter++;
  return name;
}

LogicalResult recordMemoryResources(Operation *op, Region &body,
                                    ResourceMap &resourceMap,
                                    ResourceLimits &resourceLimits) {
  std::vector<std::unique_ptr<IfOpTypes>> ifOps;
  llvm::StringMap<SmallVector<std::string>> finalTypes;

  // Insert ResourceTypes
  // This method is needed to ensure that resource uses in ifOp then and else
  // blocks can be run in parallel.
  body.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op->getParentOfType<LoopInterface>() != nullptr)
      return;

    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      ifOps.push_back(std::make_unique<IfOpTypes>(ifOp, true));
    }

    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      if (!ifOps.empty()) {
        auto &ifOpTypes = ifOps.back();
        if (ifOpTypes->inThen) {
          ifOpTypes->inThen = false;
        } else {
          std::unique_ptr<IfOpTypes> ifOpTypes = std::move(ifOps.back());
          ifOps.pop_back();
          assert(ifOpTypes.get() != nullptr);
          for (auto &it : ifOpTypes->thenTypes) {
            for (auto &rsrc : it.second)
              finalTypes[it.first()].push_back(std::move(rsrc));
          }
          for (auto &it : ifOpTypes->elseTypes) {
            for (auto &rsrc : it.second)
              finalTypes[it.first()].push_back(std::move(rsrc));
          }
        }
      }
    } else if (isa<LoopScheduleLoadOp, LoopScheduleStoreOp, LoadInterface,
                   StoreInterface>(op)) {
      std::string name;
      // Port groups (a `!amc.port<... x N>`) declare a per-cycle capacity of N
      // via the access op's `getLimit()`; honor it so N accesses may share the
      // resource in one cycle. `LoopScheduleLoad/Store` are single-ported.
      unsigned limit = 1;
      if (isa<LoopScheduleLoadOp, LoopScheduleStoreOp>(op)) {
        Value memRef = getMemref(op);
        name = "mem_" + std::to_string(hash_value(memRef));
      } else if (auto loadOp = dyn_cast<loopschedule::LoadInterface>(*op)) {
        name = loadOp.getUniqueId();
        limit = loadOp.getLimit().value_or(1);
      } else {
        auto storeOp = cast<loopschedule::StoreInterface>(*op);
        name = storeOp.getUniqueId();
        limit = storeOp.getLimit().value_or(1);
      }
      if (!ifOps.empty()) {
        auto &ifOpTypes = ifOps.back();
        auto ifOp = ifOpTypes->ifOp;
        std::string memRsrc = name + "_" + getUnqiueName(ifOp) +
                              (ifOpTypes->inThen ? "then" : "else");
        resourceMap[op].push_back(memRsrc);
        resourceLimits.insert(std::pair(memRsrc, limit));
        auto &thenOrElseMap =
            ifOpTypes->inThen ? ifOpTypes->thenTypes : ifOpTypes->elseTypes;
        for (const auto &opr : thenOrElseMap[name]) {
          resourceMap[op].push_back(opr);
        }
        thenOrElseMap[name].push_back(memRsrc);
      } else {
        finalTypes[name].push_back(name);
        resourceLimits.insert(std::pair(name, limit));
      }

      for (const auto &opr : finalTypes[name]) {
        resourceMap[op].push_back(opr);
      }
    } else if (auto loop = dyn_cast<LoopInterface>(op)) {
      assert(ifOps.empty() &&
             "Loops inside if statements is unsupported currently");
      loop.getBodyBlock()->walk([&](Operation *innerOp) {
        std::string name;
        if (isa<LoopScheduleLoadOp, LoopScheduleStoreOp>(innerOp)) {
          Value memRef = getMemref(innerOp);
          name = "mem_" + std::to_string(hash_value(memRef));
          resourceLimits.insert(std::pair(name, 1));
          finalTypes[name].push_back(name);
        } else if (isa<LoadInterface, StoreInterface>(innerOp)) {
          std::optional<unsigned> limitOpt;
          if (auto loadOp = dyn_cast<loopschedule::LoadInterface>(*innerOp)) {
            limitOpt = loadOp.getLimit();
            name = loadOp.getUniqueId();
          } else if (auto storeOp =
                         dyn_cast<loopschedule::StoreInterface>(*innerOp)) {
            limitOpt = storeOp.getLimit();
            name = storeOp.getUniqueId();
          }
          if (limitOpt.has_value()) {
            finalTypes[name].push_back(name);
            resourceLimits.insert(std::pair(name, limitOpt.value()));
          }
        }
        for (const auto &opr : finalTypes[name]) {
          resourceMap[op].push_back(opr);
        }
      });
    }
  });

  return success();
}

LogicalResult addMemoryResources(Operation *op, Region &body,
                                 scheduling::SharedOperatorsProblem &problem,
                                 ResourceMap &resourceMap,
                                 ResourceLimits &resourceLimits) {

  for (const auto &it : resourceLimits) {
    auto memRsrc = problem.getOrInsertResourceType(it.getKey());
    problem.setLimit(memRsrc, it.getValue());
  }

  for (const auto &it : resourceMap) {
    auto *op = it.first;
    auto rsrcs = it.second;
    // An access occupies its resource in the cycle it STARTS, whatever its
    // latency, so a latency-0 access is reserved like any other. It used to be
    // skipped, on the reasoning that an operator finishing in the cycle it
    // starts is combinational and therefore free — true of its data path and
    // false of its port. A first-word-fall-through stream read is latency 0 by
    // design (its data is combinational off the queue head) and still strobes
    // one read enable, so N of them placed in one cycle dequeue ONE element
    // and hand it to all N readers. That is what happened to a merged stencil
    // window: three reads, one beat, three identical values, and an II the
    // hardware could not honour. Only ops that declared a resource are in this
    // map, so nothing that was already scheduled correctly gains a constraint
    // it did not have.
    if (!problem.getLinkedOperatorType(op).has_value())
      continue; // not part of this problem
    for (const auto &name : rsrcs) {
      auto memRsrc = problem.getOrInsertResourceType(name);
      problem.addLinkedResourceType(op, memRsrc);
    }
  }

  return success();
}

struct IfOpConversionPattern : OpConversionPattern<scf::IfOp> {
public:
  IfOpConversionPattern(MLIRContext *context, PredicateMap &predicateMap)
      : OpConversionPattern<scf::IfOp>(context), predicateMap(predicateMap) {}

  LogicalResult
  matchAndRewrite(scf::IfOp ifOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto getNewPredicate = [&](Operation *op, Value cond,
                               DenseMap<Value, Value> &condMap) {
      if (condMap.contains(cond))
        return condMap.lookup(cond);
      Value newCond = cond;
      if (predicateMap.contains(op)) {
        auto currCond = predicateMap.lookup(op);
        newCond =
            arith::AndIOp::create(rewriter, ifOp.getLoc(), currCond, cond);
      }
      condMap.insert(std::pair(cond, newCond));
      return newCond;
    };
    rewriter.modifyOpInPlace(ifOp, [&]() {
      if (!ifOp.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(ifOp.thenBlock(), --ifOp.thenBlock()->end());
        DenseMap<Value, Value> condMap;
        ifOp.getThenRegion().front().walk([&](Operation *op) {
          if (isa<scf::IfOp, scf::YieldOp>(op))
            return;
          Value newCond = getNewPredicate(op, ifOp.getCondition(), condMap);
          predicateMap[op] = newCond;
        });
        rewriter.inlineBlockBefore(&ifOp.getThenRegion().front(), ifOp);
      }
      if (ifOp.elseBlock() && !ifOp.elseBlock()->without_terminator().empty()) {
        rewriter.setInsertionPoint(ifOp);
        auto constOne = arith::ConstantOp::create(
            rewriter, ifOp.getLoc(),
            rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
        auto condNot = arith::XOrIOp::create(rewriter, ifOp.getLoc(),
                                             ifOp.getCondition(), constOne);
        rewriter.splitBlock(ifOp.elseBlock(), --ifOp.elseBlock()->end());
        DenseMap<Value, Value> condMap;
        ifOp.getElseRegion().front().walk([&](Operation *op) {
          if (isa<scf::IfOp, scf::YieldOp>(op))
            return;
          Value newCond = getNewPredicate(op, condNot, condMap);
          predicateMap[op] = newCond;
        });
        rewriter.inlineBlockBefore(&ifOp.getElseRegion().front(), ifOp);
      }
    });

    return success();
  }

private:
  PredicateMap &predicateMap;
};

struct IfToSelectPattern : OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.thenBlock()->without_terminator().empty() || !op.elseBlock()) {
      return failure();
    }

    if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
      return failure();
    }

    auto thenOperands = op.thenYield().getOperands();
    auto elseOperands = op.elseYield().getOperands();

    SmallVector<Value> newValues;
    for (auto v : llvm::zip(thenOperands, elseOperands)) {
      SmallVector<Value> operands;
      operands.push_back(op.getCondition());
      operands.push_back(std::get<0>(v));
      operands.push_back(std::get<1>(v));
      auto selectOp = arith::SelectOp::create(rewriter, op.getLoc(), operands);
      newValues.push_back(selectOp.getResult());
    }
    rewriter.replaceOp(op, newValues);

    return success();
  }
};

struct EmptyIfRemovalPattern : OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.thenBlock()->without_terminator().empty()) {
      return failure();
    }

    if (!op.thenYield().getResults().empty()) {
      return failure();
    }

    if (op.elseBlock()) {
      if (!op.elseBlock()->empty()) {
        return failure();
      }

      if (!op.elseYield().getResults().empty()) {
        return failure();
      }
    }

    rewriter.eraseOp(op);

    return success();
  }
};

LogicalResult ifOpConversion(Operation *op, Region &body,
                             PredicateMap &predicateMap) {
  predicateMap.clear();
  auto *ctx = op->getContext();
  ConversionTarget target(*ctx);
  target
      .addLegalDialect<arith::ArithDialect, scf::SCFDialect, func::FuncDialect,
                       loopschedule::LoopScheduleDialect>();
  target.addIllegalOp<scf::IfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<IfOpConversionPattern>(ctx, predicateMap);
  patterns.add<IfToSelectPattern>(ctx);
  patterns.add<EmptyIfRemovalPattern>(ctx);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  return success();
}

void addPredicateDependencies(Operation *op, Region &body,
                              scheduling::SharedOperatorsProblem &problem,
                              const PredicateMap &predicateMap,
                              PredicateUse &predicateUse) {
  predicateUse.clear();
  for (auto it : predicateMap) {
    auto *op = it.first;
    auto pred = it.second;
    predicateUse[pred].push_back(op);
    auto *definingOp = pred.getDefiningOp();
    assert(problem.hasOperation(definingOp));
    assert(problem.hasOperation(op));
    Problem::Dependence dep(definingOp, op);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
  }
}

void inlineLaunchExpectPairs(LoopSchedulePipelineOp pipOp) {
  SmallVector<LoopScheduleExpectOp> expects;
  pipOp.walk([&](LoopScheduleExpectOp e) { expects.push_back(e); });

  // Collect per-expect metadata before mutating IR. A single at-stage
  // can simultaneously be an issueStage for one expect and a destStage
  // for another (nested dynamic chains, or same-stage launch + expect),
  // so we combine all stage-level edits (handle-slot retype + forward-
  // slot drop) and rebuild each distinct stage at most once.
  struct PerExpect {
    LoopScheduleExpectOp expect;
    LoopScheduleLaunchOp launch;
    Operation *payload = nullptr;
    LoopScheduleAtOp issueStage;
    LoopScheduleAtOp destStage;
    unsigned issueSlot = ~0u; // issueStage at-yield slot holding handle
    unsigned destSlot = ~0u;  // destStage at-yield slot holding expect result
    // A zero-result payload (a dynamic STORE) threads no value: its handle
    // slot is dropped rather than retyped to a payload result, and its launch
    // is erased only after the stage rebuild frees the handle.
    bool hasResult = false;
  };
  SmallVector<PerExpect> per;
  per.reserve(expects.size());
  for (auto expect : expects) {
    PerExpect pe;
    pe.expect = expect;
    pe.launch = expect.getLaunchOp();
    if (!pe.launch)
      continue;
    for (Operation &op : pe.launch.getBody().front()) {
      if (isa<LoopScheduleYieldOp>(op))
        continue;
      pe.payload = &op;
      break;
    }
    if (!pe.payload)
      continue;
    pe.hasResult = pe.payload->getNumResults() > 0;
    pe.issueStage = pe.launch->getParentOfType<LoopScheduleAtOp>();
    pe.destStage = expect->getParentOfType<LoopScheduleAtOp>();
    if (!pe.issueStage || !pe.destStage)
      continue;
    for (auto [i, operand] :
         llvm::enumerate(pe.issueStage.getYieldOp()->getOperands())) {
      if (operand == pe.launch.getHandle()) {
        pe.issueSlot = i;
        break;
      }
    }
    // A store's 0-result expect threads no value through the dest yield.
    if (pe.hasResult) {
      for (auto [i, operand] :
           llvm::enumerate(pe.destStage.getYieldOp()->getOperands())) {
        if (operand == expect.getResult(0)) {
          pe.destSlot = i;
          break;
        }
      }
    }
    per.push_back(pe);
  }

  // Hoist payloads, RAUW expect → payload, and patch the issueStage
  // yield operand back to the payload's value. These edits are safe
  // before any stage rebuild because they only touch operands, not
  // result types.
  for (auto &pe : per) {
    pe.payload->moveBefore(pe.launch);
    if (pe.hasResult) {
      pe.expect.getResult(0).replaceAllUsesWith(pe.payload->getResult(0));
      if (pe.issueSlot != ~0u)
        pe.issueStage.getYieldOp()->setOperand(pe.issueSlot,
                                               pe.payload->getResult(0));
    }
    // For a store (no result) the issue-yield slot keeps holding the handle
    // until the stage rebuild below drops it; the launch is erased afterward.
  }

  // Erase expects now (no uses). Erase LOAD launches now too (the RAUW above
  // freed their handles); defer STORE launches until after the rebuild drops
  // their still-referenced handle slot.
  for (auto &pe : per) {
    pe.expect.erase();
    if (pe.hasResult)
      pe.launch.erase();
  }

  // Collect per-stage edit lists: slots whose type changes (issue-side
  // handle → payload value) and slots that get dropped (dest-side
  // forwarding slot). A stage may have both; a stage with neither stays
  // untouched.
  struct StageEdits {
    SmallVector<unsigned> retypeSlots;
    SmallVector<unsigned> dropSlots;
    // For each dropped slot, the replacement value cross-stage users
    // should consume. For issueStage-that-is-also-destStage, this is
    // the issueStage's own (soon-to-be-rebuilt) value slot — we resolve
    // it after the rebuild via a lookup.
    SmallVector<std::pair<unsigned, std::pair<LoopScheduleAtOp, unsigned>>>
        dropRedirects;
  };
  DenseMap<LoopScheduleAtOp, StageEdits> edits;
  SmallVector<LoopScheduleAtOp> stagesInOrder;
  auto ensure = [&](LoopScheduleAtOp s) -> StageEdits & {
    auto it = edits.find(s);
    if (it != edits.end())
      return it->second;
    stagesInOrder.push_back(s);
    return edits[s];
  };
  for (auto &pe : per) {
    if (!pe.hasResult) {
      // Store: drop the appended handle slot; nothing consumes it after the
      // expect is erased, so no redirect is needed.
      if (pe.issueSlot != ~0u)
        ensure(pe.issueStage).dropSlots.push_back(pe.issueSlot);
      continue;
    }
    if (pe.issueSlot != ~0u)
      ensure(pe.issueStage).retypeSlots.push_back(pe.issueSlot);
    if (pe.destSlot != ~0u) {
      auto &e = ensure(pe.destStage);
      e.dropSlots.push_back(pe.destSlot);
      e.dropRedirects.push_back({pe.destSlot, {pe.issueStage, pe.issueSlot}});
    }
  }

  // Rebuild each stage in document order so that when an earlier stage
  // (an issueStage) is rebuilt first, later stages' cross-stage refs
  // still resolve via the RAUW we do at rebuild time.
  DenseMap<LoopScheduleAtOp, LoopScheduleAtOp> replacement;
  SmallVector<LoopScheduleAtOp> docOrder;
  pipOp.walk([&](LoopScheduleAtOp s) {
    if (edits.count(s))
      docOrder.push_back(s);
  });

  for (auto oldStage : docOrder) {
    auto &e = edits[oldStage];
    auto yield = oldStage.getYieldOp();

    // Compute the new result-type list: keep existing types, swap
    // retype slots to the current yield operand's type, then strip
    // dropped slots.
    SmallVector<Type> newTypes(oldStage.getResultTypes());
    for (unsigned slot : e.retypeSlots)
      newTypes[slot] = yield->getOperand(slot).getType();

    llvm::SmallDenseSet<unsigned> dropSet(e.dropSlots.begin(),
                                          e.dropSlots.end());
    SmallVector<Type> finalTypes;
    SmallVector<unsigned> keptSlots;
    for (unsigned i = 0, end = newTypes.size(); i < end; ++i) {
      if (dropSet.count(i))
        continue;
      finalTypes.push_back(newTypes[i]);
      keptSlots.push_back(i);
    }

    // Drop yield operands (highest index first).
    SmallVector<unsigned> sortedDrops(e.dropSlots.begin(), e.dropSlots.end());
    llvm::sort(sortedDrops, std::greater<unsigned>());
    for (unsigned slot : sortedDrops)
      yield->eraseOperand(slot);

    // Nothing actually changed? Skip the rebuild.
    bool noChange = finalTypes == SmallVector<Type>(oldStage.getResultTypes());
    if (noChange) {
      replacement[oldStage] = oldStage;
      continue;
    }

    OpBuilder rebuild(oldStage);
    auto newStage = rebuild.create<LoopScheduleAtOp>(
        oldStage.getLoc(), finalTypes, oldStage.getOffsetAttr());
    newStage.getBody().takeBody(oldStage.getBody());
    // Redirect kept-slot users directly; for dropped slots, redirect to
    // the issueStage's (possibly already rebuilt) value slot.
    for (auto [newIdx, oldIdx] : llvm::enumerate(keptSlots))
      oldStage.getResult(oldIdx).replaceAllUsesWith(newStage.getResult(newIdx));
    for (auto &rd : e.dropRedirects) {
      unsigned droppedSlot = rd.first;
      auto [issueStage, issueSlot] = rd.second;
      if (issueSlot == ~0u)
        continue;
      // The issueStage may have been rebuilt already; look up its
      // replacement. Self-cycle (issueStage == oldStage) resolves to
      // newStage.
      LoopScheduleAtOp resolvedIssue;
      if (issueStage == oldStage)
        resolvedIssue = newStage;
      else
        resolvedIssue = replacement.lookup(issueStage);
      if (!resolvedIssue)
        resolvedIssue = issueStage; // untouched stage
      oldStage.getResult(droppedSlot)
          .replaceAllUsesWith(resolvedIssue.getResult(issueSlot));
    }
    replacement[oldStage] = newStage;
    oldStage.erase();
  }

  // Erase the deferred STORE launches now that the rebuild has dropped their
  // handle slots from the (reparented) issue-stage yields, freeing the handle.
  for (auto &pe : per)
    if (!pe.hasResult)
      pe.launch.erase();
}

} // namespace loopschedule
} // namespace circt
