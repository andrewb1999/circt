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
  ModuloProblem problem(forOp);

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

      unsigned distance = memoryDep.distance;
      // if (distance > 0)
      //   continue;
      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
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

ChainingModuloProblem
getChainingModuloProblem(scf::ForOp forOp,
                         LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  // Create a modulo scheduling problem.
  ChainingModuloProblem problem(forOp);

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

      unsigned distance = memoryDep.distance;
      // if (distance > 0)
      //   continue;
      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
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
  SharedOperatorsProblem problem(forOp);

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

ChainingSharedOperatorsProblem getChainingSharedOperatorsProblem(
    scf::ForOp forOp, LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  ChainingSharedOperatorsProblem problem(forOp);

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
            Dependence dep(operand.getDefiningOp(), op);
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
            Dependence dep(operand.getDefiningOp(), op);
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
      if (isa<LoopScheduleLoadOp, LoopScheduleStoreOp>(op)) {
        Value memRef = getMemref(op);
        name = "mem_" + std::to_string(hash_value(memRef));
      } else if (auto loadOp = dyn_cast<loopschedule::LoadInterface>(*op)) {
        name = loadOp.getUniqueId();
      } else {
        auto storeOp = cast<loopschedule::StoreInterface>(*op);
        name = storeOp.getUniqueId();
      }
      if (!ifOps.empty()) {
        auto &ifOpTypes = ifOps.back();
        auto ifOp = ifOpTypes->ifOp;
        std::string memRsrc = name + "_" + getUnqiueName(ifOp) +
                              (ifOpTypes->inThen ? "then" : "else");
        resourceMap[op].push_back(memRsrc);
        resourceLimits.insert(std::pair(memRsrc, 1));
        auto &thenOrElseMap =
            ifOpTypes->inThen ? ifOpTypes->thenTypes : ifOpTypes->elseTypes;
        for (const auto &opr : thenOrElseMap[name]) {
          resourceMap[op].push_back(opr);
        }
        thenOrElseMap[name].push_back(memRsrc);
      } else {
        finalTypes[name].push_back(name);
        resourceLimits.insert(std::pair(name, 1));
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
    problem.setResourceLimit(memRsrc, it.getValue());
  }

  for (const auto &it : resourceMap) {
    auto *op = it.first;
    auto rsrcs = it.second;
    auto opr = problem.getLinkedOperatorType(op);
    if (opr.has_value()) {
      auto latency = problem.getLatency(opr.value());
      if (latency != 0) {
        for (const auto &name : rsrcs) {
          auto memRsrc = problem.getOrInsertResourceType(name);
          problem.addResourceType(op, memRsrc);
        }
      }
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
        newCond = rewriter.create<arith::AndIOp>(ifOp.getLoc(), currCond, cond);
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
        auto constOne = rewriter.create<arith::ConstantOp>(
            ifOp.getLoc(), rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
        auto condNot = rewriter.create<arith::XOrIOp>(
            ifOp.getLoc(), ifOp.getCondition(), constOne);
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
      auto selectOp = rewriter.create<arith::SelectOp>(op.getLoc(), operands);
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
    Dependence dep(definingOp, op);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
  }
}

} // namespace loopschedule
} // namespace circt
