//===- SchedulingAnalysis.cpp - scheduling analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving scheduling.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LogicalResult.h"
#include <cassert>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::scf;
using namespace circt::loopschedule;

/// CyclicSchedulingAnalysis constructs a CyclicProblem for each AffineForOp by
/// performing a memory dependence analysis and inserting dependences into the
/// problem. The client should retrieve the partially complete problem to add
/// and associate operator types.
circt::analysis::CyclicSchedulingAnalysis::CyclicSchedulingAnalysis(
    Operation *op, AnalysisManager &am) {
  auto funcOp = cast<func::FuncOp>(op);

  MemoryDependenceAnalysis &memoryAnalysis =
      am.getAnalysis<MemoryDependenceAnalysis>();

  // Only consider loops marked with pipeline attribute.
  op->walk<WalkOrder::PreOrder>([&](AffineForOp op) {
    if (!op->hasAttr("hls.pipeline"))
      return;

    analyzeForOp(op, memoryAnalysis);
  });
}

void circt::analysis::CyclicSchedulingAnalysis::analyzeForOp(
    AffineForOp forOp, MemoryDependenceAnalysis memoryAnalysis) {
  // Create a cyclic scheduling problem.
  CyclicProblem problem = CyclicProblem::get(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences = memoryAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (MemoryDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      // memoryDep.source->dump();
      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
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

  // Insert conditional dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    Block *thenBlock = nullptr;
    Block *elseBlock = nullptr;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      thenBlock = ifOp.thenBlock();
      elseBlock = ifOp.elseBlock();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      thenBlock = ifOp.getThenBlock();
      if (ifOp.hasElse())
        elseBlock = ifOp.getElseBlock();
    } else {
      return WalkResult::advance();
    }

    // No special handling required for control-only `if`s.
    if (op->getNumResults() == 0)
      return WalkResult::skip();

    // Model the implicit value flow from the `yield` to the `if`'s result(s).
    Dependence depThen(thenBlock->getTerminator(), op);
    auto depInserted = problem.insertDependence(depThen);
    assert(succeeded(depInserted));
    (void)depInserted;

    if (elseBlock) {
      Dependence depElse(elseBlock->getTerminator(), op);
      depInserted = problem.insertDependence(depElse);
      assert(succeeded(depInserted));
      (void)depInserted;
    }

    return WalkResult::advance();
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
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

  // Store the partially complete problem.
  problems.insert(std::pair<Operation *, CyclicProblem>(forOp, problem));
}

CyclicProblem &
circt::analysis::CyclicSchedulingAnalysis::getProblem(AffineForOp forOp) {
  auto problem = problems.find(forOp);
  assert(problem != problems.end() && "expected problem to exist");
  return problem->second;
}

/// SharedOperatorsSchedulingAnalysis constructs a SharedOperatorsProblem for
/// each AffineForOp by performing a memory dependence analysis and inserting
/// dependences into the problem. The client should retrieve the partially
/// complete problem to add and associate operator types.
circt::analysis::SharedOperatorsSchedulingAnalysis::
    SharedOperatorsSchedulingAnalysis(Operation *op, AnalysisManager &am)
    : memoryAnalysis(am.getAnalysis<MemoryDependenceAnalysis>()) {}

void circt::analysis::SharedOperatorsSchedulingAnalysis::analyzeForOp(
    AffineForOp forOp, MemoryDependenceAnalysis memoryAnalysis) {
  // Create a cyclic scheduling problem.
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(forOp);

  // Insert memory dependences into the problem.
  assert(forOp.getLoopRegions().size() == 1);
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopInterface>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences = memoryAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (const MemoryDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      if (memoryDep.source->getParentOfType<LoopInterface>() != nullptr)
        return;

      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // DenseMap<Operation *, SmallVector<LoopInterface>> memOps;
  // forOp.getLoopBody().walk([&](Operation *op) {
  //   for (auto &region : op->getRegions()) {
  //     for (auto loop : region.getOps<LoopInterface>()) {
  //       loop.getBodyBlock()->walk([&](Operation *op) {
  //         if (isa<AffineLoadOp, AffineStoreOp, memref::LoadOp,
  //         memref::StoreOp,
  //                 LoadInterface, StoreInterface>(op)) {
  //           memOps[op].push_back(loop);
  //         }
  //       });
  //     }
  //   }
  // });

  // forOp.getLoopBody().walk([&](LoopInterface loop) {
  //   for (auto it : memOps) {
  //     auto *memOp = it.getFirst();
  //     auto dependences = memoryAnalysis.getDependences(memOp);
  //     for (const MemoryDependence &memoryDep : dependences) {
  //       if (!hasDependence(memoryDep.dependenceType))
  //         continue;

  //       for (auto otherLoop : memOps[memoryDep.source]) {
  //         if (loop == otherLoop || !loop->isAncestor(otherLoop))
  //           continue;

  //         Dependence dep(loop, otherLoop);
  //         auto depInserted = problem.insertDependence(dep);
  //         assert(succeeded(depInserted));
  //       }
  //     }
  //   }
  // });

  // Insert conditional dependences into the problem.
  // forOp.getLoopBody().walk([&](Operation *op) {
  //   if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
  //       op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
  //     return WalkResult::advance();
  //   Block *thenBlock = nullptr;
  //   Block *elseBlock = nullptr;
  //   if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
  //     thenBlock = ifOp.thenBlock();
  //     elseBlock = ifOp.elseBlock();
  //   } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
  //     thenBlock = ifOp.getThenBlock();
  //     if (ifOp.hasElse())
  //       elseBlock = ifOp.getElseBlock();
  //   } else {
  //     return WalkResult::advance();
  //   }

  //   // No special handling required for control-only `if`s.
  //   if (op->getNumResults() == 0)
  //     return WalkResult::skip();

  //   // Model the implicit value flow from the `yield` to the `if`'s
  //   result(s). Dependence depThen(thenBlock->getTerminator(), op);
  //   auto depInserted = problem.insertDependence(depThen);
  //   assert(succeeded(depInserted));
  //   (void)depInserted;

  //   if (elseBlock) {
  //     Dependence depElse(elseBlock->getTerminator(), op);
  //     depInserted = problem.insertDependence(depElse);
  //     assert(succeeded(depInserted));
  //     (void)depInserted;
  //   }

  //   return WalkResult::advance();
  // });

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
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Store the partially complete problem.
  problems.insert(
      std::pair<Operation *, SharedOperatorsProblem>(forOp, problem));
}

void circt::analysis::SharedOperatorsSchedulingAnalysis::analyzeFuncOp(
    func::FuncOp funcOp, MemoryDependenceAnalysis memoryAnalysis) {
  // Create a cyclic scheduling problem.
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(funcOp);
  llvm::errs() << "func sched\n";

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    op->dump();
    // Insert every operation into the problem.
    problem.insertOperation(op);

    // ArrayRef<MemoryDependence> dependences =
    // memoryAnalysis.getDependences(op); if (dependences.empty())
    //   return;

    // for (const MemoryDependence &memoryDep : dependences) {
    //   // memoryDep.source->dump();
    //   // Don't insert a dependence into the problem if there is no
    //   dependence. if (!hasDependence(memoryDep.dependenceType))
    //     continue;
    //   // Insert a dependence into the problem.
    //   Dependence dep(memoryDep.source, op);
    //   auto depInserted = problem.insertDependence(dep);
    //   assert(succeeded(depInserted));
    //   (void)depInserted;
    // }
  });

  // DenseMap<Operation *, SmallVector<LoopInterface>> memOps;
  // funcOp.getBody().walk([&](LoopInterface loop) {
  //   loop.getBodyBlock()->walk([&](Operation *op) {
  //     if (isa<AffineLoadOp, AffineStoreOp, memref::LoadOp, memref::StoreOp,
  //             LoadInterface, StoreInterface>(op)) {
  //       memOps[op].push_back(loop);
  //     }
  //   });
  // });

  // funcOp.getBody().walk([&](LoopInterface loop) {
  //   for (auto it : memOps) {
  //     auto *memOp = it.getFirst();
  //     auto dependences = memoryAnalysis.getDependences(memOp);
  //     for (const MemoryDependence &memoryDep : dependences) {
  //       if (!hasDependence(memoryDep.dependenceType))
  //         continue;
  //       for (auto otherLoop : memOps[memoryDep.source]) {
  //         if (loop == otherLoop || !loop->isAncestor(otherLoop))
  //           continue;
  //         Dependence dep(loop, otherLoop);
  //         auto depInserted = problem.insertDependence(dep);
  //         assert(succeeded(depInserted));
  //       }
  //     }
  //   }
  // });

  // Insert conditional dependences into the problem.
  // funcOp.getBody().walk([&](Operation *op) {
  //   if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
  //       op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
  //     return WalkResult::advance();

  //   Block *thenBlock = nullptr;
  //   Block *elseBlock = nullptr;
  //   if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
  //     thenBlock = ifOp.thenBlock();
  //     elseBlock = ifOp.elseBlock();
  //   } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
  //     thenBlock = ifOp.getThenBlock();
  //     if (ifOp.hasElse())
  //       elseBlock = ifOp.getElseBlock();
  //   } else {
  //     return WalkResult::advance();
  //   }

  //   // No special handling required for control-only `if`s.
  //   if (op->getNumResults() == 0)
  //     return WalkResult::skip();

  //   // Model the implicit value flow from the `yield` to the `if`'s
  //   result(s). Dependence depThen(thenBlock->getTerminator(), op);
  //   auto depInserted = problem.insertDependence(depThen);
  //   assert(succeeded(depInserted));
  //   (void)depInserted;

  //   if (elseBlock) {
  //     Dependence depElse(elseBlock->getTerminator(), op);
  //     depInserted = problem.insertDependence(depElse);
  //     assert(succeeded(depInserted));
  //     (void)depInserted;
  //   }

  //   return WalkResult::advance();
  // });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = funcOp.getBody().back().getTerminator();
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;
    if (!isa<AffineStoreOp, memref::StoreOp, StoreInterface>(op))
      return;
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Store the partially complete problem.
  problems.insert(
      std::pair<Operation *, SharedOperatorsProblem>(funcOp, problem));
}

SharedOperatorsProblem &
circt::analysis::SharedOperatorsSchedulingAnalysis::getProblem(Operation *op) {
  // auto problem = problems.find(op);
  // if (problem != problems.end()) {
  //   return problem->second;
  // }
  if (auto forOp = dyn_cast<AffineForOp>(op); forOp)
    analyzeForOp(forOp, this->memoryAnalysis);
  else if (auto funcOp = dyn_cast<func::FuncOp>(op); funcOp)
    analyzeFuncOp(funcOp, this->memoryAnalysis);
  else
    op->emitOpError("Unsupported operation for scheduling");
  auto problem = problems.find(op);
  assert(problem != problems.end() && "expected problem to exist");
  return problem->second;
}
