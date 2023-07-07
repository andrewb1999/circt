//===- UnrollForLoopSchedule.cpp - Unroll nested loops for parallelism ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;

namespace {
struct UnrollForLoopSchedule
    : public circt::UnrollForLoopScheduleBase<UnrollForLoopSchedule> {
  using UnrollForLoopScheduleBase<
      UnrollForLoopSchedule>::UnrollForLoopScheduleBase;
  void runOnOperation() override;

private:
  bool hasAllocDefinition(TypedValue<MemRefType> val);
  bool usesExternalMemory(AffineForOp affineFor);
  DenseSet<AffineMapAccessInterface> updateMemoryOps(AffineForOp affineFor);
  uint64_t getMinDepDistance(AffineForOp affineFor);
  std::optional<uint64_t> consumePragma(AffineForOp affineFor);
  AffineForOp cloneIntoNewBlock(AffineForOp affineFor, IRMapping &irMapping,
                                DenseSet<Operation *> &dontTouchSet);
  LogicalResult unrollForDataParallel(AffineForOp affineFor);
  std::pair<SmallVector<AffineForOp>, unsigned>
  getDeepestNestedForOps(std::pair<SmallVector<AffineForOp>, unsigned> pair);
  LogicalResult unrollForPipelineParallel(AffineForOp affineFor);
  DenseMap<AffineForOp, DenseSet<AffineMapAccessInterface>> loopToMemOps;
  DenseMap<Operation *, unsigned> approxASAP;
  MemoryDependenceAnalysis *memDepAnalysis;
};
} // namespace

bool UnrollForLoopSchedule::hasAllocDefinition(TypedValue<MemRefType> val) {
  Operation *definitionOp = val.getDefiningOp();
  return definitionOp != nullptr && (isa<memref::AllocOp>(*definitionOp) ||
                                     isa<memref::AllocaOp>(*definitionOp));
};

bool UnrollForLoopSchedule::usesExternalMemory(AffineForOp affineFor) {

  WalkResult result = affineFor.getBody()->walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AffineLoadOp>([&](Operation *op) {
          return hasAllocDefinition(cast<AffineLoadOp>(*op).getMemref())
                     ? WalkResult::advance()
                     : WalkResult::interrupt();
        })
        .Case<AffineStoreOp>([&](Operation *op) {
          return hasAllocDefinition(cast<AffineStoreOp>(*op).getMemref())
                     ? WalkResult::advance()
                     : WalkResult::interrupt();
        })
        .Case<memref::LoadOp>([&](Operation *op) {
          return hasAllocDefinition(cast<memref::LoadOp>(*op).getMemref())
                     ? WalkResult::advance()
                     : WalkResult::interrupt();
        })
        .Case<memref::StoreOp>([&](Operation *op) {
          return hasAllocDefinition(cast<memref::StoreOp>(*op).getMemref())
                     ? WalkResult::advance()
                     : WalkResult::interrupt();
        })
        .Default([&](Operation *op) { return WalkResult::advance(); });
  });

  return result.wasInterrupted();
}

// We need to keep track of memory ops in within the body to understand the
// amount of pipeline parallelism that exists within
DenseSet<AffineMapAccessInterface>
UnrollForLoopSchedule::updateMemoryOps(AffineForOp affineFor) {
  DenseSet<AffineMapAccessInterface> memOps;
  auto &bodyRegion = affineFor.getLoopBody();
  // Only affine memory dependencies are supported
  auto affineLdOps = bodyRegion.getOps<AffineLoadOp>();
  auto affineStOps = bodyRegion.getOps<AffineStoreOp>();
  memOps.insert(affineLdOps.begin(), affineLdOps.end());
  memOps.insert(affineStOps.begin(), affineStOps.end());
  loopToMemOps[affineFor] = memOps;
  return memOps;
}

// Find the limiting carried dependence and get its distance
uint64_t UnrollForLoopSchedule::getMinDepDistance(AffineForOp affineFor) {
  auto memOps = loopToMemOps[affineFor];
  uint64_t minDistance = std::numeric_limits<uint64_t>::max();
  for (auto memOp : memOps) {

    ArrayRef<MemoryDependence> dependences =
        memDepAnalysis->getDependences(memOp);
    if (dependences.empty())
      continue;

    for (MemoryDependence memoryDep : dependences) {
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      auto lb = memoryDep.dependenceComponents.back().lb;
      if (!lb.has_value())
        continue;
      int64_t distanceComp = *memoryDep.dependenceComponents.back().lb;
      minDistance =
          std::min(minDistance, distanceComp < 0 ? (uint64_t)distanceComp * -1
                                                 : (uint64_t)distanceComp);
    }
  }

  return minDistance;
}

// Get unroll factor from IR if it is there
std::optional<uint64_t>
UnrollForLoopSchedule::consumePragma(AffineForOp affineFor) {
  auto unrollAttr = affineFor->getAttr("hls.unroll");
  std::optional<uint64_t> maxUnrollFactor;
  if (unrollAttr == nullptr)
    return maxUnrollFactor;
  if (unrollAttr.isa<UnitAttr>())
    maxUnrollFactor = std::numeric_limits<uint64_t>::max();
  if (unrollAttr.isa<StringAttr>()) {
    auto val = unrollAttr.cast<StringAttr>().getValue();
    if (val == "full")
      maxUnrollFactor = std::numeric_limits<uint64_t>::max();
  } else if (unrollAttr.isa<IntegerAttr>()) {
    maxUnrollFactor = unrollAttr.cast<IntegerAttr>().getInt() < 1
                          ? std::numeric_limits<uint64_t>::max()
                          : unrollAttr.cast<IntegerAttr>().getInt();
  }
  affineFor->removeAttr("hls.unroll");
  return maxUnrollFactor;
}

// Clone an affine loop into its own isolated block. Not all values in scope are
// copied over
AffineForOp
UnrollForLoopSchedule::cloneIntoNewBlock(AffineForOp affineFor,
                                         IRMapping &irMapping,
                                         DenseSet<Operation *> &dontTouchSet) {

  OpBuilder builder(affineFor);
  auto originalPt = builder.saveInsertionPoint();
  auto *originalBlk = originalPt.getBlock();

  SmallVector<Location> locs;
  for (auto arg : originalBlk->getArguments())
    locs.push_back(arg.getLoc());
  auto *tmpBlk =
      builder.createBlock(originalBlk, originalBlk->getArgumentTypes(), locs);

  for (unsigned i = 0; i < originalBlk->getNumArguments(); ++i)
    irMapping.map(originalBlk->getArgument(i), tmpBlk->getArgument(i));

  for (auto &op : affineFor->getBlock()->getOperations()) {
    auto *clonedOp = builder.clone(op, irMapping);
    if (isa<AffineForOp>(op) && cast<AffineForOp>(op) == affineFor)
      return cast<AffineForOp>(*clonedOp);
    dontTouchSet.insert(clonedOp);
  }

  assert(false && "unreachable");
  return nullptr;
}

// Look for data-level parallelism towards the top level
LogicalResult
UnrollForLoopSchedule::unrollForDataParallel(AffineForOp affineFor) {
  std::optional<uint64_t> maxUnrollFactor = consumePragma(affineFor);

  if (usesExternalMemory(affineFor))
    return success();

  // We are not checking for loop-carried dependencies
  // Looking for perfect-nesting so just use max value
  uint64_t unrollFactor =
      maxUnrollFactor.value_or(std::numeric_limits<uint64_t>::max());

  // If this loop nest is not perfectly nested return
  SmallVector<AffineForOp> innerLoopCount(affineFor.getOps<AffineForOp>());
  bool perfectlyNested =
      innerLoopCount.size() == 1 && isPerfectlyNested(SmallVector<AffineForOp>(
                                        {affineFor, innerLoopCount.back()}));
  if (!perfectlyNested)
    return success();

  // We need to know if we are fully-unrolling, because then the loop body
  // will get promoted to the containing region. And the anchoring ForOp will
  // change.
  bool loopIsPromoted =
      unrollFactor >= getConstantTripCount(affineFor).value_or(
                          std::numeric_limits<uint64_t>::max() - 1);

  if (unrollFactor <= 1)
    return success();

  // Since the unrolled loop may be optimized/promoted away, I put the for
  // loop in its own block to isolated the new anchoring ForOp
  IRMapping irMapping;
  DenseSet<Operation *> dontTouchSet;
  auto tmpFor = cloneIntoNewBlock(affineFor, irMapping, dontTouchSet);
  auto *tmpBlk = tmpFor->getBlock();

  if (loopUnrollUpToFactor(tmpFor, unrollFactor).failed())
    return failure();

  // Since we isolated the unrolled loop in its own block we have isolated all
  // the ops that can be trivially fused together
  Block *blockToRead =
      loopIsPromoted ? tmpBlk : &tmpFor.getLoopBody().getBlocks().front();
  SmallVector<AffineForOp> innerLoops;
  for (auto loop : blockToRead->getOps<AffineForOp>())
    if (!dontTouchSet.contains(loop))
      innerLoops.push_back(loop);

  // Check if memory dependencies limit fusion, if so abandon our work
  auto *tmpDependence = &getAnalysis<MemoryDependenceAnalysis>();
  for (auto loopI : innerLoops)
    for (auto memOp : updateMemoryOps(loopI))
      for (const auto &dep : tmpDependence->getDependences(memOp))
        for (auto comp : dep.dependenceComponents)
          for (auto loopJ : innerLoops)
            if (dep.dependenceType == DependenceResult::HasDependence &&
                loopI != loopJ && loopJ->isAncestor(comp.op)) {
              tmpBlk->erase();
              return success();
            }

  // Now we do the merging/fusing of loops
  auto destLoop = innerLoops.front();
  auto &destBlk = destLoop.getRegion().getBlocks().back();
  OpBuilder builder(affineFor);
  IRRewriter rewriter(builder);

  // First save operations in order, because we will be mutating them
  SmallVector<Operation *> operations;
  for (auto &op : blockToRead->getOperations())
    operations.push_back(&op);

  // Keep a running list of values that the loop body depends on, because
  // they will need to be moved inorder to dominate the relocation of the
  // loop body
  SmallVector<Operation *> operationsToMove;
  for (auto *op : operations) {
    if (!isa<AffineForOp>(*op) || dontTouchSet.contains(op)) {
      operationsToMove.push_back(op);
      continue;
    }

    auto srcLoop = cast<AffineForOp>(*op);
    if (srcLoop == destLoop) {
      operationsToMove.clear();
      continue;
    }

    // Move the dominating ops to above the destLoop
    for (auto &opToMove : operationsToMove)
      opToMove->moveBefore(destLoop);

    operationsToMove.clear();
    auto &srcBlk = srcLoop.getRegion().getBlocks().back();
    destBlk.getTerminator()->erase();
    rewriter.mergeBlocks(&srcBlk, &destBlk, destBlk.getArguments());
  }

  // Point old ops to the cloned & moved ones
  for (auto opMapping : irMapping.getOperationMap())
    opMapping.first->replaceAllUsesWith(opMapping.second->getResults());

  // Get rid of ops whose bodies we relocated
  innerLoops.erase(innerLoops.begin());
  for (auto innerLoop : innerLoops)
    innerLoop->erase();

  // Now we dump the fused-loop block in the location of the original loop
  rewriter.inlineBlockBefore(tmpBlk, affineFor,
                             affineFor->getBlock()->getArguments());

  auto *containingBlk = affineFor->getBlock();
  affineFor.erase();

  // Now do cleanup of duplicate Ops from cloning step
  SmallVector<Operation *> prologueToDelete;
  for (auto &op : containingBlk->getOperations()) {
    if (isa<AffineForOp>(op) && cast<AffineForOp>(op) == destLoop) {
      break;
    }
    if (op.getUses().empty() && !dontTouchSet.contains(&op))
      prologueToDelete.push_back(&op);
  }

  for (auto *op : prologueToDelete)
    op->erase();

  return success();
}

// This looks for pipeline parallelism via unrolling to increase the amount of
// work per II
LogicalResult
UnrollForLoopSchedule::unrollForPipelineParallel(AffineForOp affineFor) {

  updateMemoryOps(affineFor);
  uint64_t minDepDistance = getMinDepDistance(affineFor);
  std::optional<uint64_t> maxUnrollFactor = consumePragma(affineFor);

  // Unroll factor is roughly the min recurrence II : Latency / Distance

  // Latency is simply ASAP times, given a latency model for the operations
  // However, latency can be bounded above by the number of ops for now
  assert(affineFor.getOps<AffineForOp>().empty());
  uint64_t largeLatencyBound = 0;
  for (auto &op : affineFor.getOps())
    if (isa<arith::MulIOp>(op))
      largeLatencyBound += 3; // Mul will typically be longer latency
    else
      ++largeLatencyBound;

  // Do ceiling division
  minDepDistance = std::max(1UL, minDepDistance);
  uint64_t approxII = largeLatencyBound / minDepDistance +
                      (largeLatencyBound % minDepDistance > 0);

  uint64_t unrollFactor = std::min(
      approxII, maxUnrollFactor.value_or(std::numeric_limits<uint64_t>::max()));

  OpBuilder builder(affineFor);
  affineFor->setAttr("hls.pipeline", builder.getUnitAttr());

  if (unrollFactor <= 1)
    return success();

  if (loopUnrollUpToFactor(affineFor, unrollFactor).failed())
    return failure();

  return success();
}

std::pair<SmallVector<AffineForOp>, unsigned>
UnrollForLoopSchedule::getDeepestNestedForOps(
    std::pair<SmallVector<AffineForOp>, unsigned> pair) {
  auto roots = pair.first;
  auto rootDepth = pair.second;
  for (auto root : roots) {
    SmallVector<AffineForOp> nestedOps(root.getOps<AffineForOp>());
    if (nestedOps.empty())
      continue;
    auto recur = getDeepestNestedForOps(std::pair(nestedOps, rootDepth + 1));
    if (recur.second > pair.second) {
      pair.second = recur.second;
      pair.first = recur.first;
    }
  }
  return pair;
}

void UnrollForLoopSchedule::runOnOperation() {
  SmallVector<AffineForOp> rootForOps(getOperation().getOps<AffineForOp>());
  for (auto loop : rootForOps)
    if (unrollForDataParallel(loop).failed())
      return signalPassFailure();

  // After we unroll and merge, get rid of redundant loads
  affineScalarReplace(getOperation(), getAnalysis<DominanceInfo>(),
                      getAnalysis<PostDominanceInfo>());

  SmallVector<AffineForOp> opsToPipeline =
      getDeepestNestedForOps(
          std::pair(
              SmallVector<AffineForOp>(getOperation().getOps<AffineForOp>()),
              0))
          .first;

  // Get scheduling analyses
  memDepAnalysis = &getAnalysis<MemoryDependenceAnalysis>();

  for (auto loop : opsToPipeline)
    if (unrollForPipelineParallel(loop).failed())
      return signalPassFailure();
}

namespace circt {
std::unique_ptr<mlir::Pass> createUnrollForLoopSchedulePass() {
  return std::make_unique<UnrollForLoopSchedule>();
}
} // namespace circt
