//===- PipelineForLoopSchedule.cpp - Pipeline and unroll innermost loops
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;
using namespace circt::loopschedule;

namespace {
struct PipelineForLoopSchedule
    : public PipelineForLoopScheduleBase<PipelineForLoopSchedule> {
  using PipelineForLoopScheduleBase<
      PipelineForLoopSchedule>::PipelineForLoopScheduleBase;
  void runOnOperation() override;

private:
  DenseSet<AffineMapAccessInterface> updateMemoryOps(AffineForOp affineFor);
  uint64_t getMinDepDistance(AffineForOp affineFor);
  std::optional<uint64_t> consumePragma(AffineForOp affineFor);
  std::pair<SmallVector<AffineForOp>, unsigned>
  getDeepestNestedForOps(std::pair<SmallVector<AffineForOp>, unsigned> pair);
  LogicalResult unrollForPipelineParallel(AffineForOp affineFor);
  DenseMap<AffineForOp, DenseSet<AffineMapAccessInterface>> loopToMemOps;
  MemoryDependenceAnalysis *memDepAnalysis;
};
} // namespace

// We need to keep track of memory ops in within the body to understand the
// amount of pipeline parallelism that exists within
DenseSet<AffineMapAccessInterface>
PipelineForLoopSchedule::updateMemoryOps(AffineForOp affineFor) {
  DenseSet<AffineMapAccessInterface> memOps;
  assert(affineFor.getLoopRegions().size() == 1);
  auto &bodyRegion = *affineFor.getLoopRegions().front();
  // Only affine memory dependencies are supported
  auto affineLdOps = bodyRegion.getOps<AffineLoadOp>();
  auto affineStOps = bodyRegion.getOps<AffineStoreOp>();
  memOps.insert(affineLdOps.begin(), affineLdOps.end());
  memOps.insert(affineStOps.begin(), affineStOps.end());
  loopToMemOps[affineFor] = memOps;
  return memOps;
}

// Find the limiting carried dependence and get its distance
uint64_t PipelineForLoopSchedule::getMinDepDistance(AffineForOp affineFor) {

  // We don't have real dependence analysis on scf/memref ops so this is good
  // enough for now
  if (!affineFor.getOps<memref::StoreOp>().empty())
    return 1;

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
PipelineForLoopSchedule::consumePragma(AffineForOp affineFor) {
  auto unrollAttr = affineFor->getAttr("hls.unroll");
  std::optional<uint64_t> maxUnrollFactor;
  if (unrollAttr == nullptr)
    return maxUnrollFactor;
  if (isa<StringAttr>(unrollAttr)) {
    auto val = cast<StringAttr>(unrollAttr).getValue();
    if (val == "full")
      maxUnrollFactor = std::numeric_limits<uint64_t>::max();
    else if (val == "none")
      maxUnrollFactor = std::numeric_limits<uint64_t>::min();
  } else if (isa<IntegerAttr>(unrollAttr)) {
    maxUnrollFactor = cast<IntegerAttr>(unrollAttr).getInt() < 1
                          ? std::numeric_limits<uint64_t>::max()
                          : cast<IntegerAttr>(unrollAttr).getInt();
  } else if (isa<UnitAttr>(unrollAttr)) {
    maxUnrollFactor = std::numeric_limits<uint64_t>::max();
  }
  affineFor->removeAttr("hls.unroll");
  return maxUnrollFactor;
}

// This looks for pipeline parallelism via unrolling to increase the amount of
// work per II
LogicalResult
PipelineForLoopSchedule::unrollForPipelineParallel(AffineForOp affineFor) {

  updateMemoryOps(affineFor);
  uint64_t minDepDistance = getMinDepDistance(affineFor);
  std::optional<uint64_t> maxUnrollFactor = consumePragma(affineFor);

  // Unroll factor is roughly the min recurrence II : Latency / Distance

  // Latency is simply ASAP times / critical path, given a latency model for the
  // operations. However, latency can be bounded above by the number of ops for
  // now. Here is a super hacky heuristic to know how much to unroll by:
  assert(affineFor.getOps<AffineForOp>().empty());
  uint64_t largeLatencyBound = 0;
  for (auto &op : affineFor.getOps())
    if (isa<arith::MulIOp>(op))
      largeLatencyBound += 3; // Mul will typically be longer latency
    else if (!isa<AffineMapAccessInterface>(op) && op.getNumOperands() >= 2)
      largeLatencyBound += op.getNumOperands();

  largeLatencyBound /=
      std::max(1UL, (uint64_t)affineFor.getNumRegionIterArgs());
  // Do ceiling division
  minDepDistance = std::max(1UL, minDepDistance);
  uint64_t approxII = largeLatencyBound / minDepDistance +
                      (largeLatencyBound % minDepDistance > 0);

  uint64_t unrollFactor = std::min(
      approxII, maxUnrollFactor.value_or(std::numeric_limits<uint64_t>::max()));

  OpBuilder builder(affineFor);
  affineFor->setAttr("hls.pipeline", builder.getUnitAttr());
  affineFor->setAttr("hls.approxII", builder.getI64IntegerAttr(approxII));

  if (unrollFactor <= 1)
    return success();

  if (loopUnrollUpToFactor(affineFor, unrollFactor).failed())
    return failure();

  return success();
}

std::pair<SmallVector<AffineForOp>, unsigned>
PipelineForLoopSchedule::getDeepestNestedForOps(
    std::pair<SmallVector<AffineForOp>, unsigned> pair) {
  auto roots = pair.first;
  auto rootDepth = pair.second;
  for (auto root : roots) {
    SmallVector<AffineForOp> nestedOps(root.getOps<AffineForOp>());
    if (nestedOps.empty())
      continue;
    auto recur = getDeepestNestedForOps(std::pair(nestedOps, rootDepth + 1));
    if (recur.second > pair.second ||
        (recur.second == pair.second &&
         recur.first.size() > pair.first.size())) {
      pair.second = recur.second;
      pair.first = recur.first;
    }
  }
  return pair;
}

void PipelineForLoopSchedule::runOnOperation() {

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
namespace loopschedule {
std::unique_ptr<mlir::Pass> createPipelineForLoopSchedulePass() {
  return std::make_unique<PipelineForLoopSchedule>();
}
} // namespace loopschedule
} // namespace circt
