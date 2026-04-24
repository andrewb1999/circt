//===- LoopScheduleStartTimeAnalysis.cpp - Bindable-op occupancy ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements `LoopScheduleStartTimeAnalysis`: walks a scheduled subtree,
// identifies bindable ops, and records each op's AP-of-intervals occupancy
// plus the concurrency group that forms its shared wall-clock window.
//
// For an op directly inside `loopschedule.at K` of a `loopschedule.frame`
// F, occupancy is `(startTime=K, period=0, count=1)` and group = F.
//
// For an op inside a pipeline P launched at `at K` within frame F, the
// pipeline is inline (same hw.module as F) and its stage offsets fold into
// F's timebase. For stage S, occupancy is
//   (startTime = K + S, period = P.II, count = P.tripCount)
// and group = F — so the pipeline's ops compete in the same concurrency
// group as F's static at-ops, which is what the inline lowering requires
// for correct binding.
//
// For an op directly inside `loopschedule.at K` of a `loopschedule.func_pipeline`,
// occupancy is `(startTime=K, period=func.II, count=1)` (we treat the func
// body as one iteration of a pipeline — downstream callers that issue
// multiple transactions can set count themselves if needed), group = func.
//
// We stop walking at `loopschedule.sequential` bodies (their ops live in
// a submodule) and at `loopschedule.call` (opaque from this analysis's
// perspective).
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/LoopScheduleStartTimeAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"

using namespace circt;
using namespace circt::analysis;
using namespace circt::loopschedule;
using namespace mlir;

/// Per-op attribute set by the scheduler on oplib-linked arith ops.
static constexpr llvm::StringLiteral kOperatorAttr = "loopschedule.operator";

static bool isBindableOp(Operation *op) {
  if (op->hasAttr(kOperatorAttr))
    return true;
  if (isa<LoopScheduleLoadOp, LoopScheduleStoreOp>(op))
    return true;
  if (isa<LoadInterface, StoreInterface>(op))
    return true;
  return false;
}

namespace {
/// Occupancy triple accumulated while climbing an op's ancestor chain.
/// Kept file-local (not the public `Occupancy` struct from the header)
/// because that one is private to the analysis class.
struct Occ {
  unsigned startTime;
  unsigned period;
  unsigned count;
  Operation *group;
};
} // namespace

/// Climbs \p op's ancestor chain, accumulating at-offsets and recording
/// any enclosing pipeline's II + trip count. Terminates at the innermost
/// `Frame` or `FuncPipeline` (the concurrency group). Returns the triple
/// `(startTime, period, count, group)` or nullopt if no recognizable
/// enclosing region is found.
static std::optional<Occ> computeOccupancy(Operation *op) {
  // Climb the ancestor chain. The first `at` we see contributes its offset.
  // Then the `at`'s parent is either a pipeline (fold stage offset + II +
  // trip count into the record) or the concurrency group itself (frame or
  // func_pipeline).
  Operation *cur = op;
  unsigned startTime = 0;
  unsigned period = 0;
  unsigned count = 1;

  bool haveInnerAt = false;
  for (Operation *p = cur->getParentOp(); p; cur = p, p = p->getParentOp()) {
    if (auto at = dyn_cast<LoopScheduleAtOp>(p)) {
      if (!haveInnerAt) {
        // Innermost `at` — its offset is the op's activation time within
        // its scope.
        startTime = static_cast<unsigned>(at.getOffset());
        haveInnerAt = true;
      } else {
        // Outer `at` — its offset adds into the activation time (handles
        // a pipeline-inside-frame where the pipeline launch's at-K adds
        // to the pipeline-stage-S inside).
        startTime += static_cast<unsigned>(at.getOffset());
      }
      continue;
    }
    if (auto pipe = dyn_cast<LoopSchedulePipelineOp>(p)) {
      // This op is inside a pipeline. The pipeline's II is the period and
      // its trip count is the count. We continue climbing to find the
      // enclosing frame (which is the concurrency group — pipelines lower
      // inline).
      period = static_cast<unsigned>(pipe.getII());
      if (auto tc = pipe.getTripCount())
        count = static_cast<unsigned>(*tc);
      continue;
    }
    if (isa<LoopScheduleFrameOp>(p)) {
      return Occ{startTime, period, count, p};
    }
    if (auto funcPipe = dyn_cast<LoopScheduleFuncPipelineOp>(p)) {
      // `func_pipeline` contains `at` stages directly. Its II applies to
      // any op inside unless a nested pipeline's II already claimed the
      // period slot (which would be unusual — nested pipelines inside
      // func_pipeline aren't a supported shape today). Treat the func
      // pipeline as the concurrency group.
      if (period == 0)
        period = static_cast<unsigned>(funcPipe.getII());
      return Occ{startTime, period, count, p};
    }
    if (isa<LoopScheduleSequentialOp>(p)) {
      // Crossing into a sequential body means we're inside a submodule;
      // this analysis doesn't cover ops that belong to that submodule.
      return std::nullopt;
    }
    if (isa<LoopScheduleCallOp>(p)) {
      // Calls are opaque — their body is defined elsewhere.
      return std::nullopt;
    }
    // Any other parent (launch, if, etc.) is just transparent nesting —
    // keep climbing.
  }
  return std::nullopt;
}

LoopScheduleStartTimeAnalysis::LoopScheduleStartTimeAnalysis(Operation *root) {
  root->walk([&](Operation *op) {
    if (!isBindableOp(op))
      return;
    auto occ = computeOccupancy(op);
    if (!occ)
      return;
    occupancy[op] = Occupancy{occ->startTime, occ->period, occ->count,
                                occ->group};
    bindableOps.push_back(op);
  });
}

std::optional<unsigned>
LoopScheduleStartTimeAnalysis::getStartTime(Operation *op) const {
  auto it = occupancy.find(op);
  if (it == occupancy.end())
    return std::nullopt;
  return it->second.startTime;
}

std::optional<unsigned>
LoopScheduleStartTimeAnalysis::getPeriod(Operation *op) const {
  auto it = occupancy.find(op);
  if (it == occupancy.end())
    return std::nullopt;
  return it->second.period;
}

std::optional<unsigned>
LoopScheduleStartTimeAnalysis::getCount(Operation *op) const {
  auto it = occupancy.find(op);
  if (it == occupancy.end())
    return std::nullopt;
  return it->second.count;
}

Operation *LoopScheduleStartTimeAnalysis::getConcurrencyGroup(
    Operation *op) const {
  auto it = occupancy.find(op);
  if (it == occupancy.end())
    return nullptr;
  return it->second.group;
}
