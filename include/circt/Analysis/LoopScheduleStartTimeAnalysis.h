//===- LoopScheduleStartTimeAnalysis.h - Bindable-op occupancy --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Determines the cycle-accurate occupancy of every *bindable* op in a
// scheduled LoopSchedule IR — the input the binding framework consumes.
//
// For each bindable op the analysis records:
//
//   - `startTime`: the cycle at which the op first activates, measured in
//     the timebase of its `concurrencyGroup`.
//   - `period`: cycles between successive activations (0 for static ops,
//     II for ops inside a pipeline).
//   - `count`: number of activations (1 for static ops, tripCount for ops
//     inside a pipeline).
//   - `concurrencyGroup`: the innermost *inline* enclosing op that defines
//     a shared wall-clock window (a `loopschedule.frame` or a
//     `loopschedule.func_pipeline`). Ops inside a pipeline that is itself
//     inside a frame are attributed to the frame — pipelines lower inline,
//     so their stages run in the same wall-clock window as the frame's
//     static at-ops. Callers may conflict-check ops from different
//     concurrency groups cheaply (they can never conflict).
//
// A *bindable* op is one that competes for a limited hardware resource:
//   - any op carrying a `loopschedule.operator = @symbol` attribute
//     (oplib-linked arith ops), and
//   - any op implementing the `LoadInterface` / `StoreInterface` or that
//     is a `LoopScheduleLoadOp` / `LoopScheduleStoreOp`.
//
// The analysis is read-only and intentionally narrow: it does not assign
// instance ids, does not translate to a `binding::BindingProblem`, and
// does not reason about inter-hw.module ordering. Callers run one of
// these per scheduled container and populate their own problem.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_LOOPSCHEDULE_START_TIME_ANALYSIS_H
#define CIRCT_ANALYSIS_LOOPSCHEDULE_START_TIME_ANALYSIS_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace circt {
namespace analysis {

struct LoopScheduleStartTimeAnalysis {
  /// Walk \p root, identify every bindable op, and cache its occupancy and
  /// concurrency group. \p root is typically a `loopschedule.func_sequential`
  /// / `loopschedule.func_pipeline`, but any op is accepted — the walk
  /// descends the entire subtree, stopping at submodule boundaries that
  /// don't share an hw.module with the root (`loopschedule.sequential`,
  /// `loopschedule.call`).
  explicit LoopScheduleStartTimeAnalysis(Operation *root);

  /// True if \p op was classified as bindable and has recorded occupancy.
  bool isBindable(Operation *op) const { return occupancy.contains(op); }

  /// The cycle offset, in \p op's concurrency-group timebase, at which
  /// \p op first activates. For a static at-K op in a frame: K. For a
  /// pipeline stage-S op at pipeline-launched-at-K inside that frame:
  /// K + S (flattened into the frame's timebase).
  std::optional<unsigned> getStartTime(Operation *op) const;

  /// Cycles between successive activations. 0 for static ops; the
  /// enclosing pipeline's II for ops inside a pipeline.
  std::optional<unsigned> getPeriod(Operation *op) const;

  /// Number of activations. 1 for static ops; the enclosing pipeline's
  /// trip count for ops inside a pipeline (defaults to 1 if the pipeline
  /// has no declared trip_count attribute).
  std::optional<unsigned> getCount(Operation *op) const;

  /// The innermost inline region forming \p op's shared wall-clock window
  /// (a `loopschedule.frame` or `loopschedule.func_pipeline`). Null for
  /// ops that are not bindable or that lack a recognizable enclosing
  /// inline region.
  Operation *getConcurrencyGroup(Operation *op) const;

  /// Every bindable op recorded by the analysis, in walk order.
  llvm::ArrayRef<Operation *> getBindableOps() const { return bindableOps; }

private:
  struct Occupancy {
    unsigned startTime;
    unsigned period;
    unsigned count;
    Operation *group;
  };
  llvm::DenseMap<Operation *, Occupancy> occupancy;
  llvm::SmallVector<Operation *> bindableOps;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_LOOPSCHEDULE_START_TIME_ANALYSIS_H
