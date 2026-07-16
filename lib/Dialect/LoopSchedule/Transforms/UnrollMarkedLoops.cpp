//===- UnrollMarkedLoops.cpp - Unroll Marked Loops Pass ----------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Consumes the `loopschedule.parallel = N` in-place unroll directive on
// affine.for loops (Allo's `s.unroll(axis, factor)` is the producer). N is
// an unroll factor, with 0 meaning "fully unroll" — Allo's encoding for a
// factor-less `s.unroll(axis)`. A factor of 1 is a no-op that still
// consumes the directive.
//
// Anything the pass cannot honor is an error naming the loop and the
// reason: a dropped directive silently produces hardware N times narrower
// than the source asked for, which is indistinguishable from a working
// compile.
//
// Placement invariant: like all unrolling in this flow, this must run BEFORE
// ConstructMemoryDependencies (dependences are name-keyed with iteration
// distances; unrolled clones sharing one name would turn same-iteration lane
// orderings into misread next-initiation constraints) and BEFORE allocation
// (each unrolled access site gets its own port, so banked memories can feed
// the lanes in parallel).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// Note: SCF.h is used by the generated pass base (dependent dialects).
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"

namespace circt {
namespace loopschedule {
#define GEN_PASS_DEF_UNROLLMARKEDLOOPS
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"
} // namespace loopschedule
} // namespace circt

using namespace circt;
using namespace loopschedule;
using namespace mlir;
using namespace mlir::affine;

namespace {

struct UnrollMarkedLoopsPass
    : public circt::loopschedule::impl::UnrollMarkedLoopsBase<UnrollMarkedLoopsPass> {
  void runOnOperation() override;
};

/// The unroll directive this pass owns.
constexpr StringRef kUnrollDirective = "loopschedule.parallel";

} // end anonymous namespace

/// Consume `loop`'s unroll directive and honor it. The directive is removed
/// up front so the remainder-loop clone that loopUnrollByFactor makes for
/// non-divisible trip counts does not re-carry it (it does keep the other
/// attributes, notably `hls.pipeline`).
static LogicalResult unrollMarkedLoop(AffineForOp loop) {
  Attribute attr = loop->getAttr(kUnrollDirective);
  loop->removeAttr(kUnrollDirective);

  auto intAttr = dyn_cast<IntegerAttr>(attr);
  if (!intAttr)
    return loop.emitOpError("has invalid ")
           << kUnrollDirective << " attribute " << attr
           << "; expected an integer unroll factor (0 to fully unroll)";

  int64_t factor = intAttr.getInt();
  if (factor < 0)
    return loop.emitOpError("has invalid ")
           << kUnrollDirective << " factor " << factor
           << "; expected a non-negative integer (0 to fully unroll)";

  // Factor 1 requests no unrolling at all.
  if (factor == 1)
    return success();

  if (factor == 0) {
    // A zero-trip loop contributes nothing: its results are its inits.
    // (loopUnrollFull leaves such loops in place.)
    if (std::optional<uint64_t> trip = getConstantTripCount(loop);
        trip && *trip == 0) {
      loop->replaceAllUsesWith(loop.getInits());
      loop->erase();
      return success();
    }
    if (failed(loopUnrollFull(loop)))
      return loop.emitOpError("marked ")
             << kUnrollDirective
             << " = 0 could not be fully unrolled; full unrolling requires a "
                "constant trip count";
    return success();
  }

  if (failed(loopUnrollByFactor(loop, factor)))
    return loop.emitOpError("could not be unrolled by ")
           << kUnrollDirective << " factor " << factor;
  return success();
}

void UnrollMarkedLoopsPass::runOnOperation() {
  // Collect-then-process: unrolling mutates the tree under our feet
  // (loopUnrollFull erases the loop it promotes), which is not legal under
  // walk(). Post-order collection means inner marked loops unroll before
  // any enclosing marked loop replicates them, so a directive is consumed
  // exactly once.
  SmallVector<AffineForOp> markedLoops;
  getOperation().walk<WalkOrder::PostOrder>([&](AffineForOp loop) {
    if (loop->hasAttr(kUnrollDirective))
      markedLoops.push_back(loop);
  });

  for (AffineForOp loop : markedLoops)
    if (failed(unrollMarkedLoop(loop)))
      return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::loopschedule::createUnrollMarkedLoopsPass() {
  return std::make_unique<UnrollMarkedLoopsPass>();
}
