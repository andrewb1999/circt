//===- UnrollSubLoops.cpp - Make pipelined loop bodies loop-free -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pipelined loop's body must be loop-free: the pipeline scheduler assigns
// every op a fixed cycle offset, so a nested loop (a variable op count per
// iteration) cannot be scheduled — it survives to SCFToLoopSchedule as an
// scf.while and dies with `unsupported operation "scf.condition"`. This pass
// fully unrolls constant-trip loops (affine.for and scf.for) nested inside
// `hls.pipeline`-marked loops, innermost-first so trip counts stay constant.
//
// Loops that cannot be statically dissolved are loud errors, not silent
// pass-through: dynamic bounds and scf.while are unschedulable inside a
// static pipeline by construction. While loops are unsupported by design —
// pipeline the while loop itself instead of an ancestor.
//
// Placement invariant: like all unrolling in this flow, this must run BEFORE
// ConstructMemoryDependencies. Dependences are name-keyed with iteration
// distances; body replication would leave clones sharing one name and turn
// same-iteration lane orderings into misread next-initiation constraints.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
namespace loopschedule {
#define GEN_PASS_DEF_UNROLLSUBLOOPS
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"
} // namespace loopschedule
} // namespace circt

using namespace circt;
using namespace loopschedule;
using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct UnrollSubLoopsPass
    : public circt::loopschedule::impl::UnrollSubLoopsBase<UnrollSubLoopsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static bool hasPipelinedAncestor(Operation *op) {
  for (Operation *parent = op->getParentOp();
       parent && !isa<ModuleOp>(parent); parent = parent->getParentOp())
    if (parent->hasAttr("hls.pipeline"))
      return true;
  return false;
}

/// A zero-trip loop contributes nothing: its results are its inits.
static void eraseZeroTripLoop(Operation *op, ValueRange inits) {
  op->replaceAllUsesWith(inits);
  op->erase();
}

/// Fully unroll one loop nested inside a pipelined loop, erasing it (the
/// unroll utilities promote the single remaining iteration). Loops that
/// cannot be statically dissolved get a diagnostic naming why.
static LogicalResult fullyUnroll(Operation *op) {
  if (auto affineFor = dyn_cast<AffineForOp>(op)) {
    std::optional<uint64_t> trip = getConstantTripCount(affineFor);
    if (!trip)
      return affineFor.emitOpError(
          "has a non-constant trip count and cannot be fully unrolled inside "
          "a pipelined loop; pipelined loop bodies must be loop-free");
    if (*trip == 0) {
      eraseZeroTripLoop(affineFor, affineFor.getInits());
      return success();
    }
    return loopUnrollFull(affineFor);
  }

  if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
    std::optional<int64_t> lb = getConstantIntValue(scfFor.getLowerBound());
    std::optional<int64_t> ub = getConstantIntValue(scfFor.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(scfFor.getStep());
    if (!lb || !ub || !step)
      return scfFor.emitOpError(
          "has non-constant bounds and cannot be fully unrolled inside a "
          "pipelined loop; pipelined loop bodies must be loop-free");
    if (*step <= 0)
      return scfFor.emitOpError("expected a positive step");
    int64_t trip =
        *ub <= *lb ? 0 : llvm::divideCeilSigned(*ub - *lb, *step);
    if (trip == 0) {
      eraseZeroTripLoop(scfFor, scfFor.getInitArgs());
      return success();
    }
    return loopUnrollByFactor(scfFor, trip);
  }

  if (isa<scf::WhileOp>(op))
    return op->emitOpError(
        "cannot be statically unrolled inside a pipelined loop; pipeline the "
        "while loop itself instead");

  return success();
}

void UnrollSubLoopsPass::runOnOperation() {
  auto funcOp = getOperation();

  // Outermost pipelined loops of any loop type. Inner pipelined loops are
  // dissolved into their pipelined ancestor like any other sub-loop.
  SmallVector<Operation *> roots;
  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<AffineForOp, scf::ForOp, scf::WhileOp>(op) ||
        !op->hasAttr("hls.pipeline"))
      return;
    if (hasPipelinedAncestor(op))
      return;
    roots.push_back(op);
  });

  for (Operation *root : roots) {
    // Post-order: innermost loops unroll first, so enclosing loops keep
    // constant trip counts and their unrolls replicate straight-line code.
    // Unrolling erases only the processed loop, so the collected outer
    // loops stay valid.
    SmallVector<Operation *> subLoops;
    for (Region &region : root->getRegions())
      region.walk<WalkOrder::PostOrder>([&](Operation *op) {
        if (isa<AffineForOp, scf::ForOp, scf::WhileOp>(op))
          subLoops.push_back(op);
      });
    for (Operation *subLoop : subLoops)
      if (failed(fullyUnroll(subLoop)))
        return signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::loopschedule::createUnrollSubLoopsPass() {
  return std::make_unique<UnrollSubLoopsPass>();
}
