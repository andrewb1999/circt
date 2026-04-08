//===- SCFWhileLoopFlattening.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flattens a perfect nest of scf.while loops (each structurally a for-loop
// per getSCFWhileConstantTripCount) into a single scf.while that tracks
// each level's IV in its own iter-arg and updates them via an odometer
// (compares + selects) instead of div/mod.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/SCFWhileTripCountAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
#define GEN_PASS_DEF_SCFWHILELOOPFLATTENING
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::scf;

namespace {

/// Per-level information derived from a canonical-form scf.while.
struct NestLevel {
  WhileOp loop;
  unsigned ivIdx;          // Iter-arg / result index of the IV.
  APInt lb;                // Constant lower bound (initial value).
  APInt ubNormalized;      // Exclusive upper bound after predicate
                           // normalization (sle→slt+1, ule→ult+1).
  APInt step;              // Constant positive step.
  arith::CmpIPredicate origPred;
  arith::AddIOp updateOp;  // `addi iv_after, step` feeding the yield.
  BlockArgument ivAfter;   // The after-region IV block argument.
  IntegerType ivType;
};

/// Returns true if `pred` is one of the supported comparison kinds.
static bool isSupportedPredicate(arith::CmpIPredicate p) {
  switch (p) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
  case arith::CmpIPredicate::ne:
    return true;
  default:
    return false;
  }
}

static bool isSignedPredicate(arith::CmpIPredicate p) {
  return p == arith::CmpIPredicate::slt || p == arith::CmpIPredicate::sle;
}

/// Wrap predicate corresponding to each original predicate. `iv_next` reaches
/// its bound when this predicate is true against the normalized exclusive
/// upper bound.
static arith::CmpIPredicate wrapPredicate(arith::CmpIPredicate p) {
  switch (p) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::eq;
  default:
    llvm_unreachable("unsupported predicate");
  }
}

/// Re-run the canonical-form matcher to recover lb/ub/step/pred for a single
/// scf.while. Mirrors the structural checks in SCFWhileTripCountAnalysis.cpp
/// but also returns the extra fields the flattening pass needs.
static std::optional<NestLevel> matchLevel(WhileOp whileOp) {
  // Canonical form implies a single iter-arg. The analysis also requires
  // this, but the matcher here walks the structure directly so we can grab
  // the addi op reference.
  if (whileOp.getNumResults() != 1)
    return std::nullopt;

  ConditionOp condOp = whileOp.getConditionOp();
  auto cmp = condOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp || !isSupportedPredicate(cmp.getPredicate()))
    return std::nullopt;

  auto ivBefore = dyn_cast<BlockArgument>(cmp.getLhs());
  if (!ivBefore || ivBefore.getOwner() != whileOp.getBeforeBody())
    return std::nullopt;
  unsigned idx = ivBefore.getArgNumber();

  APInt ubConst;
  if (!matchPattern(cmp.getRhs(), m_ConstantInt(&ubConst)))
    return std::nullopt;

  APInt lb;
  if (!matchPattern(whileOp.getInits()[idx], m_ConstantInt(&lb)))
    return std::nullopt;

  auto ivType = dyn_cast<IntegerType>(ivBefore.getType());
  if (!ivType)
    return std::nullopt;
  unsigned bw = ivType.getWidth();
  if (lb.getBitWidth() != bw || ubConst.getBitWidth() != bw)
    return std::nullopt;

  // Find the IV in the after region: the condition forwards before-args
  // positionally, so after_args[j] == condition.args[j]. We look for the j
  // where the forwarded value is the IV before-arg.
  Block *afterBlock = whileOp.getAfterBody();
  BlockArgument ivAfter = nullptr;
  for (auto [j, fwd] : llvm::enumerate(condOp.getArgs())) {
    if (fwd == ivBefore) {
      ivAfter = afterBlock->getArgument(j);
      break;
    }
  }
  if (!ivAfter)
    return std::nullopt;

  YieldOp yieldOp = whileOp.getYieldOp();
  if (idx >= yieldOp.getNumOperands())
    return std::nullopt;
  auto addOp = yieldOp.getOperand(idx).getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;

  APInt step;
  Value addLhs = addOp.getLhs();
  Value addRhs = addOp.getRhs();
  if (addLhs == ivAfter) {
    if (!matchPattern(addRhs, m_ConstantInt(&step)))
      return std::nullopt;
  } else if (addRhs == ivAfter) {
    if (!matchPattern(addLhs, m_ConstantInt(&step)))
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  if (!step.isStrictlyPositive())
    return std::nullopt;

  // Normalize inclusive predicates to exclusive bounds.
  APInt ub = ubConst;
  bool isSigned = isSignedPredicate(cmp.getPredicate());
  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    break;
  case arith::CmpIPredicate::sle: {
    bool ov = false;
    ub = ub.sadd_ov(APInt(bw, 1), ov);
    if (ov)
      return std::nullopt;
    break;
  }
  case arith::CmpIPredicate::ule: {
    bool ov = false;
    ub = ub.uadd_ov(APInt(bw, 1), ov);
    if (ov)
      return std::nullopt;
    break;
  }
  case arith::CmpIPredicate::ne: {
    // Require lb < ub so the loop actually walks toward the bound.
    bool lbLtUb = isSigned ? lb.slt(ub) : lb.ult(ub);
    if (!lbLtUb)
      return std::nullopt;
    break;
  }
  default:
    return std::nullopt;
  }

  return NestLevel{whileOp, idx,  lb,     ub,     step,
                   cmp.getPredicate(), addOp, ivAfter, ivType};
}

/// Walk down from `outer`, collecting a maximal chain of nested canonical
/// scf.while loops whose after regions have the shape
/// `{addi, inner while, yield}`. Returns the chain (≥2 elements) or
/// std::nullopt if `outer` does not anchor a flattenable perfect nest.
static std::optional<SmallVector<NestLevel>>
collectPerfectNest(WhileOp outer) {
  SmallVector<NestLevel> nest;

  WhileOp current = outer;
  while (true) {
    auto level = matchLevel(current);
    if (!level)
      return std::nullopt;

    // Require a non-zero constant trip count so the flattened loop isn't
    // degenerate. We call the public analysis here for consistency with
    // the rest of the codebase.
    auto tc = circt::analysis::getSCFWhileConstantTripCount(current);
    if (!tc || tc->isZero())
      return std::nullopt;

    nest.push_back(*level);

    // Look for a nested scf.while inside current.getAfterBody(). For an
    // intermediate level the after region must contain exactly
    // {addi, inner while, yield}. For the innermost level we stop.
    Block *afterBlock = current.getAfterBody();
    WhileOp nextInner = nullptr;
    unsigned opCount = 0;
    for (Operation &op : afterBlock->without_terminator()) {
      opCount++;
      if (auto w = dyn_cast<WhileOp>(op)) {
        if (nextInner) // more than one while -> not perfect
          return std::nullopt;
        nextInner = w;
      }
    }
    if (!nextInner)
      break; // innermost reached

    // Intermediate level: exactly {addi, inner while} besides the yield.
    if (opCount != 2)
      return std::nullopt;
    // The addi must be our level's update op.
    // (opCount == 2 and one of them is the inner while, so the other is
    // something; verify it is exactly `level->updateOp`.)
    bool sawAddi = false;
    for (Operation &op : afterBlock->without_terminator()) {
      if (&op == nextInner.getOperation())
        continue;
      if (&op == level->updateOp.getOperation()) {
        sawAddi = true;
        continue;
      }
      return std::nullopt;
    }
    if (!sawAddi)
      return std::nullopt;

    // The inner loop's inits must not depend on the outer IV (the
    // canonical-form matcher already requires them to be constants, but
    // double check their defining ops lie outside the nest.) `matchLevel`
    // will verify this when we descend.

    // Also ensure the inner loop's result does not feed the outer yield
    // IV operand — that is already implied because the outer yield at
    // ivIdx is the `addi` result, not the while result.

    current = nextInner;
  }

  if (nest.size() < 2)
    return std::nullopt;
  return nest;
}

/// Emit the flattened scf.while for a collected perfect nest.
static void emitFlattenedNest(OpBuilder &builder,
                              ArrayRef<NestLevel> nest) {
  WhileOp outer = nest.front().loop;
  WhileOp inner = nest.back().loop;
  unsigned N = nest.size();
  MLIRContext *ctx = builder.getContext();
  Location loc = outer.getLoc();

  builder.setInsertionPoint(outer);

  // --- Inits: one per level. ---
  SmallVector<Value> inits;
  SmallVector<Type> ivTypes;
  inits.reserve(N);
  ivTypes.reserve(N);
  for (NestLevel lvl : nest) {
    auto cst = builder.create<arith::ConstantOp>(
        loc, lvl.ivType, IntegerAttr::get(lvl.ivType, lvl.lb));
    inits.push_back(cst);
    ivTypes.push_back(lvl.ivType);
  }

  auto flat = builder.create<WhileOp>(loc, ivTypes, inits);

  // --- Before region. ---
  {
    Block *beforeBlock =
        builder.createBlock(&flat.getBefore(), {}, ivTypes,
                            SmallVector<Location>(N, loc));
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(beforeBlock);

    // Rebuild the outermost condition (cmp) on the flattened iv_0 using
    // the ORIGINAL (pre-normalization) ub constant and predicate.
    const NestLevel &outerLvl = nest.front();
    auto ubAttr = IntegerAttr::get(
        outerLvl.ivType,
        // Re-derive the original rhs: for slt/ult/ne it's ubNormalized; for
        // sle/ule the normalization added 1, so subtract it to recover the
        // original rhs.
        (outerLvl.origPred == arith::CmpIPredicate::sle ||
         outerLvl.origPred == arith::CmpIPredicate::ule)
            ? outerLvl.ubNormalized - APInt(outerLvl.ivType.getWidth(), 1)
            : outerLvl.ubNormalized);
    auto ubCst = builder.create<arith::ConstantOp>(loc, outerLvl.ivType, ubAttr);
    auto cond = builder.create<arith::CmpIOp>(
        loc, outerLvl.origPred, beforeBlock->getArgument(0), ubCst);

    builder.create<ConditionOp>(loc, cond,
                                ValueRange(beforeBlock->getArguments()));
  }

  // --- After region. ---
  {
    Block *afterBlock =
        builder.createBlock(&flat.getAfter(), {}, ivTypes,
                            SmallVector<Location>(N, loc));
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(afterBlock);

    // Map each level's after-region IV block arg to the corresponding
    // flattened block arg.
    IRMapping mapping;
    for (unsigned k = 0; k < N; ++k)
      mapping.map(nest[k].ivAfter, afterBlock->getArgument(k));

    // Clone the innermost body, skipping the IV update addi and the yield.
    Block *innerAfter = inner.getAfterBody();
    arith::AddIOp innerUpdateOp = nest.back().updateOp;
  Operation *innerUpdate = innerUpdateOp.getOperation();
    for (Operation &op : innerAfter->without_terminator()) {
      if (&op == innerUpdate)
        continue;
      builder.clone(op, mapping);
    }

    // --- Odometer update. ---
    SmallVector<Value> ivOut(N);
    Value doneChild; // done_{k+1}; null before the first iteration.

    for (int k = int(N) - 1; k >= 0; --k) {
      const NestLevel &lvl = nest[k];
      Value curIv = afterBlock->getArgument(k);

      auto stepCst = builder.create<arith::ConstantOp>(
          loc, lvl.ivType, IntegerAttr::get(lvl.ivType, lvl.step));

      // iv_tmp = iv + step
      Value ivTmp = builder.create<arith::AddIOp>(loc, curIv, stepCst);

      // For the innermost level, advance unconditionally. For middle and
      // outermost levels, advance only when the child wrapped.
      Value ivNext;
      if (k == int(N) - 1) {
        ivNext = ivTmp;
      } else {
        ivNext = builder.create<arith::SelectOp>(loc, doneChild, ivTmp, curIv);
      }

      if (k == 0) {
        // Outermost: never reset. iv_out = iv_next (which already only
        // advances when done_1 fires).
        ivOut[0] = ivNext;
        break;
      }

      // done_k = cmpi wrapPred(iv_next, ubNormalized)
      auto ubCst = builder.create<arith::ConstantOp>(
          loc, lvl.ivType, IntegerAttr::get(lvl.ivType, lvl.ubNormalized));
      Value reached = builder.create<arith::CmpIOp>(
          loc, wrapPredicate(lvl.origPred), ivNext, ubCst);

      Value doneHere;
      if (k == int(N) - 1) {
        doneHere = reached;
      } else {
        doneHere = builder.create<arith::AndIOp>(loc, doneChild, reached);
      }

      // iv_out = select(done_k, lb, iv_next)
      auto lbCst = builder.create<arith::ConstantOp>(
          loc, lvl.ivType, IntegerAttr::get(lvl.ivType, lvl.lb));
      ivOut[k] = builder.create<arith::SelectOp>(loc, doneHere, lbCst, ivNext);

      doneChild = doneHere;
    }

    builder.create<YieldOp>(loc, ivOut);
  }

  // Replace the outer while's single result with the flat loop's result[0].
  // (All other flat results are unused.)
  outer.getResult(0).replaceAllUsesWith(flat.getResult(0));

  // Erase nest from outer down; each inner sits inside its parent's after
  // region, so erasing the outer takes the whole chain with it.
  outer.erase();
  (void)ctx;
}

struct SCFWhileLoopFlatteningPass
    : public circt::impl::SCFWhileLoopFlatteningBase<
          SCFWhileLoopFlatteningPass> {
  using SCFWhileLoopFlatteningBase<
      SCFWhileLoopFlatteningPass>::SCFWhileLoopFlatteningBase;
  void runOnOperation() override;
};

} // namespace

void SCFWhileLoopFlatteningPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Collect candidate outermost scf.while ops: those whose parent is not a
  // perfectly-nested scf.while after region (i.e., whose enclosing while,
  // if any, would not match as the outer of a flattenable nest with this
  // loop as its child). Easier: walk all scf.while ops and, for each one,
  // try to collect a perfect nest rooted there; process in source order and
  // track which ops have already been consumed.
  SmallVector<WhileOp> roots;
  func.walk<WalkOrder::PreOrder>([&](WhileOp op) { roots.push_back(op); });

  DenseSet<Operation *> consumed;
  OpBuilder builder(&getContext());
  for (WhileOp op : roots) {
    if (consumed.contains(op.getOperation()))
      continue;
    auto nest = collectPerfectNest(op);
    if (!nest)
      continue;
    for (NestLevel &lvl : *nest)
      consumed.insert(lvl.loop.getOperation());
    emitFlattenedNest(builder, *nest);
  }
}

namespace circt {
std::unique_ptr<mlir::Pass> createSCFWhileLoopFlatteningPass() {
  return std::make_unique<SCFWhileLoopFlatteningPass>();
}
} // namespace circt
