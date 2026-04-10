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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

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

/// A constant-coefficient linear combination of the nest's IVs:
///   value == offset + sum_k strides[k] * iv_k
/// Used to recognize address chains (e.g., the output of FlattenMemRefs)
/// that can be promoted to a single iter-arg incremented by the odometer.
struct LinearAddr {
  int64_t offset;
  SmallVector<int64_t> strides; // length == nest.size()
};

/// Try to express `v` as a constant-coefficient linear combination of the
/// nest's after-region IVs. Recognizes the canonical patterns emitted by
/// circt's FlattenMemRefs (constants, IV references through arith.index_cast,
/// addi/subi, and muli/shli with one constant operand).
static std::optional<LinearAddr> matchLinear(Value v,
                                             ArrayRef<NestLevel> nest) {
  LinearAddr out;
  out.offset = 0;
  out.strides.assign(nest.size(), 0);

  // Constant.
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c))) {
    out.offset = c.getSExtValue();
    return out;
  }

  // Direct IV reference.
  for (unsigned k = 0, e = nest.size(); k < e; ++k) {
    if (v == nest[k].ivAfter) {
      out.strides[k] = 1;
      return out;
    }
  }

  // Transparent index_cast (the bridge between integer-typed loop IVs and
  // index-typed memref indices).
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return matchLinear(cast.getIn(), nest);
  if (auto cast = v.getDefiningOp<arith::IndexCastUIOp>())
    return matchLinear(cast.getIn(), nest);

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    auto a = matchLinear(add.getLhs(), nest);
    auto b = matchLinear(add.getRhs(), nest);
    if (!a || !b)
      return std::nullopt;
    out.offset = a->offset + b->offset;
    for (unsigned k = 0, e = nest.size(); k < e; ++k)
      out.strides[k] = a->strides[k] + b->strides[k];
    return out;
  }

  if (auto sub = v.getDefiningOp<arith::SubIOp>()) {
    auto a = matchLinear(sub.getLhs(), nest);
    auto b = matchLinear(sub.getRhs(), nest);
    if (!a || !b)
      return std::nullopt;
    out.offset = a->offset - b->offset;
    for (unsigned k = 0, e = nest.size(); k < e; ++k)
      out.strides[k] = a->strides[k] - b->strides[k];
    return out;
  }

  if (auto mul = v.getDefiningOp<arith::MulIOp>()) {
    APInt cm;
    Value other;
    if (matchPattern(mul.getLhs(), m_ConstantInt(&cm)))
      other = mul.getRhs();
    else if (matchPattern(mul.getRhs(), m_ConstantInt(&cm)))
      other = mul.getLhs();
    else
      return std::nullopt;
    auto a = matchLinear(other, nest);
    if (!a)
      return std::nullopt;
    int64_t k = cm.getSExtValue();
    out.offset = a->offset * k;
    for (unsigned j = 0, e = nest.size(); j < e; ++j)
      out.strides[j] = a->strides[j] * k;
    return out;
  }

  if (auto shl = v.getDefiningOp<arith::ShLIOp>()) {
    APInt cm;
    if (!matchPattern(shl.getRhs(), m_ConstantInt(&cm)))
      return std::nullopt;
    auto a = matchLinear(shl.getLhs(), nest);
    if (!a)
      return std::nullopt;
    int64_t k = int64_t(1) << cm.getSExtValue();
    out.offset = a->offset * k;
    for (unsigned j = 0, e = nest.size(); j < e; ++j)
      out.strides[j] = a->strides[j] * k;
    return out;
  }

  return std::nullopt;
}

/// A memory address inside the innermost body that we plan to promote to a
/// per-iteration iter-arg of the flattened scf.while.
struct AddrCandidate {
  Value root;       // Original SSA value defined inside the inner body.
  Type type;        // root.getType() — the iter-arg's type.
  LinearAddr linear;
};

/// Find values used as memory-op indices inside `inner.getAfterBody()` that
/// are constant-coefficient linear combinations of the nest's IVs. Returns
/// one entry per distinct root value (loads/stores sharing an SSA address
/// value share an iter-arg automatically).
static SmallVector<AddrCandidate>
collectAddrCandidates(WhileOp inner, ArrayRef<NestLevel> nest) {
  SmallVector<AddrCandidate> result;
  DenseMap<Value, unsigned> seen;

  auto consider = [&](Value idx) {
    if (seen.contains(idx))
      return;
    auto lin = matchLinear(idx, nest);
    if (!lin)
      return;
    // Only promote linear forms whose computation actually does work each
    // iteration: at least one stride |c_k| > 1 (a real scaled term, e.g.
    // emitted by FlattenMemRefs) or a nonzero constant offset. Skip pure
    // constants and trivial unit-coefficient sums of IVs (like a bare
    // `arith.index_cast %iv` or `%a + %b`), which the existing per-level
    // iter-args already track.
    bool anyStride = false;
    bool genuinelyScaled = false;
    for (int64_t s : lin->strides) {
      if (s != 0)
        anyStride = true;
      if (s > 1 || s < -1)
        genuinelyScaled = true;
    }
    if (!anyStride)
      return;
    if (!genuinelyScaled && lin->offset == 0)
      return;
    seen[idx] = result.size();
    result.push_back({idx, idx.getType(), *lin});
  };

  inner.getAfterBody()->walk([&](Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      for (Value idx : load.getIndices())
        consider(idx);
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      for (Value idx : store.getIndices())
        consider(idx);
    }
  });

  return result;
}

/// Emit the flattened scf.while for a collected perfect nest.
static void emitFlattenedNest(OpBuilder &builder,
                              ArrayRef<NestLevel> nest) {
  WhileOp outer = nest.front().loop;
  WhileOp inner = nest.back().loop;
  unsigned N = nest.size();
  MLIRContext *ctx = builder.getContext();
  Location loc = outer.getLoc();

  // Identify any memory-op address chains in the inner body that are
  // constant-coefficient linear combinations of the loop IVs. Each such
  // value will be promoted to a dedicated iter-arg of the flattened while,
  // updated incrementally from the odometer's done signals (eliminating
  // the per-iteration mul/add chain that FlattenMemRefs leaves behind).
  SmallVector<AddrCandidate> candidates = collectAddrCandidates(inner, nest);
  unsigned M = candidates.size();

  builder.setInsertionPoint(outer);

  // --- Inits: one per level, plus one per address candidate. ---
  SmallVector<Value> inits;
  SmallVector<Type> ivTypes;
  inits.reserve(N + M);
  ivTypes.reserve(N + M);
  for (NestLevel lvl : nest) {
    auto cst = builder.create<arith::ConstantOp>(
        loc, lvl.ivType, IntegerAttr::get(lvl.ivType, lvl.lb));
    inits.push_back(cst);
    ivTypes.push_back(lvl.ivType);
  }
  for (const AddrCandidate &cand : candidates) {
    int64_t initVal = cand.linear.offset;
    for (unsigned k = 0; k < N; ++k)
      initVal += cand.linear.strides[k] * nest[k].lb.getSExtValue();
    auto cst = builder.create<arith::ConstantOp>(
        loc, cand.type, IntegerAttr::get(cand.type, initVal));
    inits.push_back(cst);
    ivTypes.push_back(cand.type);
  }

  auto flat = builder.create<WhileOp>(loc, ivTypes, inits);

  // Preserve the pipeline attribute from the innermost loop so that
  // downstream scheduling passes (SCFToLoopSchedule) still recognize
  // the flattened loop as pipelined.
  if (auto pipeAttr = inner->getAttr("hls.pipeline"))
    flat->setAttr("hls.pipeline", pipeAttr);

  // --- Before region. ---
  {
    Block *beforeBlock =
        builder.createBlock(&flat.getBefore(), {}, ivTypes,
                            SmallVector<Location>(N + M, loc));
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
                            SmallVector<Location>(N + M, loc));
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

    // For each promoted address candidate, replace uses of the cloned arith
    // chain with the new iter-arg, then erase the now-dead chain (no
    // canonicalization required to see the optimization). We do this
    // *before* emitting the odometer update so the only post-clone uses
    // inside the after region are inside loads and stores (or other body
    // code that already had the address Value).
    for (unsigned c = 0; c < M; ++c) {
      Value clonedRoot = mapping.lookup(candidates[c].root);
      if (!clonedRoot)
        continue;
      // Collect the transitive defining ops of the cloned root that live
      // inside the new after region, in def-before-use order.
      SmallVector<Operation *> chain;
      llvm::SmallPtrSet<Operation *, 8> visited;
      std::function<void(Value)> collect = [&](Value v) {
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != afterBlock)
          return;
        if (!visited.insert(def).second)
          return;
        for (Value o : def->getOperands())
          collect(o);
        chain.push_back(def);
      };
      collect(clonedRoot);
      clonedRoot.replaceAllUsesWith(afterBlock->getArgument(N + c));
      // Erase use-before-def so each erase only touches an op with no
      // remaining users.
      for (auto it = chain.rbegin(), end = chain.rend(); it != end; ++it) {
        if ((*it)->use_empty())
          (*it)->erase();
      }
    }

    // --- Odometer update. ---
    // Each promoted address has a running value `addrAccum[c]` that starts
    // from the after-arg and accumulates per-level deltas as we walk from
    // the innermost level to the outermost. The increment for the innermost
    // level is unconditional (`stride[N-1] * step[N-1]`); for outer levels
    // it is gated on the existing odometer's `doneChild` signal — i.e.,
    // "the level immediately below me wrapped on this iteration".
    SmallVector<Value> ivOut(N);
    SmallVector<Value> addrAccum(M);
    for (unsigned c = 0; c < M; ++c)
      addrAccum[c] = afterBlock->getArgument(N + c);

    // Innermost: unconditional contribution `stride[N-1] * step[N-1]`.
    {
      const NestLevel &innerLvl = nest[N - 1];
      int64_t stepInner = innerLvl.step.getSExtValue();
      for (unsigned c = 0; c < M; ++c) {
        int64_t base = candidates[c].linear.strides[N - 1] * stepInner;
        if (base == 0)
          continue;
        Type t = candidates[c].type;
        auto baseCst = builder.create<arith::ConstantOp>(
            loc, t, IntegerAttr::get(t, base));
        addrAccum[c] =
            builder.create<arith::AddIOp>(loc, addrAccum[c], baseCst);
      }
    }

    Value doneChild; // done_{k+1}; null before the first iteration.

    for (int k = int(N) - 1; k >= 0; --k) {
      const NestLevel &lvl = nest[k];

      // Per-level address contribution for level k (k < N-1): gated on
      // `doneChild`, which currently holds "level k+1 wrapped". The amount
      // is `stride[k]*step[k] - stride[k+1]*(ub_norm[k+1]-lb[k+1])`: the
      // first term is what level k contributes when it advances by step,
      // the second term reverses the wrap-around of level k+1 that just
      // reset back to its lb.
      if (k < int(N) - 1) {
        const NestLevel &child = nest[k + 1];
        int64_t childRange =
            (child.ubNormalized - child.lb).getSExtValue();
        int64_t stepK = lvl.step.getSExtValue();
        for (unsigned c = 0; c < M; ++c) {
          int64_t addendVal =
              candidates[c].linear.strides[k] * stepK -
              candidates[c].linear.strides[k + 1] * childRange;
          if (addendVal == 0)
            continue;
          Type t = candidates[c].type;
          auto addendCst = builder.create<arith::ConstantOp>(
              loc, t, IntegerAttr::get(t, addendVal));
          auto zeroCst = builder.create<arith::ConstantOp>(
              loc, t, IntegerAttr::get(t, 0));
          Value sel = builder.create<arith::SelectOp>(loc, doneChild,
                                                      addendCst, zeroCst);
          addrAccum[c] =
              builder.create<arith::AddIOp>(loc, addrAccum[c], sel);
        }
      }

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

    SmallVector<Value> yieldOperands(ivOut.begin(), ivOut.end());
    yieldOperands.append(addrAccum.begin(), addrAccum.end());
    builder.create<YieldOp>(loc, yieldOperands);
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
