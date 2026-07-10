//===- SCFWhileLoopFlattening.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flattens a nest of scf.while loops (each structurally a for-loop per
// getSCFWhileConstantTripCount) into a single scf.while that tracks each
// level's IV in its own iter-arg and updates them via an odometer
// (compares + selects) instead of div/mod.
//
// Besides perfect nests, ALMOST-perfect nests are supported (matching what
// Vitis HLS's pipeline-driven auto-flattening accepts): a level's after
// region may additionally contain
//   - PRE-ops: pure ops before the inner loop (e.g. an accumulator's zero
//     constant, invariant address math). These are recomputed every
//     flattened iteration, which is safe because they are pure.
//   - CARRIED CHAINS: the inner loops may carry extra iter-args (e.g. a
//     reduction accumulator) initialized at some origin boundary from a
//     constant, a nest-invariant value, or a LOAD-SHAPED pre-op (an
//     accumulator resuming from an indexed buffer, sum[p]); threaded
//     unchanged through intermediate levels, and updated only in the
//     innermost body. In the flattened loop each chain becomes one
//     iter-arg that RESETS to its init whenever the levels below its
//     origin wrap (via arith.select, or a predicated re-load with the
//     next pass's IVs for load-initialized chains).
//   - POST-ops: pure ops and memref.stores after the inner loop (e.g. the
//     reduction's store to memory). These become an scf.if predicated on
//     "the levels below just wrapped", i.e. they execute exactly on the
//     iterations where the original epilogue ran. Downstream scheduling
//     (ifOpConversion) turns the scf.if into predicated operations.
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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
/// but also returns the extra fields the flattening pass needs. Loops may
/// carry extra (non-IV) iter-args; those are resolved into carried chains by
/// collectNest, which requires the condition to forward the before-args
/// identically (so init/yield/result/after-arg indices all coincide).
static std::optional<NestLevel> matchLevel(WhileOp whileOp) {
  // Identity forwarding: condition args are exactly the before-args, in
  // order. This makes iter-arg index == after-arg index == result index,
  // which the carried-chain threading below relies on.
  ConditionOp condOp = whileOp.getConditionOp();
  if (condOp.getArgs().size() != whileOp.getBeforeBody()->getNumArguments())
    return std::nullopt;
  for (auto [j, fwd] : llvm::enumerate(condOp.getArgs())) {
    auto arg = dyn_cast<BlockArgument>(fwd);
    if (!arg || arg.getOwner() != whileOp.getBeforeBody() ||
        arg.getArgNumber() != j)
      return std::nullopt;
  }
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

/// A scalar value carried through the nest's inner levels: initialized at an
/// origin boundary (or above the nest entirely), threaded unchanged through
/// intermediate levels' iter-args, updated only in the innermost body, and
/// consumed either by post-ops at the origin boundary or (for origin == -1)
/// by uses of the outermost loop's result.
struct CarriedChain {
  /// Boundary index: the chain is initialized in level originLevel's after
  /// region, right before level originLevel+1. -1 means the init lives
  /// above the whole nest and the outermost loop carries (and returns) it.
  int originLevel;
  /// Init value: an arith.constant, a value defined above the nest, or the
  /// result of `initLoad` (a load-shaped pre-op at the origin boundary).
  Value init;
  /// Non-null when the init is produced by a load-shaped pre-op (e.g. a
  /// reduction accumulating into an indexed buffer: acc starts from
  /// sum[p]). The flattened loop re-executes it once per pass: an
  /// unpredicated clone before the loop for the first pass, and a clone
  /// inside the wrap-predicated scf.if (with next-pass IVs) for the rest.
  Operation *initLoad = nullptr;
  /// Iter-arg index of this chain at each level it threads through
  /// (levels originLevel+1 .. N-1); index into `argIdx` is the level.
  SmallVector<int> argIdx; // size N; -1 where the chain is absent.
  /// After-region block argument at the innermost level (mapped into the
  /// flattened body).
  BlockArgument innerAfterArg;
  /// Value the innermost body yields for this chain (the updated value).
  Value innerYieldVal;
  /// The loop result consumed by the chain's users: result of
  /// nest[originLevel+1] (or of nest[0] when originLevel == -1).
  Value originResult;
};

/// A matched (almost-)perfect nest.
struct NestInfo {
  SmallVector<NestLevel> levels;
  SmallVector<CarriedChain> chains;
  /// Pure ops preceding the inner while in level k's after region.
  SmallVector<SmallVector<Operation *>> preOps;  // size N (innermost empty)
  /// Epilogue ops following the inner while in level k's after region
  /// (pure ops and memref.stores, in program order).
  SmallVector<SmallVector<Operation *>> postOps; // size N (innermost empty)
};

/// True if `v` is safe to reference from the flattened loop's init AND from
/// inside its body: a value defined above `outer` (including constants that
/// were CSE'd to function scope).
static bool isDefinedAboveNest(Value v, WhileOp outer) {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return !outer->isAncestor(arg.getOwner()->getParentOp());
  Operation *def = v.getDefiningOp();
  return def && !outer->isAncestor(def);
}

/// True if `v` is produced by a ConstantLike op (rematerializable anywhere;
/// covers integer and float accumulator inits).
static bool isConstantValue(Value v) {
  Operation *def = v.getDefiningOp();
  return def && def->hasTrait<OpTrait::ConstantLike>();
}

/// Walk down from `outer`, collecting a maximal chain of nested canonical
/// scf.while loops, allowing pure pre-ops, carried iter-arg chains, and
/// store/pure post-ops at each boundary (see file header). Returns the
/// matched nest (≥2 levels) or std::nullopt.
static std::optional<NestInfo> collectNest(WhileOp outer) {
  NestInfo info;
  SmallVector<NestLevel> &nest = info.levels;

  WhileOp current = outer;
  while (true) {
    auto level = matchLevel(current);
    if (!level)
      return std::nullopt;

    // Require a non-zero constant trip count so the flattened loop isn't
    // degenerate. (Computed locally: the public analysis only accepts
    // single-iter-arg loops, and levels here may carry chains.)
    {
      APInt range = level->ubNormalized - level->lb;
      if (!range.isStrictlyPositive())
        return std::nullopt;
      if (level->origPred == arith::CmpIPredicate::ne &&
          !range.urem(level->step).isZero())
        return std::nullopt; // `ne` bound never hit exactly -> not a for.
    }

    nest.push_back(*level);
    info.preOps.emplace_back();
    info.postOps.emplace_back();

    // Find the (unique) nested while first; the innermost level's body is
    // the pipelined payload and is not subject to boundary classification.
    Block *afterBlock = current.getAfterBody();
    WhileOp nextInner = nullptr;
    for (Operation &op : afterBlock->without_terminator()) {
      if (auto w = dyn_cast<WhileOp>(op)) {
        if (nextInner) // more than one while -> not flattenable
          return std::nullopt;
        nextInner = w;
      }
    }
    if (!nextInner)
      break; // innermost reached

    // Partition the rest into {pre-ops, post-ops}; the IV update addi may
    // sit anywhere and is skipped.
    bool seenInner = false;
    for (Operation &op : afterBlock->without_terminator()) {
      if (&op == level->updateOp.getOperation())
        continue;
      if (&op == nextInner.getOperation()) {
        seenInner = true;
        continue;
      }
      if (!seenInner) {
        // Pre-op: pure (recomputed every flattened iteration), or
        // load-shaped — one result, no regions, not pure — which may ONLY
        // feed a carried chain's init (validated during chain resolution;
        // re-executed once per pass in the flattened loop). Ops with a
        // memory-effects interface must be read-only to qualify.
        if (!isPure(&op)) {
          bool loadShaped =
              op.getNumResults() == 1 && op.getNumRegions() == 0;
          if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(&op))
            loadShaped &= memEffects.onlyHasEffect<MemoryEffects::Read>();
          if (!loadShaped)
            return std::nullopt;
        }
        info.preOps.back().push_back(&op);
      } else {
        // Post-op: pure, or a store-shaped op — resultless and region-free
        // (memref.store, loopschedule.store, ...). Its only observable
        // behavior is its side effect, which the scf.if gates to exactly
        // the iterations where the original epilogue ran.
        bool storeShaped =
            op.getNumResults() == 0 && op.getNumRegions() == 0;
        if (!isPure(&op) && !storeShaped)
          return std::nullopt;
        info.postOps.back().push_back(&op);
      }
    }

    current = nextInner;
  }

  if (nest.size() < 2)
    return std::nullopt;

  unsigned N = nest.size();

  // Almost-perfect structure (carried chains, pre/post ops) is only
  // handled for PIPELINED innermost loops: that is where eliminating the
  // per-pass fill/drain pays, and where the downstream scheduler's
  // predication machinery (ifOpConversion / launch-expect wrapping) is
  // exercised. Sequential frames lower predicated dynamic accesses
  // differently, so a sequential nest must be perfect to flatten.
  if (!nest.back().loop->hasAttr("hls.pipeline")) {
    bool hasBoundaryStructure = false;
    for (auto &ops : info.preOps)
      hasBoundaryStructure |= !ops.empty();
    for (auto &ops : info.postOps)
      hasBoundaryStructure |= !ops.empty();
    for (unsigned k = 0; k < N; ++k)
      hasBoundaryStructure |= nest[k].loop.getInits().size() > 1;
    if (hasBoundaryStructure)
      return std::nullopt;
  }

  // The innermost level must have no partition entries (it has no inner
  // while, so everything landed in preOps; that's its body, not a boundary).
  info.preOps.back().clear();
  info.postOps.back().clear();

  // --- Resolve extra iter-args into carried chains. ---
  // For each level's non-IV iter-args, walk the init side upward: an init
  // that is the parent's extra after-arg is threading; anything else must
  // be a constant or nest-invariant value and anchors the chain's origin.
  // Chains are keyed by their (level, argIdx) at the innermost loop.
  //
  // chainAt[k] maps iter-arg index at level k -> chain id (or absent).
  SmallVector<DenseMap<unsigned, unsigned>> chainAt(N);

  for (unsigned k = 0; k < N; ++k) {
    WhileOp loop = nest[k].loop;
    for (unsigned a = 0; a < loop.getInits().size(); ++a) {
      if (a == nest[k].ivIdx)
        continue;
      Value init = loop.getInits()[a];

      int origin;
      if (k == 0) {
        // Outermost extra arg: carried across the whole nest.
        if (!isConstantValue(init) && !isDefinedAboveNest(init, outer))
          return std::nullopt;
        origin = -1;
      } else {
        auto parentArg = dyn_cast<BlockArgument>(init);
        if (parentArg &&
            parentArg.getOwner() == nest[k - 1].loop.getAfterBody()) {
          // Threading from the parent: the parent must carry a chain at
          // this arg index, and the parent's yield for that index must be
          // exactly this loop's result at index `a`.
          unsigned pIdx = parentArg.getArgNumber();
          auto it = chainAt[k - 1].find(pIdx);
          if (it == chainAt[k - 1].end())
            return std::nullopt;
          Value parentYield =
              nest[k - 1].loop.getYieldOp().getOperand(pIdx);
          if (parentYield != loop.getResult(a))
            return std::nullopt;
          // Intermediate levels must not otherwise use the value: the
          // after-arg's only allowed uses are this loop's init (checked
          // here structurally: it IS this init) and nothing else.
          for (OpOperand &use : parentArg.getUses()) {
            if (use.getOwner() == loop.getOperation())
              continue;
            return std::nullopt;
          }
          chainAt[k][a] = it->second;
          info.chains[it->second].argIdx[k] = a;
          continue;
        }
        // New chain anchored at boundary k-1. Its init must be reset-safe:
        // a constant, a nest-invariant value, or a load-shaped pre-op of
        // the origin boundary (re-executed once per pass).
        Operation *initLoad = nullptr;
        if (!isConstantValue(init) && !isDefinedAboveNest(init, outer)) {
          Operation *def = init.getDefiningOp();
          if (def && !isPure(def) &&
              llvm::is_contained(info.preOps[k - 1], def))
            initLoad = def;
          else
            return std::nullopt;
        }
        origin = int(k) - 1;

        CarriedChain chain;
        chain.originLevel = origin;
        chain.init = init;
        chain.initLoad = initLoad;
        chain.argIdx.assign(N, -1);
        chain.argIdx[k] = a;
        chainAt[k][a] = info.chains.size();
        info.chains.push_back(chain);
        continue;
      }

      CarriedChain chain;
      chain.originLevel = origin;
      chain.init = init;
      chain.argIdx.assign(N, -1);
      chain.argIdx[k] = a;
      chainAt[k][a] = info.chains.size();
      info.chains.push_back(chain);
    }
  }

  // Every chain must reach the innermost loop (it is updated there); fill
  // in innerAfterArg / innerYieldVal / originResult and validate uses.
  for (CarriedChain &chain : info.chains) {
    int innerIdx = chain.argIdx[N - 1];
    if (innerIdx < 0)
      return std::nullopt; // dead-ends at an intermediate level
    WhileOp inner = nest[N - 1].loop;
    chain.innerAfterArg = inner.getAfterBody()->getArgument(innerIdx);
    chain.innerYieldVal = inner.getYieldOp().getOperand(innerIdx);

    // The result consumed by the chain's users.
    unsigned firstLevel = unsigned(chain.originLevel + 1);
    WhileOp originLoop = nest[firstLevel].loop;
    chain.originResult = originLoop.getResult(chain.argIdx[firstLevel]);
  }

  // --- Validate post-op operands and result escapes. ---
  // Post-ops at boundary k may consume: values defined above the nest,
  // IV after-args of levels <= k, pre-op results at boundaries <= k,
  // chain origin results anchored at boundary k, and other post-ops of the
  // same boundary. Their results must not escape the boundary's post-op set.
  for (unsigned k = 0; k + 1 < N; ++k) {
    llvm::SmallPtrSet<Operation *, 8> postSet;
    for (Operation *op : info.postOps[k])
      postSet.insert(op);
    llvm::SmallPtrSet<Operation *, 8> preSet;
    for (unsigned j = 0; j <= k; ++j)
      for (Operation *op : info.preOps[j])
        preSet.insert(op);

    for (Operation *op : info.postOps[k]) {
      for (Value operand : op->getOperands()) {
        if (isDefinedAboveNest(operand, outer))
          continue;
        if (auto arg = dyn_cast<BlockArgument>(operand)) {
          // IV after-arg of an enclosing level.
          bool ok = false;
          for (unsigned j = 0; j <= k; ++j)
            if (arg == nest[j].ivAfter)
              ok = true;
          if (ok)
            continue;
          return std::nullopt;
        }
        Operation *def = operand.getDefiningOp();
        if (postSet.contains(def) || preSet.contains(def))
          continue;
        // A chain result anchored at this boundary.
        bool isChainResult = false;
        for (const CarriedChain &chain : info.chains)
          if (chain.originLevel == int(k) && operand == chain.originResult)
            isChainResult = true;
        if (isChainResult)
          continue;
        return std::nullopt;
      }
      // Results may only feed other post-ops of this boundary.
      for (Value result : op->getResults())
        for (OpOperand &use : result.getUses())
          if (!postSet.contains(use.getOwner()))
            return std::nullopt;
    }
  }

  // --- Validate chain-result uses. ---
  // A chain's origin result may only be consumed by post-ops of its origin
  // boundary; for origin == -1 it escapes the nest (rewired to the
  // flattened loop's result).
  for (const CarriedChain &chain : info.chains) {
    if (chain.originLevel < 0)
      continue;
    llvm::SmallPtrSet<Operation *, 8> postSet;
    for (Operation *op : info.postOps[chain.originLevel])
      postSet.insert(op);
    for (OpOperand &use : chain.originResult.getUses())
      if (!postSet.contains(use.getOwner()))
        return std::nullopt;
  }

  // --- Validate load-shaped pre-op uses. ---
  // A non-pure pre-op's single result may only be a chain init (its
  // re-execution schedule is defined by the chain's pass structure).
  for (unsigned k = 0; k + 1 < N; ++k) {
    for (Operation *op : info.preOps[k]) {
      if (isPure(op))
        continue;
      bool isChainInit = false;
      for (const CarriedChain &chain : info.chains)
        if (chain.initLoad == op)
          isChainInit = true;
      if (!isChainInit)
        return std::nullopt;
      for (OpOperand &use : op->getResult(0).getUses())
        if (use.getOwner() != nest[k + 1].loop.getOperation())
          return std::nullopt;
    }
  }

  // --- Validate pre-op operand availability. ---
  // Pre-ops at boundary k may use: values above the nest, IV after-args of
  // levels <= k, and earlier pre-op results (any boundary <= k).
  {
    llvm::SmallPtrSet<Operation *, 16> preSoFar;
    for (unsigned k = 0; k + 1 < N; ++k) {
      for (Operation *op : info.preOps[k]) {
        for (Value operand : op->getOperands()) {
          if (isDefinedAboveNest(operand, outer))
            continue;
          if (auto arg = dyn_cast<BlockArgument>(operand)) {
            bool ok = false;
            for (unsigned j = 0; j <= k; ++j)
              if (arg == nest[j].ivAfter)
                ok = true;
            if (ok)
              continue;
            return std::nullopt;
          }
          if (preSoFar.contains(operand.getDefiningOp()))
            continue;
          return std::nullopt;
        }
        preSoFar.insert(op);
      }
    }
  }

  // Intermediate IV results and dead extras: the loops' IV results must be
  // unused (the enclosing yield uses the addi, not the result); any other
  // use would break after flattening.
  for (unsigned k = 1; k < N; ++k) {
    WhileOp loop = nest[k].loop;
    for (unsigned r = 0; r < loop.getNumResults(); ++r) {
      if (int(r) == int(nest[k].ivIdx)) {
        if (!loop.getResult(r).use_empty())
          return std::nullopt;
        continue;
      }
      // Chain results were validated above; anything else unused is fine,
      // used is not.
      bool isChain = chainAt[k].contains(r);
      if (!isChain && !loop.getResult(r).use_empty())
        return std::nullopt;
      // Threaded (non-origin) chain results are consumed by the parent's
      // yield only — already verified during threading.
    }
  }
  // The outermost loop: IV result may be used (rewired to the flat loop's
  // result); chain results with origin -1 are rewired too. Other extras
  // were rejected during chain construction.

  return info;
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

/// Emit the flattened scf.while for a collected (almost-)perfect nest.
static void emitFlattenedNest(OpBuilder &builder, NestInfo &info) {
  ArrayRef<NestLevel> nest = info.levels;
  WhileOp outer = nest.front().loop;
  WhileOp inner = nest.back().loop;
  unsigned N = nest.size();
  unsigned C = info.chains.size();
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

  // --- Inits: one per level, one per address candidate, one per chain. ---
  SmallVector<Value> inits;
  SmallVector<Type> ivTypes;
  inits.reserve(N + M + C);
  ivTypes.reserve(N + M + C);
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
  // Chain inits: values above the nest are used directly; constants that
  // live inside the nest (e.g. the zero right before the reduction loop)
  // are rematerialized here so dominance holds. Load-initialized chains
  // clone their pure index pre-ops and the load itself with every IV at
  // its lower bound — the first pass's init, executed once before the
  // loop just as the original program did.
  IRMapping lbMap;
  for (unsigned k = 0; k < N; ++k) {
    auto lbCst = builder.create<arith::ConstantOp>(
        loc, nest[k].ivType, IntegerAttr::get(nest[k].ivType, nest[k].lb));
    lbMap.map(nest[k].ivAfter, lbCst);
  }
  bool clonedLbPreOps = false;
  SmallVector<Value> chainInits;
  for (const CarriedChain &chain : info.chains) {
    Value init = chain.init;
    if (chain.initLoad) {
      if (!clonedLbPreOps) {
        // Clone the pure pre-ops of every boundary once, in program
        // order, at lower-bound IVs (dead ones fall to later CSE/DCE).
        for (unsigned k = 0; k + 1 < N; ++k)
          for (Operation *op : info.preOps[k])
            if (isPure(op))
              builder.clone(*op, lbMap);
        clonedLbPreOps = true;
      }
      Operation *cloned = builder.clone(*chain.initLoad, lbMap);
      if (auto nameAttr = cloned->getAttrOfType<StringAttr>(
              "loopschedule.name"))
        cloned->setAttr("loopschedule.name",
                        StringAttr::get(ctx, nameAttr.getValue() + ".init"));
      init = cloned->getResult(0);
    } else if (!isDefinedAboveNest(init, outer)) {
      Operation *cloned = builder.clone(*init.getDefiningOp());
      init = cloned->getResult(cast<OpResult>(chain.init).getResultNumber());
    }
    chainInits.push_back(init);
    inits.push_back(init);
    ivTypes.push_back(init.getType());
  }

  auto flat = builder.create<WhileOp>(loc, ivTypes, inits);

  // Preserve the pipeline attribute from the innermost loop so that
  // downstream scheduling passes (SCFToLoopSchedule) still recognize
  // the flattened loop as pipelined.
  if (auto pipeAttr = inner->getAttr("hls.pipeline"))
    flat->setAttr("hls.pipeline", pipeAttr);

  // The flattened trip count is the product of the per-level trips;
  // restore the metadata the per-level loops carried.
  {
    uint64_t total = 1;
    for (const NestLevel &lvl : nest) {
      APInt range = lvl.ubNormalized - lvl.lb;
      APInt step = lvl.step;
      uint64_t trips =
          (range.zext(64) + (step.zext(64) - 1)).udiv(step.zext(64))
              .getZExtValue();
      total *= trips;
    }
    flat->setAttr("loopschedule.trip_count",
                  IntegerAttr::get(IntegerType::get(ctx, 64), total));
  }

  // --- Before region. ---
  {
    Block *beforeBlock =
        builder.createBlock(&flat.getBefore(), {}, ivTypes,
                            SmallVector<Location>(N + M + C, loc));
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
                            SmallVector<Location>(N + M + C, loc));
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(afterBlock);

    // Map each level's after-region IV block arg to the corresponding
    // flattened block arg, and each chain's innermost after-arg to its
    // flattened iter-arg.
    IRMapping mapping;
    for (unsigned k = 0; k < N; ++k)
      mapping.map(nest[k].ivAfter, afterBlock->getArgument(k));
    for (unsigned c = 0; c < C; ++c)
      mapping.map(info.chains[c].innerAfterArg,
                  afterBlock->getArgument(N + M + c));

    // Clone the PURE boundary pre-ops (outermost boundary first,
    // preserving program order and def-use among them); re-executing them
    // every flattened iteration is safe, and inner-body ops that
    // referenced them keep working through the mapping. Load-shaped
    // pre-ops are NOT body ops — the chain machinery re-executes them
    // once per pass.
    for (unsigned k = 0; k + 1 < N; ++k)
      for (Operation *op : info.preOps[k])
        if (isPure(op))
          builder.clone(*op, mapping);

    // Clone the innermost body, skipping the IV update addi and the yield.
    Block *innerAfter = inner.getAfterBody();
    arith::AddIOp innerUpdateOp = nest.back().updateOp;
    Operation *innerUpdate = innerUpdateOp.getOperation();
    for (Operation &op : innerAfter->without_terminator()) {
      if (&op == innerUpdate)
        continue;
      builder.clone(op, mapping);
    }

    // The chains' updated values as visible in the flattened body.
    SmallVector<Value> chainNext(C);
    for (unsigned c = 0; c < C; ++c)
      chainNext[c] = mapping.lookupOrDefault(info.chains[c].innerYieldVal);

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
    // done signal per level: doneAtLevel[k] == "levels k..N-1 all wrapped
    // this iteration" (defined for k >= 1). Boundary k's epilogue and chain
    // resets key off doneAtLevel[k+1].
    SmallVector<Value> doneAtLevel(N, Value());

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
      doneAtLevel[k] = doneHere;
    }

    // --- Boundary epilogues. ---
    // Post-ops at boundary b ran in the original program each time the
    // levels below b completed a full pass: guard them with
    // scf.if(doneAtLevel[b+1]). Innermost boundary first, matching original
    // execution order when several boundaries complete on the same
    // iteration. Chain origin results map to the chains' updated values.
    for (unsigned c = 0; c < C; ++c)
      mapping.map(info.chains[c].originResult, chainNext[c]);
    for (int b = int(N) - 2; b >= 0; --b) {
      if (info.postOps[b].empty())
        continue;
      auto ifOp = builder.create<scf::IfOp>(loc, doneAtLevel[b + 1],
                                            /*withElseRegion=*/false);
      OpBuilder::InsertionGuard g2(builder);
      builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
      for (Operation *op : info.postOps[b])
        builder.clone(*op, mapping);
    }

    // --- Chain yields: reset to the init when the levels below the origin
    // wrapped (the next iteration starts a fresh pass); origin -1 chains
    // never reset. Constant/invariant inits reset via arith.select;
    // load-initialized chains re-load under the wrap predicate with the
    // NEXT pass's IVs (the odometer's ivOut values). ---
    IRMapping nextMap;
    for (unsigned k = 0; k < N; ++k)
      nextMap.map(nest[k].ivAfter, ivOut[k]);
    SmallVector<Value> chainOut(C);
    for (unsigned c = 0; c < C; ++c) {
      const CarriedChain &chain = info.chains[c];
      if (chain.originLevel < 0) {
        chainOut[c] = chainNext[c];
        continue;
      }
      Value reset = doneAtLevel[chain.originLevel + 1];
      if (!chain.initLoad) {
        chainOut[c] = builder.create<arith::SelectOp>(
            loc, reset, chainInits[c], chainNext[c]);
        continue;
      }
      auto ifOp = builder.create<scf::IfOp>(
          loc, TypeRange{chain.init.getType()}, reset,
          /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        for (unsigned k = 0; k + 1 < N; ++k)
          for (Operation *op : info.preOps[k])
            if (isPure(op))
              builder.clone(*op, nextMap);
        // This clone KEEPS the original loopschedule.name: it is the one
        // inside the pipelined loop, so the memory-dependence records that
        // reference the original load by name must resolve to it.
        Operation *cloned = builder.clone(*chain.initLoad, nextMap);
        builder.create<scf::YieldOp>(loc, cloned->getResult(0));
      }
      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc, chainNext[c]);
      }
      chainOut[c] = ifOp.getResult(0);
    }

    SmallVector<Value> yieldOperands(ivOut.begin(), ivOut.end());
    yieldOperands.append(addrAccum.begin(), addrAccum.end());
    yieldOperands.append(chainOut.begin(), chainOut.end());
    builder.create<YieldOp>(loc, yieldOperands);
  }

  // Rewire the outer while's used results: its IV result maps to the flat
  // loop's corresponding IV result, and origin == -1 chains map to their
  // flat iter-arg results.
  outer.getResult(nest.front().ivIdx)
      .replaceAllUsesWith(flat.getResult(0));
  for (unsigned c = 0; c < C; ++c) {
    CarriedChain &chain = info.chains[c];
    if (chain.originLevel == -1)
      chain.originResult.replaceAllUsesWith(flat.getResult(N + M + c));
  }

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
    auto nest = collectNest(op);
    if (!nest)
      continue;
    for (NestLevel &lvl : nest->levels)
      consumed.insert(lvl.loop.getOperation());
    emitFlattenedNest(builder, *nest);
  }
}

namespace circt {
std::unique_ptr<mlir::Pass> createSCFWhileLoopFlatteningPass() {
  return std::make_unique<SCFWhileLoopFlatteningPass>();
}
} // namespace circt
