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
//     constant, a nest-invariant value, or a LOAD-SHAPED pre-op, possibly
//     under pure ops (an accumulator resuming from an indexed buffer,
//     sum[p], or from a scaled one, D[i][j] * beta); threaded
//     unchanged through intermediate levels, and updated only in the
//     innermost body. In the flattened loop each chain becomes one
//     iter-arg that RESETS to its init on every pass: a constant or
//     invariant init via arith.select when the levels below its origin
//     wrap; a load-initialized one by re-loading it (and its pure ops) on
//     the pass's first iteration, under an scf.if, in place of the carried
//     value, as Vitis HLS's flattened loops do.
//   - POST-ops: pure ops and memref.stores after the inner loop (e.g. the
//     reduction's store to memory). These become an scf.if predicated on
//     "the levels below just wrapped", i.e. they execute exactly on the
//     iterations where the original epilogue ran. Downstream scheduling
//     (ifOpConversion) turns the scf.if into predicated operations.
//
// MEMORY DEPENDENCES. The function's dependence table (loopschedule.
// dependencies) was built on the separate loops, where a pipeline over the
// inner one never overlapped two outer iterations. Flattening merges them,
// so each entry between two accesses of the nest is re-expressed in the
// flattened loop's iterations: one carried by an outer level (tagged
// `outer<depth, [...]>` by construct-memory-dependencies) becomes a
// distance of that level's stride (gemm's k-carried C[i][j] update, NJ
// iterations apart), and one with an access in a prologue or epilogue
// moves with it (a re-load runs on its pass's first iteration, an epilogue
// on the pass's last iteration). A nest whose entry has no such distance stays
// unflattened: one that needs a runtime trip count (syrk's
// `for k: for j < i + 1`, whose k-carried update is i + 1 iterations
// apart), or whose flattened order could run the destination first.
//
// Bounds may be CONSTANT or RUNTIME. A runtime (SSA) upper bound must
// dominate the whole nest (nest-invariance — a bound loaded per outer
// iteration, spmv's rowptr, is structurally excluded), with an slt/ult
// predicate (already exclusive, nothing to normalize) and a constant step.
// The odometer's wrap compare then reads the held SSA value, and the
// flattened entry condition ANDs every runtime level's compare so a
// zero-trip inner level (cols == 0) exits before executing anything —
// exactly the original perfect nest's behavior. A runtime bound on an
// INNER level takes a PERFECT nest with no dynamic-latency access
// (matching Vitis, which flattens the on-chip scatter nest and disables
// flattening for its m_axi variant): the original per-pass epilogue runs
// even when the levels below zero-trip, but the wrap-predicated form
// would skip it. A runtime bound on the OUTERMOST level alone (a
// triangular `for j < M - i` inside an unflattened `i`) is no such case
// and takes any nest the constant path does, AXI accesses included, as
// Vitis flattens covariance's and correlation's m_axi `j`/`k` nests.
//
// LOADS used in the inner body are NOT absorbed here — a load pre-op still
// rejects the nest. AmcLoopReperfection (which runs immediately before this
// pass in the AMC pipeline) owns their placement: it SINKS on-chip invariant
// load pre-ops back into the loop body (restoring perfection, re-reads are
// free) and HOISTS AXI word-mode invariant loads out (a redundant read is a
// bus round trip; that nest then correctly stays unflattened). A load
// pre-op reaching this pass therefore means reperfection had a reason to
// refuse (a store dependence, multiple consumer loops, an on-chip arbiter
// port), and the conservative rejection below is the right response — do
// not "fix" it by absorbing loads here.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/SCFWhileTripCountAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <limits>

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
  APInt lb;                // Constant lower bound (initial value);
                           // meaningless when lbValue is set.
  APInt ubNormalized;      // Exclusive upper bound after predicate
                           // normalization (sle→slt+1, ule→ult+1).
                           // Meaningless when ubValue is set.
  APInt step;              // Constant positive step.
  arith::CmpIPredicate origPred;
  arith::AddIOp updateOp;  // `addi iv_after, step` feeding the yield.
  BlockArgument ivAfter;   // The after-region IV block argument.
  IntegerType ivType;
  /// Non-null for a RUNTIME upper bound: the original compare's rhs, an
  /// SSA value collectNest verifies dominates the whole nest. Only
  /// slt/ult levels may carry one (the bound is already exclusive).
  Value ubValue = nullptr;
  /// Non-null for a RUNTIME lower bound: the loop's IV init, an SSA value.
  /// Only the outermost level may carry one (collectNest): it never wraps,
  /// so the bound is just its start, and the inner levels restart from
  /// their constant lower bounds.
  Value lbValue = nullptr;
  bool isRuntime() const { return ubValue || lbValue; }
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
  Value ubValue = nullptr;
  bool constantUb = matchPattern(cmp.getRhs(), m_ConstantInt(&ubConst));
  if (!constantUb) {
    // Runtime upper bound: slt/ult only (already-exclusive bounds need no
    // normalization). Nest-invariance is checked by collectNest, which
    // knows the outermost loop.
    if (cmp.getPredicate() != arith::CmpIPredicate::slt &&
        cmp.getPredicate() != arith::CmpIPredicate::ult)
      return std::nullopt;
    ubValue = cmp.getRhs();
  }

  auto ivType = dyn_cast<IntegerType>(ivBefore.getType());
  if (!ivType)
    return std::nullopt;
  unsigned bw = ivType.getWidth();

  APInt lb(bw, 0);
  Value lbValue = nullptr;
  if (!matchPattern(whileOp.getInits()[idx], m_ConstantInt(&lb))) {
    // Runtime lower bound (e.g. a triangular `for j in i+1..M`): not with
    // `ne`, whose bound must be known to be ahead of the IV.
    if (cmp.getPredicate() == arith::CmpIPredicate::ne)
      return std::nullopt;
    lbValue = whileOp.getInits()[idx];
  }
  if (lb.getBitWidth() != bw || (constantUb && ubConst.getBitWidth() != bw))
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

  // Normalize inclusive predicates to exclusive bounds. A runtime bound
  // is slt/ult by construction — already exclusive, nothing to compute.
  if (!constantUb)
    return NestLevel{whileOp, idx,  lb,     APInt::getZero(bw), step,
                     cmp.getPredicate(), addOp, ivAfter, ivType, ubValue,
                     lbValue};
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
    // Require lb < ub so the loop actually walks toward the bound (a
    // runtime lb was refused above).
    bool lbLtUb = isSigned ? lb.slt(ub) : lb.ult(ub);
    if (!lbLtUb)
      return std::nullopt;
    break;
  }
  default:
    return std::nullopt;
  }

  return NestLevel{whileOp, idx,  lb,     ub,     step,
                   cmp.getPredicate(), addOp, ivAfter, ivType, nullptr,
                   lbValue};
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
  /// sum[p]). The flattened loop re-executes it once per pass, on the
  /// pass's first iteration, under an scf.if.
  Operation *initLoad = nullptr;
  /// The pure pre-ops computing the init from initLoad's result, in program
  /// order (an accumulator resuming from a scaled element: D[i][j] * beta);
  /// empty when the init is the load itself. They run with the load, never
  /// in the flattened body.
  SmallVector<Operation *> initOps;
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
  /// Dependence-table entries between accesses of the nest, with their
  /// distance in the flattened loop's iterations (0: the same iteration,
  /// the source first). Applied once the nest is flattened.
  SmallVector<std::pair<circt::loopschedule::LoopScheduleAccessOp, int64_t>>
      flatDeps;
  /// Entries between accesses of the nest carried by a loop above it, with
  /// their inner distances once the nest is one loop.
  SmallVector<std::pair<circt::loopschedule::LoopScheduleAccessOp,
                        SmallVector<int64_t>>>
      aboveDeps;
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

/// Hoist out of every loop the pure ops between the loops of the
/// single-child while chain rooted at `outer` that depend on no value
/// defined in a loop: an inner level's runtime bound the affine lowering
/// computed per outer iteration (fdtd_2d's `range(1, NY)`, normalized to
/// `NY - 1` inside the `i` loop). Left there, it is a pre-op and a bound
/// not defined above the nest, either of which keeps a runtime nest
/// unflattened. Only all the way out: an op invariant in `outer` alone,
/// moved into an enclosing loop's body, would break that loop's nest.
static void hoistInvariantPreOps(WhileOp outer) {
  Operation *top = outer;
  while (isa_and_nonnull<WhileOp, scf::ForOp>(top->getParentOp()))
    top = top->getParentOp();
  if (top->getParentOp() != outer->getParentOfType<func::FuncOp>())
    return;
  for (WhileOp current = outer; current;) {
    WhileOp inner = nullptr;
    for (Operation &op : current.getAfterBody()->without_terminator())
      if (auto w = dyn_cast<WhileOp>(op)) {
        if (inner)
          return;
        inner = w;
      }
    if (!inner)
      return;
    for (Operation &op : llvm::make_early_inc_range(
             current.getAfterBody()->without_terminator())) {
      if (&op == inner.getOperation())
        break;
      if (isPure(&op) && op.getNumRegions() == 0 &&
          llvm::all_of(op.getOperands(), [&](Value v) {
            if (auto arg = dyn_cast<BlockArgument>(v))
              return !top->isAncestor(arg.getOwner()->getParentOp());
            return !top->isAncestor(v.getDefiningOp());
          }))
        op.moveBefore(top);
    }
    current = inner;
  }
}

/// True if `v` is produced by a ConstantLike op (rematerializable anywhere;
/// covers integer and float accumulator inits).
static bool isConstantValue(Value v) {
  Operation *def = v.getDefiningOp();
  return def && def->hasTrait<OpTrait::ConstantLike>();
}

/// Match a chain init computed at a boundary from exactly one load-shaped
/// pre-op in `preOps`: the load itself, or pure pre-ops over its result.
/// Sets `load`, and `ops` to the pure pre-ops between the load and the
/// init, in program order. (Pure pre-ops the init needs that do not depend
/// on the load, e.g. the load's address, stay ordinary pre-ops.)
static bool matchInitLoad(Value init, ArrayRef<Operation *> preOps,
                          Operation *&load, SmallVector<Operation *> &ops) {
  Operation *def = init.getDefiningOp();
  if (!def || !llvm::is_contained(preOps, def))
    return false;
  if (!isPure(def)) {
    load = def;
    return true;
  }
  llvm::SetVector<Operation *> slice;
  SmallVector<Operation *> worklist{def};
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!slice.insert(op))
      continue;
    for (Value operand : op->getOperands())
      if (Operation *d = operand.getDefiningOp();
          d && llvm::is_contained(preOps, d))
        worklist.push_back(d);
  }
  for (Operation *op : slice) {
    if (isPure(op))
      continue;
    if (load)
      return false; // more than one load
    load = op;
  }
  if (!load)
    return false; // a pure function of the IVs: not handled
  llvm::SmallPtrSet<Operation *, 8> fromLoad{load};
  for (Operation *op : preOps) {
    if (op == load || !slice.contains(op))
      continue;
    if (llvm::any_of(op->getOperands(), [&](Value v) {
          return fromLoad.contains(v.getDefiningOp());
        })) {
      fromLoad.insert(op);
      ops.push_back(op);
    }
  }
  return !ops.empty() && ops.back() == def;
}

/// A level's trip count, when its bounds are constants.
static std::optional<int64_t> constantTrips(const NestLevel &lvl) {
  if (lvl.isRuntime())
    return std::nullopt;
  APInt range = lvl.ubNormalized - lvl.lb;
  APInt step = lvl.step;
  return (range.zext(64) + (step.zext(64) - 1))
      .udiv(step.zext(64))
      .getZExtValue();
}

/// The depth of `op` among every loop around it, itself included (1 =
/// outermost), as the dependence table's `outer<depth, ...>` counts.
static unsigned loopDepth(Operation *op) {
  unsigned depth = 0;
  for (Operation *p = op; p; p = p->getParentOp())
    depth += isa<LoopLikeOpInterface>(p);
  return depth;
}

/// Re-express the dependence-table entries between accesses of the nest in
/// the flattened loop's iterations (`info.flatDeps`, `info.aboveDeps`). The
/// table was built on the separate loops, where a pipeline over the inner
/// one never overlapped two outer iterations: a dependence carried by an
/// outer level (`outer<...>`), or one with an access in a boundary's
/// prologue or epilogue, reaches the flattened pipeline only through this.
/// Fails when an entry has no such distance: it needs a runtime trip count
/// (syrk's `for k: for j < i + 1: C[i][j] += ...`, whose k-carried
/// dependence is i + 1 flattened iterations apart), or the flattened order
/// could run the destination first.
static bool planDependences(NestInfo &info) {
  using namespace circt::loopschedule;
  ArrayRef<NestLevel> nest = info.levels;
  WhileOp outer = nest.front().loop;
  auto func = outer->getParentOfType<func::FuncOp>();
  auto depsRef =
      func ? func->getAttrOfType<SymbolRefAttr>("loopschedule.dependencies")
           : nullptr;
  auto depsOp = depsRef ? dyn_cast_or_null<LoopScheduleDependenciesOp>(
                              SymbolTable::lookupNearestSymbolFrom(func, depsRef))
                        : nullptr;
  if (!depsOp)
    return true;
  unsigned N = nest.size();

  // Where each access of the nest runs once flattened: the deepest level
  // around it, and its place in the flattened body (the chains' re-loads,
  // then the innermost body, then the boundaries' epilogues innermost
  // first).
  enum Kind { Body, Epilogue, Reload };
  struct Position {
    Kind kind;
    unsigned level;
    std::pair<unsigned, unsigned> order;
  };
  llvm::StringMap<std::optional<Position>> positions;
  unsigned walkIndex = 0;
  outer->walk<WalkOrder::PreOrder>([&](Operation *op) {
    ++walkIndex;
    auto name = op->getAttrOfType<StringAttr>("loopschedule.name");
    if (!name)
      return;
    std::optional<Position> &pos = positions[name.getValue()];
    if (nest.back().loop->isProperAncestor(op)) {
      pos = Position{Body, N - 1, {1, walkIndex}};
      return;
    }
    for (auto [c, chain] : llvm::enumerate(info.chains))
      if (chain.initLoad == op)
        pos = Position{Reload, unsigned(chain.originLevel), {0, unsigned(c)}};
    for (unsigned k = 0; k + 1 < N; ++k)
      if (llvm::is_contained(info.postOps[k], op))
        pos = Position{Epilogue, k, {N - k, walkIndex}};
  });

  SmallVector<std::optional<int64_t>> trips;
  for (const NestLevel &lvl : nest)
    trips.push_back(constantTrips(lvl));
  // Flattened iterations per iteration of level x.
  auto stride = [&](unsigned x) -> std::optional<int64_t> {
    std::optional<int64_t> p = 1;
    for (unsigned y = x + 1; y < N && p; ++y)
      p = trips[y] ? llvm::checkedMul(*p, *trips[y]) : std::nullopt;
    return p;
  };
  auto mulAdd = [](std::optional<int64_t> acc, std::optional<int64_t> a,
                   std::optional<int64_t> b) -> std::optional<int64_t> {
    if (!acc || !a || !b)
      return std::nullopt;
    return llvm::checkedMulAdd(*a, *b, *acc);
  };
  // Table distances are in IV units.
  auto iterations = [&](int64_t ivDist, unsigned x) {
    return llvm::divideCeilSigned(ivDist, nest[x].step.getSExtValue());
  };
  constexpr int64_t unknown = std::numeric_limits<int64_t>::min();
  unsigned base = loopDepth(outer);

  for (auto dependsOn : depsOp.getBody().getOps<LoopScheduleDependsOnOp>()) {
    auto dstIt = positions.find(dependsOn.getAccessName());
    if (dstIt == positions.end())
      continue;
    for (auto access : dependsOn.getBody().getOps<LoopScheduleAccessOp>()) {
      auto srcIt = positions.find(access.getAccessName());
      if (srcIt == positions.end())
        continue;
      // An access of the nest flattening has no place for.
      if (!srcIt->second || !dstIt->second)
        return false;
      const Position &src = *srcIt->second, &dst = *dstIt->second;
      unsigned c0 = std::min(src.level, dst.level);
      int64_t dist = access.getDist().value_or(0);
      ArrayRef<int64_t> inner =
          access.getInnerDist().value_or(ArrayRef<int64_t>{});

      // Carried by a loop above the nest: the bounds below it now end at
      // the flattened loop, whose own one is unknown.
      std::optional<int64_t> carrier;
      if (auto depth = access.getOuterDepth()) {
        carrier = int64_t(*depth) - int64_t(base);
        if (*carrier < 0) {
          SmallVector<int64_t> above(inner.take_front(
              std::min<size_t>(inner.size(), base - *depth - 1)));
          above.push_back(unknown);
          info.aboveDeps.push_back({access, std::move(above)});
          continue;
        }
        if (*carrier > int64_t(c0))
          return false;
      }

      // Lower bound of the distance from the levels around both:
      // sum over x of (iterations apart in level x) * stride(x).
      std::optional<int64_t> distance = 0;
      if (carrier) {
        distance = mulAdd(distance, iterations(dist, *carrier),
                          stride(*carrier));
        for (unsigned x = *carrier + 1; x <= c0; ++x) {
          size_t i = x - *carrier - 1;
          std::optional<int64_t> apart;
          if (i < inner.size() && inner[i] != unknown)
            apart = iterations(inner[i], x);
          if (trips[x])
            apart = std::max(apart.value_or(unknown), 1 - *trips[x]);
          distance = mulAdd(distance, apart, stride(x));
        }
      } else if (dist > 0) {
        distance = mulAdd(distance, iterations(dist, c0), stride(c0));
      }
      // ... plus where in their iteration of levels 0..c0 the two run:
      // anywhere in the body; on the levels' last iteration in an epilogue;
      // on a pass's first iteration for a re-load.
      auto offsets = [&](const Position &p)
          -> std::optional<std::pair<int64_t, int64_t>> {
        std::optional<int64_t> all = stride(c0), own = stride(p.level);
        if (!all || !own)
          return std::nullopt;
        switch (p.kind) {
        case Body:
          return std::make_pair(int64_t(0), *all - 1);
        case Epilogue:
          return std::make_pair(*own - 1, *all - 1);
        case Reload:
          return std::make_pair(int64_t(0), *all - *own);
        }
        llvm_unreachable("unknown position");
      };
      auto srcOffsets = offsets(src), dstOffsets = offsets(dst);
      if (!distance || !srcOffsets || !dstOffsets)
        return false;
      int64_t flat = *distance + dstOffsets->first - srcOffsets->second;
      bool srcFirst = src.order < dst.order;
      if (flat < 0 || (flat == 0 && !srcFirst))
        return false;
      // A source earlier in the body only takes a same-iteration edge,
      // which orders every later iteration too.
      info.flatDeps.push_back({access, srcFirst ? 0 : flat});
    }
  }
  return true;
}

/// Rewrite the dependence-table entries `planDependences` placed.
static void applyDependences(NestInfo &info) {
  for (auto [access, dist] : info.flatDeps) {
    Builder b(access);
    access.removeOuterDepthAttr();
    access.removeInnerDistAttr();
    if (dist > 0)
      access.setDistAttr(b.getI64IntegerAttr(dist));
    else
      access.removeDistAttr();
  }
  for (auto &[access, inner] : info.aboveDeps)
    access.setInnerDistAttr(Builder(access).getDenseI64ArrayAttr(inner));
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

    if (level->isRuntime()) {
      // A runtime bound must be nest-invariant: usable from the flattened
      // loop's condition and wrap compares, i.e. defined above the whole
      // nest. (spmv's per-row `rowptr[i]` bound is defined inside it and
      // lands here.) Zero-trip degeneracy is handled dynamically by the
      // ANDed entry condition instead of a static trip-count check.
      if (level->ubValue && !isDefinedAboveNest(level->ubValue, outer))
        return std::nullopt;
      // A runtime lower bound only on the outermost level, whose init is
      // the nest's own operand: an inner level restarts from its lower
      // bound every pass, which the odometer's constant reset cannot do.
      if (level->lbValue && !nest.empty())
        return std::nullopt;
    } else {
      // Require a non-zero constant trip count so the flattened loop isn't
      // degenerate. (Computed locally: the public analysis only accepts
      // single-iter-arg loops, and levels here may carry chains.)
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

  // Nests with a runtime INNER bound must be PERFECT: the wrap-predicated
  // form skips a boundary epilogue whenever the levels below zero-trip,
  // but the original ran it — so any pre-op, post-op, or carried chain
  // refuses (Vitis draws the same line: it flattens the perfect scatter
  // nest and leaves every reduction-epilogue nest alone). They must also
  // be free of dynamic-latency accesses (Vitis disables flattening on the
  // scatter's m_axi variant). A runtime bound on the outermost level alone
  // is neither case: the inner levels' trips are constant and non-zero,
  // and the outermost level's own zero trip is the flattened loop's entry
  // condition, as in the original. Vitis flattens such nests, epilogue,
  // AXI accesses and all: covariance's and correlation's triangular `j`
  // around the `k` reduction, one II 2 pipeline where an unflattened `k`
  // drains and restarts on every `j`.
  bool runtimeInner = llvm::any_of(
      llvm::drop_begin(nest), [](const NestLevel &l) { return l.isRuntime(); });
  if (runtimeInner) {
    for (NestLevel lvl : nest)
      if (lvl.loop.getInits().size() != 1)
        return std::nullopt;
    for (auto &ops : info.preOps)
      if (!ops.empty())
        return std::nullopt;
    for (auto &ops : info.postOps)
      if (!ops.empty())
        return std::nullopt;
    bool hasDynAccess = false;
    nest.back().loop->walk([&](Operation *op) {
      bool dyn = false;
      if (auto l = dyn_cast<circt::loopschedule::LoadInterface>(op))
        dyn = l.isDynamic();
      else if (auto st = dyn_cast<circt::loopschedule::StoreInterface>(op))
        dyn = st.isDynamic();
      if (!dyn)
        return WalkResult::advance();
      hasDynAccess = true;
      return WalkResult::interrupt();
    });
    if (hasDynAccess)
      return std::nullopt;
  }

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
        // the origin boundary, possibly under pure ops (re-executed once
        // per pass).
        Operation *initLoad = nullptr;
        SmallVector<Operation *> initOps;
        if (!isConstantValue(init) && !isDefinedAboveNest(init, outer) &&
            !matchInitLoad(init, info.preOps[k - 1], initLoad, initOps))
          return std::nullopt;
        origin = int(k) - 1;

        CarriedChain chain;
        chain.originLevel = origin;
        chain.init = init;
        chain.initLoad = initLoad;
        chain.initOps = std::move(initOps);
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
  // A non-pure pre-op's single result may only compute ONE chain's init
  // (its re-execution schedule is defined by that chain's pass
  // structure): it and the pure ops over it feed only each other and the
  // next level's init.
  for (unsigned k = 0; k + 1 < N; ++k) {
    for (Operation *op : info.preOps[k]) {
      if (isPure(op))
        continue;
      const CarriedChain *owner = nullptr;
      for (const CarriedChain &chain : info.chains) {
        if (chain.initLoad != op)
          continue;
        if (owner)
          return std::nullopt; // one load, two chains
        owner = &chain;
      }
      if (!owner)
        return std::nullopt;
      llvm::SmallPtrSet<Operation *, 8> initSet(owner->initOps.begin(),
                                                owner->initOps.end());
      initSet.insert(op);
      for (Operation *initOp : initSet)
        for (OpOperand &use : initOp->getUses())
          if (use.getOwner() != nest[k + 1].loop.getOperation() &&
              !initSet.contains(use.getOwner()))
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

  if (!planDependences(info))
    return std::nullopt;
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
  // Linearized-address promotion needs every child level's RANGE as a
  // compile-time constant (the wrap-reversal addend); with a runtime
  // bound the body's own address arithmetic keeps working through the IV
  // mapping — same expressions the unflattened inner body computed — so
  // promotion is simply skipped.
  bool anyRuntime =
      llvm::any_of(nest, [](const NestLevel &l) { return l.isRuntime(); });
  SmallVector<AddrCandidate> candidates;
  if (!anyRuntime)
    candidates = collectAddrCandidates(inner, nest);
  unsigned M = candidates.size();

  builder.setInsertionPoint(outer);

  // --- Inits: one per level, one per address candidate, one per chain. ---
  SmallVector<Value> inits;
  SmallVector<Type> ivTypes;
  inits.reserve(N + M + C);
  ivTypes.reserve(N + M + C);
  for (NestLevel lvl : nest) {
    inits.push_back(lvl.lbValue
                        ? lvl.lbValue
                        : builder.create<arith::ConstantOp>(
                              loc, lvl.ivType,
                              IntegerAttr::get(lvl.ivType, lvl.lb)));
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
  // are rematerialized here so dominance holds. A load-initialized chain
  // loads its init on every pass's first iteration, the first included
  // (see the body): its carried value starts as a zero no iteration reads.
  // The pure pre-ops that compute a chain init from its load: they run
  // with the load, never with the other pre-ops.
  llvm::SmallPtrSet<Operation *, 8> initOps;
  for (const CarriedChain &chain : info.chains)
    initOps.insert(chain.initOps.begin(), chain.initOps.end());
  auto isBodyPreOp = [&](Operation *op) {
    return isPure(op) && !initOps.contains(op);
  };

  SmallVector<Value> chainInits;
  for (const CarriedChain &chain : info.chains) {
    Value init = chain.init;
    if (chain.initLoad) {
      Type type = chain.init.getType();
      init = builder.create<arith::ConstantOp>(loc, type,
                                               builder.getZeroAttr(type));
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
  // restore the metadata the per-level loops carried. Unknown for a
  // runtime-bound nest — the scheduler already handles bound-less loops
  // (none of the unflattened runtime loops carried the attribute either).
  if (!anyRuntime) {
    uint64_t total = 1;
    for (const NestLevel &lvl : nest)
      total *= *constantTrips(lvl);
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
    // the ORIGINAL (pre-normalization) ub and predicate.
    const NestLevel &outerLvl = nest.front();
    Value outerUb;
    if (outerLvl.ubValue) {
      outerUb = outerLvl.ubValue;
    } else {
      auto ubAttr = IntegerAttr::get(
          outerLvl.ivType,
          // Re-derive the original rhs: for slt/ult/ne it's ubNormalized;
          // for sle/ule the normalization added 1, so subtract it to
          // recover the original rhs.
          (outerLvl.origPred == arith::CmpIPredicate::sle ||
           outerLvl.origPred == arith::CmpIPredicate::ule)
              ? outerLvl.ubNormalized - APInt(outerLvl.ivType.getWidth(), 1)
              : outerLvl.ubNormalized);
      outerUb = builder.create<arith::ConstantOp>(loc, outerLvl.ivType, ubAttr);
    }
    Value cond = builder.create<arith::CmpIOp>(
        loc, outerLvl.origPred, beforeBlock->getArgument(0), outerUb);

    // Runtime INNER levels contribute their compare too. On every
    // non-entry iteration a wrapped IV is back at lb (< ub whenever the
    // loop is live), so the extra terms only bite at ENTRY — which is
    // exactly the zero-trip case: with cols == 0, `j < cols` fails on the
    // first evaluation and the flattened loop executes nothing, matching
    // the original perfect nest. Constant inner levels keep their static
    // non-zero-trip guarantee and need no term.
    for (unsigned k = 1; k < N; ++k) {
      const NestLevel &lvl = nest[k];
      if (!lvl.ubValue)
        continue;
      Value term = builder.create<arith::CmpIOp>(
          loc, lvl.origPred, beforeBlock->getArgument(k), lvl.ubValue);
      cond = builder.create<arith::AndIOp>(loc, cond, term);
    }

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
        if (isBodyPreOp(op))
          builder.clone(*op, mapping);

    // A load-initialized chain starts each pass from its init, re-loaded
    // (with its pure ops) on the pass's first iteration, every level below
    // the origin at its lower bound; the other iterations read the carried
    // value. The load KEEPS its loopschedule.name: the memory-dependence
    // records that reference it must resolve to it.
    for (unsigned c = 0; c < C; ++c) {
      const CarriedChain &chain = info.chains[c];
      if (!chain.initLoad)
        continue;
      Value fresh;
      for (unsigned k = chain.originLevel + 1; k < N; ++k) {
        Value lb = nest[k].lbValue;
        if (!lb)
          lb = builder.create<arith::ConstantOp>(
              loc, nest[k].ivType,
              IntegerAttr::get(nest[k].ivType, nest[k].lb));
        Value atLb = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, afterBlock->getArgument(k), lb);
        fresh = fresh ? builder.create<arith::AndIOp>(loc, fresh, atLb)
                      : atLb;
      }
      auto ifOp = builder.create<scf::IfOp>(
          loc, TypeRange{chain.init.getType()}, fresh,
          /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        builder.clone(*chain.initLoad, mapping);
        for (Operation *op : chain.initOps)
          builder.clone(*op, mapping);
        builder.create<scf::YieldOp>(loc,
                                     mapping.lookupOrDefault(chain.init));
      }
      {
        OpBuilder::InsertionGuard g2(builder);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc,
                                     afterBlock->getArgument(N + M + c));
      }
      mapping.map(chain.innerAfterArg, ifOp.getResult(0));
    }

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

      // done_k = cmpi wrapPred(iv_next, ub) — against the held SSA value
      // for a runtime bound (it dominates the nest, so it dominates the
      // flattened loop).
      Value ubVal;
      if (lvl.ubValue) {
        ubVal = lvl.ubValue;
      } else {
        ubVal = builder.create<arith::ConstantOp>(
            loc, lvl.ivType, IntegerAttr::get(lvl.ivType, lvl.ubNormalized));
      }
      Value reached = builder.create<arith::CmpIOp>(
          loc, wrapPredicate(lvl.origPred), ivNext, ubVal);

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

    // --- Chain yields: a constant/invariant-init chain resets to its init
    // when the levels below the origin wrapped (the next iteration starts a
    // fresh pass); a load-initialized one carries its value on (the next
    // pass re-loads its init, above); origin -1 chains never reset. ---
    SmallVector<Value> chainOut(C);
    for (unsigned c = 0; c < C; ++c) {
      const CarriedChain &chain = info.chains[c];
      if (chain.originLevel < 0 || chain.initLoad) {
        chainOut[c] = chainNext[c];
        continue;
      }
      Value reset = doneAtLevel[chain.originLevel + 1];
      chainOut[c] = builder.create<arith::SelectOp>(loc, reset, chainInits[c],
                                                    chainNext[c]);
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
    hoistInvariantPreOps(op);
    auto nest = collectNest(op);
    if (!nest)
      continue;
    for (NestLevel &lvl : nest->levels)
      consumed.insert(lvl.loop.getOperation());
    emitFlattenedNest(builder, *nest);
    applyDependences(*nest);
  }
}

namespace circt {
std::unique_ptr<mlir::Pass> createSCFWhileLoopFlatteningPass() {
  return std::make_unique<SCFWhileLoopFlatteningPass>();
}
} // namespace circt
