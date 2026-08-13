//===- BitwidthReductionForLoopSchedule.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "circt/Analysis/NameAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <cstdint>
#include <limits>

using namespace mlir;
namespace circt {
namespace loopschedule {
#define GEN_PASS_DEF_BITWIDTHREDUCTIONFORLOOPSCHEDULE
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"
} // namespace loopschedule
} // namespace circt

using namespace mlir::affine;
using namespace mlir::arith;
using namespace circt;
using namespace circt::loopschedule;

namespace {
struct BitwidthReductionForLoopSchedule
    : public circt::loopschedule::impl::BitwidthReductionForLoopScheduleBase<
          BitwidthReductionForLoopSchedule> {
  using circt::loopschedule::impl::BitwidthReductionForLoopScheduleBase<
      BitwidthReductionForLoopSchedule>::BitwidthReductionForLoopScheduleBase;
  void runOnOperation() override;
};
} // namespace

/// True if `v` is provably non-negative: a non-negative constant, a
/// zero-extension, a sign-extension of a non-negative value, the induction
/// variable of an `scf.for` whose constant lower bound is non-negative (with a
/// positive constant step the IV never goes below its lower bound), or an
/// add / multiply / left-shift composition of those — the shapes address
/// computations take after tiling, step normalization and linearization.
/// The composed cases assume the arithmetic does not wrap, the same
/// in-bounds assumption the address-narrowing patterns below already make.
static bool isProvablyNonNegative(Value v) {
  if (auto ext = v.getDefiningOp<ExtUIOp>())
    return ext.getIn().getType().getIntOrFloatBitWidth() <
           ext.getOut().getType().getIntOrFloatBitWidth();
  if (auto ext = v.getDefiningOp<ExtSIOp>())
    return isProvablyNonNegative(ext.getIn());
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return cst.isNonNegative();
  if (auto add = v.getDefiningOp<AddIOp>())
    return isProvablyNonNegative(add.getLhs()) &&
           isProvablyNonNegative(add.getRhs());
  if (auto mul = v.getDefiningOp<MulIOp>())
    return isProvablyNonNegative(mul.getLhs()) &&
           isProvablyNonNegative(mul.getRhs());
  if (auto shl = v.getDefiningOp<ShLIOp>())
    return isProvablyNonNegative(shl.getLhs()) &&
           isProvablyNonNegative(shl.getRhs());
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
    if (forOp && arg == forOp.getInductionVar()) {
      APInt lb, step;
      return matchPattern(forOp.getLowerBound(), m_ConstantInt(&lb)) &&
             lb.isNonNegative() &&
             matchPattern(forOp.getStep(), m_ConstantInt(&step)) &&
             step.isStrictlyPositive();
    }
  }
  return false;
}

/// Match `idx` as `iv * S + r` with `r` provably non-negative and `S` a
/// positive compile-time scale built from muli / shli by constants. Returns
/// S on match. This is what an IV looks like inside a linearized 2-D index
/// (`row * 128 + k` spelled `shl(row, 7) + k`) or after step normalization
/// rewrites a tiled loop to a unit-step tile counter (`t * 16 + i1`).
static std::optional<uint64_t> matchScaledIV(Value idx, Value iv) {
  if (idx == iv)
    return 1;
  if (auto add = idx.getDefiningOp<AddIOp>()) {
    if (auto s = matchScaledIV(add.getLhs(), iv))
      if (isProvablyNonNegative(add.getRhs()))
        return s;
    if (auto s = matchScaledIV(add.getRhs(), iv))
      if (isProvablyNonNegative(add.getLhs()))
        return s;
    return std::nullopt;
  }
  if (auto mul = idx.getDefiningOp<MulIOp>()) {
    APInt c;
    Value other;
    if (matchPattern(mul.getRhs(), m_ConstantInt(&c)))
      other = mul.getLhs();
    else if (matchPattern(mul.getLhs(), m_ConstantInt(&c)))
      other = mul.getRhs();
    else
      return std::nullopt;
    if (!c.isStrictlyPositive() || c.getActiveBits() > 31)
      return std::nullopt;
    if (auto s = matchScaledIV(other, iv)) {
      uint64_t scale = *s * c.getZExtValue();
      if (scale >> 31)
        return std::nullopt;
      return scale;
    }
    return std::nullopt;
  }
  if (auto shl = idx.getDefiningOp<ShLIOp>()) {
    APInt c;
    if (!matchPattern(shl.getRhs(), m_ConstantInt(&c)) ||
        c.getZExtValue() > 30)
      return std::nullopt;
    if (auto s = matchScaledIV(shl.getLhs(), iv)) {
      uint64_t scale = *s << c.getZExtValue();
      if (scale >> 31)
        return std::nullopt;
      return scale;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// Strip this pass's own bound rewrite: `select(ub < 0, 0, trunci(ub))`
/// roots at `ub`. Lets an already-narrowed sibling loop's bound be compared
/// against another loop's still-wide bound by identity.
static Value rootBound(Value v) {
  auto sel = v.getDefiningOp<SelectOp>();
  if (!sel)
    return v;
  auto cmp = sel.getCondition().getDefiningOp<CmpIOp>();
  auto tr = sel.getFalseValue().getDefiningOp<TruncIOp>();
  APInt zero, zeroT;
  if (cmp && tr && cmp.getPredicate() == CmpIPredicate::slt &&
      matchPattern(cmp.getRhs(), m_ConstantInt(&zero)) && zero.isZero() &&
      matchPattern(sel.getTrueValue(), m_ConstantInt(&zeroT)) &&
      zeroT.isZero() && cmp.getLhs() == tr.getIn())
    return tr.getIn();
  return v;
}

/// Collect the blocks guaranteed to execute at least once per iteration of
/// the loop owning `block`: the body block itself plus, recursively, the
/// bodies of nested `scf.for` that provably run at least one iteration —
/// constant bounds with a positive trip count, or a runtime bound that is
/// the SAME value as the licensing loop's own bound with a lower bound no
/// later than its (then `lb_outer < ub` at entry implies `lb <= lb_outer <
/// ub`, so the nested loop enters whenever the outer one did). Anything
/// else (`scf.if`, unrelated runtime bounds, `scf.while`) is deliberately
/// not entered: an access that may not execute licenses nothing.
static void collectGuaranteedBlocks(Block *block, const APInt &outerLb,
                                    Value outerUbRoot,
                                    SmallVectorImpl<Block *> &out) {
  out.push_back(block);
  for (Operation &op : *block) {
    if (auto nested = dyn_cast<scf::ForOp>(&op)) {
      APInt lb, ub, step;
      if (!matchPattern(nested.getLowerBound(), m_ConstantInt(&lb)) ||
          !matchPattern(nested.getStep(), m_ConstantInt(&step)) ||
          !step.isStrictlyPositive())
        continue;
      if (matchPattern(nested.getUpperBound(), m_ConstantInt(&ub))) {
        if (lb.slt(ub))
          collectGuaranteedBlocks(nested.getBody(), outerLb, outerUbRoot, out);
        continue;
      }
      if (rootBound(nested.getUpperBound()) == outerUbRoot &&
          lb.getSExtValue() <= outerLb.getSExtValue())
        collectGuaranteedBlocks(nested.getBody(), lb, outerUbRoot, out);
    }
  }
}

/// Infer an exclusive upper bound on the values the IV of `loop` takes in any
/// well-defined execution, from the loop's own memory accesses: an access
/// indexed by the IV (or by IV + a non-negative value) that is guaranteed to
/// execute on every iteration is out of bounds — undefined behavior — for any
/// iterate at or past its dimension size, so every well-defined execution
/// keeps the IV below that dimension. This is the same license the
/// address-narrowing patterns below already rely on when they truncate
/// indices to the dimension width.
/// Invoke `fn(index, dim)` for every index operand of a memory access `op`
/// whose dimension has a known static extent — memref dims for the
/// loopschedule ops, `1 << getDimBitwidth` for interface ops (Log2Ceil of the
/// extent, so the shift over-approximates: a weaker but still sound license).
static void forEachAccessIndex(Operation *op,
                               llvm::function_ref<void(Value, uint64_t)> fn) {
  if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
    auto memTy = load.getMemRefType();
    for (auto [i, idx] : llvm::enumerate(load.getIndices()))
      if (!memTy.isDynamicDim(i))
        fn(idx, memTy.getDimSize(i));
  } else if (auto store = dyn_cast<LoopScheduleStoreOp>(op)) {
    auto memTy = store.getMemRefType();
    for (auto [i, idx] : llvm::enumerate(store.getIndices()))
      if (!memTy.isDynamicDim(i))
        fn(idx, memTy.getDimSize(i));
  } else if (auto load = dyn_cast<LoadInterface>(op)) {
    for (auto [i, idx] : llvm::enumerate(load.getIndices())) {
      int64_t bw = load.getDimBitwidth(i);
      if (bw > 0 && bw < 48)
        fn(idx, uint64_t(1) << bw);
    }
  } else if (auto store = dyn_cast<StoreInterface>(op)) {
    for (auto [i, idx] : llvm::enumerate(store.getIndices())) {
      int64_t bw = store.getDimBitwidth(i);
      if (bw > 0 && bw < 48)
        fn(idx, uint64_t(1) << bw);
    }
  }
}

static std::optional<uint64_t> licensedIVBound(scf::ForOp loop) {
  Value iv = loop.getInductionVar();

  std::optional<uint64_t> bound;
  // An access at `iv * S + r`, r >= 0, is out of bounds once
  // iv * S >= dim, so every well-defined iterate satisfies
  // iv <= (dim - 1) / S.
  auto considerIdx = [&](Value idx, uint64_t dim) {
    auto s = matchScaledIV(idx, iv);
    if (!s || dim == 0)
      return;
    uint64_t b = (dim - 1) / *s + 1;
    bound = bound ? std::min(*bound, b) : b;
  };

  APInt outerLb;
  if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&outerLb)))
    return std::nullopt;
  SmallVector<Block *> blocks;
  collectGuaranteedBlocks(loop.getBody(), outerLb,
                          rootBound(loop.getUpperBound()), blocks);
  for (Block *block : blocks)
    for (Operation &op : *block)
      forEachAccessIndex(&op, considerIdx);
  return bound;
}

/// Infer a bound on the VALUE of `loop`'s runtime upper bound from accesses
/// that execute whenever the loop does: every op directly in an ANCESTOR
/// block runs in any invocation that reaches the loop (structured control
/// flow completes its blocks), so an index operand there of the form
/// `ub * S + r`, r >= 0, against extent D keeps `ub <= (D - 1) / S` in every
/// well-defined execution. This licenses a loop whose IV never reaches
/// memory but whose bound rides a hoisted burst request as a length or
/// repeat count — the AXI stream shape. Returned as an exclusive bound on
/// the iterates (iterate < ub <= bound), compatible with licensedIVBound's.
static std::optional<uint64_t> licensedUbBound(scf::ForOp loop) {
  Value ubRoot = rootBound(loop.getUpperBound());
  std::optional<uint64_t> bound;
  auto considerIdx = [&](Value idx, uint64_t dim) {
    auto s = matchScaledIV(idx, ubRoot);
    if (!s || dim == 0)
      return;
    uint64_t b = (dim - 1) / *s;
    if (b == 0)
      return;
    bound = bound ? std::min(*bound, b) : b;
  };

  Operation *cur = loop;
  while (Block *block = cur->getBlock()) {
    for (Operation &op : *block)
      if (&op != loop)
        forEachAccessIndex(&op, considerIdx);
    cur = block->getParentOp();
    if (!cur || isa<func::FuncOp>(cur))
      break;
  }
  return bound;
}

struct SCFForIterationReduction : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto constantLB = op.getLowerBound().getDefiningOp<ConstantOp>();
    auto constantUB = op.getUpperBound().getDefiningOp<ConstantOp>();
    auto constantStep = op.getStep().getDefiningOp<ConstantOp>();
    if (constantLB == nullptr || constantUB == nullptr ||
        constantStep == nullptr) {
      // A loop with a RUNTIME upper bound. The IV's width is inferred from
      // the loop's own accesses (see licensedIVBound); the bound operand is
      // narrowed under a sign guard:
      //
      //   ub' = (ub < 0) ? 0 : trunc(ub)
      //
      // A negative bound is a well-defined zero-trip loop truncation would
      // break, so it is forced to zero (lb >= 0 makes that zero-trip). Any
      // non-negative bound the truncation changes implies an iterate at or
      // past the licensed dimension executes its access, so the original
      // execution is undefined and the narrowed behavior is legal.
      auto wideTy = dyn_cast<IntegerType>(op.getInductionVar().getType());
      if (!wideTy)
        return failure();
      APInt lbCst, stepCst;
      if (!matchPattern(op.getLowerBound(), m_ConstantInt(&lbCst)) ||
          lbCst.isNegative() || lbCst.getActiveBits() > 32)
        return failure();
      if (!matchPattern(op.getStep(), m_ConstantInt(&stepCst)) ||
          !stepCst.isStrictlyPositive() || stepCst.getActiveBits() > 32)
        return failure();
      auto bound = licensedIVBound(op);
      if (auto ubBound = licensedUbBound(op))
        bound = bound ? std::min(*bound, *ubBound) : ubBound;
      if (!bound)
        return failure();
      uint64_t lb = lbCst.getZExtValue();
      uint64_t step = stepCst.getZExtValue();
      // A lower bound at or past the dimension makes zero-trip the only
      // well-defined behavior, which truncation cannot preserve.
      if (lb >= *bound)
        return failure();
      // Signed width holding every well-defined value: iterates stay below
      // `bound`, so a well-defined upper bound is at most bound + step - 1
      // and the final IV update reaches at most bound - 1 + step.
      uint64_t maxVal = *bound + step - 1;
      unsigned bitwidth = llvm::Log2_64_Ceil(maxVal + 1) + 1;
      if (bitwidth >= wideTy.getWidth())
        return failure();
      auto newType = rewriter.getIntegerType(bitwidth);

      // Place the narrowed bound at its operand's definition point (the
      // function entry block for a kernel argument), so it is evaluated
      // once rather than per entry of an enclosing loop.
      auto ub = op.getUpperBound();
      if (auto barg = dyn_cast<BlockArgument>(ub))
        rewriter.setInsertionPointToStart(barg.getOwner());
      else
        rewriter.setInsertionPointAfter(ub.getDefiningOp());
      auto loc = op.getLoc();
      auto zeroWide = ConstantOp::create(rewriter, loc,
                                         rewriter.getIntegerAttr(wideTy, 0));
      auto isNeg = CmpIOp::create(rewriter, loc, CmpIPredicate::slt, ub,
                                  zeroWide);
      auto zeroNarrow = ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(newType, 0));
      auto ubTrunc = TruncIOp::create(rewriter, loc, newType, ub);
      auto ubNarrow =
          SelectOp::create(rewriter, loc, isNeg, zeroNarrow, ubTrunc);

      rewriter.setInsertionPoint(op);
      op.setUpperBound(ubNarrow);
      op.setLowerBound(ConstantOp::create(
          rewriter, loc, rewriter.getIntegerAttr(newType, lb)));
      op.setStep(ConstantOp::create(rewriter, loc,
                                    rewriter.getIntegerAttr(newType, step)));

      auto induction = op.getInductionVar();
      induction.setType(newType);
      rewriter.setInsertionPointToStart(&op.getRegion().front());
      auto newExt =
          arith::ExtSIOp::create(rewriter, op.getLoc(), wideTy, induction);
      rewriter.replaceAllUsesExcept(induction, newExt.getOut(), newExt);
      return success();
    }

    auto upperBoundAttr = dyn_cast<IntegerAttr>(constantUB.getValue());
    auto upperBound = upperBoundAttr.getValue();
    upperBound = upperBound + 1;
    auto bitwidth = upperBound.ceilLogBase2() + 1;

    auto induction = op.getInductionVar();
    auto newType = rewriter.getIntegerType(bitwidth);
    if (induction.getType() == newType)
      return failure();

    // Replace lowerBound, upperBound and step
    auto lbValue = cast<IntegerAttr>(constantLB.getValue()).getInt();
    auto newLBAttr = rewriter.getIntegerAttr(newType, lbValue);
    auto newLB = ConstantOp::create(rewriter, op.getLoc(), newLBAttr);
    op.setLowerBound(newLB);

    auto ubValue = cast<IntegerAttr>(constantUB.getValue()).getInt();
    auto newUBAttr = rewriter.getIntegerAttr(newType, ubValue);
    auto newUB = ConstantOp::create(rewriter, op.getLoc(), newUBAttr);
    op.setUpperBound(newUB);

    auto stepValue = cast<IntegerAttr>(constantStep.getValue()).getInt();
    auto newStepAttr = rewriter.getIntegerAttr(newType, stepValue);
    auto newStep = ConstantOp::create(rewriter, op.getLoc(), newStepAttr);
    op.setStep(newStep);

    induction.setType(newType);
    rewriter.setInsertionPointToStart(&op.getRegion().front());
    auto newExt = arith::ExtSIOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64Type(), induction);
    rewriter.replaceAllUsesExcept(induction, newExt.getOut(), newExt);
    return success();
  }
};

struct SCFForCleanupPattern : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    auto cast = op.getLowerBound().getDefiningOp<UnrealizedConversionCastOp>();
    if (cast != nullptr && cast.getInputs().size() == 1) {
      auto newVal = cast.getInputs().front();
      op.setLowerBound(newVal);
      changed = true;
    }

    cast = op.getUpperBound().getDefiningOp<UnrealizedConversionCastOp>();
    if (cast != nullptr && cast.getInputs().size() == 1) {
      auto newVal = cast.getInputs().front();
      op.setUpperBound(newVal);
      changed = true;
    }

    cast = op.getStep().getDefiningOp<UnrealizedConversionCastOp>();
    if (cast != nullptr && cast.getInputs().size() == 1) {
      auto newVal = cast.getInputs().front();
      op.setStep(newVal);
      changed = true;
    }

    if (!changed)
      return failure();
    return success();
  }
};

struct TruncCleanupPattern : OpRewritePattern<TruncIOp> {
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncIOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getIn()))
      return failure();
    auto *definingOp = op.getIn().getDefiningOp();
    auto outputType = op.getOut().getType();
    Operation *newOp = nullptr;
    if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
      newOp = ExtUIOp::create(rewriter, op.getLoc(), outputType, extUI.getIn());
    } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
      newOp = ExtSIOp::create(rewriter, op.getLoc(), outputType, extSI.getIn());
    }

    if (!newOp)
      return failure();

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct LoadCleanupPattern : OpRewritePattern<LoopScheduleLoadOp> {
  using OpRewritePattern<LoopScheduleLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
        // } else if (auto unreal =
        //                dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        //   if (unreal.getInputs().size() != 1)
        //     continue;
        //   idx.set(unreal.getInputs().front());
        //   updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct StoreCleanupPattern : OpRewritePattern<LoopScheduleStoreOp> {
  using OpRewritePattern<LoopScheduleStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
        // } else if (auto unreal =
        //                dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        //   if (unreal.getInputs().size() != 1)
        //     continue;
        //   idx.set(unreal.getInputs().front());
        //   updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct LoadAddressNarrowingPattern : OpRewritePattern<LoopScheduleLoadOp> {
  using OpRewritePattern<LoopScheduleLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto v : llvm::enumerate(indices)) {
      auto &idx = v.value();
      auto i = v.index();
      auto dimSize = op.getMemRefType().getDimSize(i);
      auto bitwidth = llvm::Log2_64_Ceil(dimSize) + 1;
      auto newType = rewriter.getIntegerType(bitwidth);
      auto oldType = dyn_cast_or_null<IntegerType>(idx.get().getType());
      if (oldType) {
        if (newType.getIntOrFloatBitWidth() < oldType.getIntOrFloatBitWidth()) {
          auto newIdx = arith::TruncIOp::create(rewriter, op.getLoc(), newType,
                                                idx.get());
          idx.set(newIdx);
          updated = true;
        }
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct StoreAddressNarrowingPattern : OpRewritePattern<LoopScheduleStoreOp> {
  using OpRewritePattern<LoopScheduleStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto v : llvm::enumerate(indices)) {
      auto &idx = v.value();
      auto i = v.index();
      auto dimSize = op.getMemRefType().getDimSize(i);
      auto bitwidth = llvm::Log2_64_Ceil(dimSize) + 1;
      auto newType = rewriter.getIntegerType(bitwidth);
      auto oldType = dyn_cast_or_null<IntegerType>(idx.get().getType());
      if (oldType) {
        if (newType.getIntOrFloatBitWidth() < oldType.getIntOrFloatBitWidth()) {
          auto newIdx = arith::TruncIOp::create(rewriter, op.getLoc(), newType,
                                                idx.get());
          idx.set(newIdx);
          updated = true;
        }
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct LoadInterfaceCleanupPattern : OpInterfaceRewritePattern<LoadInterface> {
  using OpInterfaceRewritePattern<LoadInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoadInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
        // } else if (auto unreal =
        //                dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        //   if (unreal.getInputs().size() != 1)
        //     continue;
        //   idx.set(unreal.getInputs().front());
        //   updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct StoreInterfaceCleanupPattern
    : OpInterfaceRewritePattern<StoreInterface> {
  using OpInterfaceRewritePattern<StoreInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(StoreInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
        // } else if (auto unreal =
        //                dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        //   if (unreal.getInputs().size() != 1)
        //     continue;
        //   idx.set(unreal.getInputs().front());
        //   updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct LoadInterfaceAddressNarrowingPattern
    : OpInterfaceRewritePattern<LoadInterface> {
  using OpInterfaceRewritePattern<LoadInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoadInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto v : llvm::enumerate(indices)) {
      auto &idx = v.value();
      auto i = v.index();
      auto bitwidth = op.getDimBitwidth(i) + 1;
      auto newType = rewriter.getIntegerType(bitwidth);
      auto oldType = dyn_cast_or_null<IntegerType>(idx.get().getType());
      if (oldType) {
        if (newType.getIntOrFloatBitWidth() < oldType.getIntOrFloatBitWidth()) {
          auto newIdx = arith::TruncIOp::create(rewriter, op.getLoc(), newType,
                                                idx.get());
          idx.set(newIdx);
          updated = true;
        }
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct StoreInterfaceAddressNarrowingPattern
    : OpInterfaceRewritePattern<StoreInterface> {
  using OpInterfaceRewritePattern<StoreInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(StoreInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto v : llvm::enumerate(indices)) {
      auto &idx = v.value();
      auto i = v.index();
      auto bitwidth = op.getDimBitwidth(i) + 1;
      auto newType = rewriter.getIntegerType(bitwidth);
      auto oldType = dyn_cast_or_null<IntegerType>(idx.get().getType());
      if (oldType) {
        if (newType.getIntOrFloatBitWidth() < oldType.getIntOrFloatBitWidth()) {
          auto newIdx = arith::TruncIOp::create(rewriter, op.getLoc(), newType,
                                                idx.get());
          idx.set(newIdx);
          updated = true;
        }
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

template <typename T>
struct ImplicitTruncPattern : OpRewritePattern<TruncIOp> {
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncIOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getIn())) {
      return failure();
    }

    auto *inputOp = op.getIn().getDefiningOp();
    if (!isa<T>(inputOp)) {
      return failure();
    }

    auto users = inputOp->getUsers();

    // if (std::distance(users.begin(), users.end()) > 1) {
    //   return failure();
    // }

    auto newType = op.getOut().getType();

    SmallVector<Value> newOperands;

    rewriter.setInsertionPoint(inputOp);
    for (auto &operand : inputOp->getOpOperands()) {
      auto val = operand.get();
      auto newOperand = TruncIOp::create(rewriter, op.getLoc(), newType, val);
      newOperands.push_back(newOperand);
    }

    rewriter.replaceOpWithNewOp<T>(op, newOperands);

    return success();
  }
};

namespace {
class DataFlowListener : public RewriterBase::Listener {
public:
  DataFlowListener(DataFlowSolver &s) : s(s) {}

protected:
  void notifyOperationErased(Operation *op) override {
    s.eraseState(s.getProgramPointAfter(op));
    for (Value res : op->getResults())
      s.eraseState(res);
  }

  DataFlowSolver &s;
};
} // namespace

void BitwidthReductionForLoopSchedule::runOnOperation() {
  auto op = getOperation();
  auto &context = getContext();

  // Minimize SCFFor iteration argument bitwidth to enable further bitwidth
  // reduction
  RewritePatternSet patterns(&context);
  patterns.add<SCFForIterationReduction>(&context);

  GreedyRewriteConfig config;
  if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  // Apply the core integer narrowing pass
  patterns.clear();
  SmallVector<unsigned> bitwidthsSupported;
  for (unsigned i = 1; i <= 1024; ++i) {
    bitwidthsSupported.push_back(i);
  }

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  DataFlowListener listener(solver);

  populateIntRangeNarrowingPatterns(patterns, solver, bitwidthsSupported);

  GreedyRewriteConfig narrowingConfig;
  // We specifically need bottom-up traversal as cmpi pattern needs range
  // data, attached to its original argument values.
  narrowingConfig.setUseTopDownTraversal(false);
  narrowingConfig.setListener(&listener);

  if (failed(applyPatternsGreedily(op, std::move(patterns), narrowingConfig))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);
  patterns.add<LoadAddressNarrowingPattern>(&context);
  patterns.add<StoreAddressNarrowingPattern>(&context);
  patterns.add<LoadInterfaceCleanupPattern>(&context);
  patterns.add<StoreInterfaceCleanupPattern>(&context);
  patterns.add<LoadInterfaceAddressNarrowingPattern>(&context);
  patterns.add<StoreInterfaceAddressNarrowingPattern>(&context);
  // patterns.add<ImplicitTruncPattern<MulIOp>>(&context);
  // patterns.add<ImplicitTruncPattern<DivSIOp>>(&context);
  // patterns.add<ImplicitTruncPattern<DivUIOp>>(&context);
  // patterns.add<ImplicitTruncPattern<AddIOp>>(&context);
  // patterns.add<ImplicitTruncPattern<SubIOp>>(&context);

  if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  // Perform dead code elimination
  mlir::IRRewriter rewriter(&context);
  (void)mlir::runRegionDCE(rewriter, op->getRegions());
}

namespace circt {
namespace loopschedule {
std::unique_ptr<mlir::Pass> createBitwidthReductionForLoopSchedulePass() {
  return std::make_unique<BitwidthReductionForLoopSchedule>();
}
} // namespace loopschedule
} // namespace circt
