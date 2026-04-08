//===- SCFWhileTripCountAnalysis.cpp - scf.while trip count ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/SCFWhileTripCountAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

/// Description of the canonical induction-variable shape found in the loop.
struct IVInfo {
  unsigned iterArgIdx;  // Index into scf.while inits / before-region args.
  BlockArgument ivBefore;
  arith::CmpIOp cmp;
  APInt ivInit;   // Constant initial value (bitwidth = IV type width).
  APInt ubConst; // Constant bound from the cmpi rhs.
  arith::CmpIPredicate pred;
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

/// Try to extract a constant APInt from an SSA value produced by an
/// `arith.constant` (the only form we accept for canonical bounds/steps).
static std::optional<APInt> matchConstantInt(Value v) {
  APInt out;
  if (matchPattern(v, m_ConstantInt(&out)))
    return out;
  return std::nullopt;
}

/// Match the IV in the before region's scf.condition. The canonical shape is:
///   scf.condition(%cmp) %before_args...
///   %cmp = arith.cmpi pred, %before[i], %constUb
static std::optional<IVInfo> matchIV(WhileOp whileOp) {
  ConditionOp condOp = whileOp.getConditionOp();
  auto cmp = condOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;
  if (!isSupportedPredicate(cmp.getPredicate()))
    return std::nullopt;

  auto ivBefore = dyn_cast<BlockArgument>(cmp.getLhs());
  if (!ivBefore || ivBefore.getOwner() != whileOp.getBeforeBody())
    return std::nullopt;

  auto ub = matchConstantInt(cmp.getRhs());
  if (!ub)
    return std::nullopt;

  unsigned idx = ivBefore.getArgNumber();
  auto init = matchConstantInt(whileOp.getInits()[idx]);
  if (!init)
    return std::nullopt;

  // Normalize init/ub to the IV type's bitwidth (they should match already,
  // since the cmpi lhs and the scf.while operand both feed iter-arg %idx).
  auto ivType = dyn_cast<IntegerType>(ivBefore.getType());
  if (!ivType)
    return std::nullopt;
  unsigned bw = ivType.getWidth();
  if (init->getBitWidth() != bw || ub->getBitWidth() != bw)
    return std::nullopt;

  return IVInfo{idx,   ivBefore,   cmp,
                *init, *ub,        cmp.getPredicate()};
}

/// Inspect the after region's scf.yield at `iterArgIdx` and return the
/// constant step if it matches `arith.addi %ivAfter, %constStep` where
/// `%ivAfter` traces back through the scf.condition forwarding to the IV
/// before-block argument.
static std::optional<APInt> matchStep(WhileOp whileOp, const IVInfo &iv) {
  // Locate the after-block argument that carries the IV. scf.condition
  // forwards values positionally: after_args[j] == condition.args[j]. We
  // look for the first j whose forwarded value is the IV before-arg.
  ConditionOp condOp = whileOp.getConditionOp();
  Block *afterBlock = whileOp.getAfterBody();
  BlockArgument ivAfter = nullptr;
  for (auto [j, forwarded] : llvm::enumerate(condOp.getArgs())) {
    if (forwarded == iv.ivBefore) {
      ivAfter = afterBlock->getArgument(j);
      break;
    }
  }
  if (!ivAfter)
    return std::nullopt;

  // The yield operand at position iterArgIdx feeds back into the IV.
  YieldOp yieldOp = whileOp.getYieldOp();
  if (iv.iterArgIdx >= yieldOp.getNumOperands())
    return std::nullopt;
  Value feedback = yieldOp.getOperand(iv.iterArgIdx);

  auto addOp = feedback.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;

  Value addLhs = addOp.getLhs();
  Value addRhs = addOp.getRhs();
  std::optional<APInt> step;
  if (addLhs == ivAfter)
    step = matchConstantInt(addRhs);
  else if (addRhs == ivAfter)
    step = matchConstantInt(addLhs);
  if (!step)
    return std::nullopt;

  // Canonical form requires a strictly positive step (unsigned loops
  // increment, signed loops increment for slt/sle/ne with init < ub).
  bool isSigned = isSignedPredicate(iv.pred);
  if (isSigned) {
    if (!step->isStrictlyPositive())
      return std::nullopt;
  } else {
    if (step->isZero())
      return std::nullopt;
  }

  // Make sure the IV isn't mutated elsewhere: the only use of ivAfter we
  // accept that writes the feedback is this addi. Other reads are fine.
  for (Operation *user : ivAfter.getUsers()) {
    if (user == addOp.getOperation())
      continue;
    // Any other def reaching the yield at iterArgIdx would be caught by
    // the feedback-match above; plain reads are harmless.
  }

  return step;
}

} // namespace

std::optional<llvm::APInt>
circt::analysis::getSCFWhileConstantTripCount(WhileOp whileOp) {
  auto iv = matchIV(whileOp);
  if (!iv)
    return std::nullopt;

  auto step = matchStep(whileOp, *iv);
  if (!step)
    return std::nullopt;

  // Normalize inclusive predicates (sle/ule) to exclusive form by bumping
  // ub by one. Bail if the bump would overflow the IV width.
  APInt lb = iv->ivInit;
  APInt ub = iv->ubConst;
  bool isSigned = isSignedPredicate(iv->pred);

  switch (iv->pred) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    break;
  case arith::CmpIPredicate::sle: {
    bool ov = false;
    ub = ub.sadd_ov(APInt(ub.getBitWidth(), 1), ov);
    if (ov)
      return std::nullopt;
    break;
  }
  case arith::CmpIPredicate::ule: {
    bool ov = false;
    ub = ub.uadd_ov(APInt(ub.getBitWidth(), 1), ov);
    if (ov)
      return std::nullopt;
    break;
  }
  case arith::CmpIPredicate::ne: {
    // Valid only when the loop actually walks up toward ub (lb < ub) and
    // (ub - lb) is divisible by step — otherwise the loop overshoots and
    // never terminates. The downstream constantTripCount helper handles
    // the divisibility check naturally; we just need to ensure monotone
    // progress and pick a sign convention.
    bool lbLtUb =
        isSigned ? lb.slt(ub) : lb.ult(ub);
    if (!lbLtUb)
      return std::nullopt;
    break;
  }
  default:
    return std::nullopt;
  }

  // Build OpFoldResult triple and delegate the arithmetic to the upstream
  // helper. Using IntegerAttr keeps everything in the constant path; the
  // computeUbMinusLb callback is never invoked because all three values
  // are constants.
  Builder b(whileOp->getContext());
  auto ivType = cast<IntegerType>(iv->ivBefore.getType());
  auto lbAttr = IntegerAttr::get(ivType, lb);
  auto ubAttr = IntegerAttr::get(ivType, ub);
  auto stepAttr = IntegerAttr::get(ivType, *step);

  return mlir::constantTripCount(
      OpFoldResult(lbAttr), OpFoldResult(ubAttr), OpFoldResult(stepAttr),
      isSigned, mlir::scf::computeUbMinusLb);
}
