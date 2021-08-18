//===- CombFolds.cpp - Folds + Canonicalization for Comb operations -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace comb;

using mlir::constFoldBinaryOp;

static Attribute getIntAttr(const APInt &value, MLIRContext *context) {
  return IntegerAttr::get(IntegerType::get(context, value.getBitWidth()),
                          value);
}

namespace {
struct ConstantIntMatcher {
  APInt &value;
  ConstantIntMatcher(APInt &value) : value(value) {}
  bool match(Operation *op) {
    if (auto cst = dyn_cast<hw::ConstantOp>(op)) {
      value = cst.value();
      return true;
    }
    return false;
  }
};
} // end anonymous namespace

static inline ConstantIntMatcher m_RConstant(APInt &value) {
  return ConstantIntMatcher(value);
}

/// Flattens a single input in `op` if `hasOneUse` is true and it can be defined
/// as an Op. Returns true if successful, and false otherwise.
///
/// Example: op(1, 2, op(3, 4), 5) -> op(1, 2, 3, 4, 5)  // returns true
///
template <typename Op>
static bool tryFlatteningOperands(Op op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();

  for (size_t i = 0, size = inputs.size(); i != size; ++i) {
    // Don't duplicate logic.
    if (!inputs[i].hasOneUse())
      continue;
    auto flattenOp = inputs[i].template getDefiningOp<Op>();
    if (!flattenOp)
      continue;
    auto flattenOpInputs = flattenOp.inputs();

    SmallVector<Value, 4> newOperands;
    newOperands.reserve(size + flattenOpInputs.size());

    auto flattenOpIndex = inputs.begin() + i;
    newOperands.append(inputs.begin(), flattenOpIndex);
    newOperands.append(flattenOpInputs.begin(), flattenOpInputs.end());
    newOperands.append(flattenOpIndex + 1, inputs.end());

    rewriter.replaceOpWithNewOp<Op>(op, op.getType(), newOperands);
    return true;
  }
  return false;
}

// Given a range of uses of an operation, find the lowest and highest bits
// inclusive that are ever referenced. The range of uses must not be empty.
static std::pair<size_t, size_t>
getLowestBitAndHighestBitRequired(Operation *op, bool narrowTrailingBits,
                                  size_t originalOpWidth) {
  auto users = op->getUsers();
  assert(!users.empty() &&
         "getLowestBitAndHighestBitRequired cannot operate on "
         "a empty list of uses.");

  // when we don't want to narrowTrailingBits (namely in arithmetic
  // operations), forcing lowestBitRequired = 0
  size_t lowestBitRequired = narrowTrailingBits ? originalOpWidth - 1 : 0;
  size_t highestBitRequired = 0;

  for (auto *user : users) {
    if (auto extractOp = dyn_cast<ExtractOp>(user)) {
      size_t lowBit = extractOp.lowBit();
      size_t highBit = extractOp.getType().getWidth() + lowBit - 1;
      highestBitRequired = std::max(highestBitRequired, highBit);
      lowestBitRequired = std::min(lowestBitRequired, lowBit);
      continue;
    }

    highestBitRequired = originalOpWidth - 1;
    lowestBitRequired = 0;
    break;
  }

  return {lowestBitRequired, highestBitRequired};
}

// narrowOperationWidth(...) attempts to create a new instance of Op
// with createOp using narrowed operands. Ie: transforming f(x, y) into
// f(x[n:m], y[n:m]), It analyzes all usage sites of narrowingCandidate and
// mutate them to the newly created narrowed version if it's benefitial.
//
// narrowTrailingBits determines whether undemanded trailing bits should
// be aggressively stripped off.
//
// This function requires the narrowingCandidate to have at least 1 use.
//
// Returns true if IR is mutated. IR is mutated iff narrowing happens.
static bool
narrowOperationWidth(Operation *narrowingCandidate, ValueRange inputs,
                     bool narrowTrailingBits, PatternRewriter &rewriter,
                     function_ref<Value(ArrayRef<Value>)> createOp) {
  // If the result is never used, no point optimizing this. It will
  // also complicated error handling in getLowestBitAndHigestBitRequired.
  assert(!narrowingCandidate->getUsers().empty() &&
         "narrowingCandidate must have at least one use.");
  assert(narrowingCandidate->getNumResults() == 1 &&
         "narrowingCandidate must have exactly one result");
  IntegerType narrowingCandidateType =
      narrowingCandidate->getResultTypes().front().cast<IntegerType>();

  size_t highestBitRequired;
  size_t lowestBitRequired;
  size_t originalOpWidth = narrowingCandidateType.getIntOrFloatBitWidth();

  std::tie(lowestBitRequired, highestBitRequired) =
      getLowestBitAndHighestBitRequired(narrowingCandidate, narrowTrailingBits,
                                        originalOpWidth);

  // Give up, because we can't make it narrower than it already is.
  if (lowestBitRequired == 0 && highestBitRequired == originalOpWidth - 1)
    return false;

  auto loc = narrowingCandidate->getLoc();
  size_t narrowedWidth = highestBitRequired - lowestBitRequired + 1;
  auto narrowedType = rewriter.getIntegerType(narrowedWidth);

  // Insert the new narrowedOperation at a point where all narrowingCandidate's
  // operands are available, and resides before all users.
  rewriter.setInsertionPoint(narrowingCandidate);
  Value narrowedOperation =
      createOp(SmallVector<Value>(llvm::map_range(inputs, [&](auto input) {
        return rewriter.create<ExtractOp>(loc, narrowedType, input,
                                          lowestBitRequired);
      })));

  auto getNarrowedExtractReplacement = [&](ExtractOp extractUse) -> Value {
    auto oldLowBit = extractUse.lowBit();

    assert(oldLowBit >= lowestBitRequired &&
           "incorrectly deduced the lowest bit required in usage arguments.");

    auto loc = extractUse.getLoc();

    if (narrowedWidth == extractUse.getType().getWidth()) {
      return narrowedOperation;
    }

    uint32_t newLowBit = oldLowBit - lowestBitRequired;
    Value narrowedExtract = rewriter.create<ExtractOp>(
        loc, extractUse.getType(), narrowedOperation, newLowBit);
    return narrowedExtract;
  };

  // Replaces all existing use sites with the newly created narrowed operation.
  // This loop captures both the root and non-root use sites.  Note that
  // PatternRewriter allows calling rewriter.replaceOp on non-root users.
  for (Operation *user :
       llvm::make_early_inc_range(narrowingCandidate->getUsers())) {
    auto extractUser = cast<ExtractOp>(user);

    // This is necessary because the rewriter's insertion point can be a
    // replaced extractUser
    rewriter.setInsertionPoint(extractUser);
    rewriter.replaceOp(extractUser, getNarrowedExtractReplacement(extractUser));
  }

  // narrowingCandidate can be safely removed now, as all old users of
  // narrowingCandidate are replaced.
  rewriter.eraseOp(narrowingCandidate);

  return true;
}

template <class Op>
static bool narrowOperationWidth(Op op, ValueRange inputs,
                                 bool narrowTrailingBits,
                                 PatternRewriter &rewriter) {
  auto createOp = [&](ArrayRef<Value> args) -> Value {
    return rewriter.create<Op>(op.getLoc(), args);
  };
  return narrowOperationWidth(op, inputs, narrowTrailingBits, rewriter,
                              createOp);
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

OpFoldResult SExtOp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    auto destWidth = getType().getWidth();
    return getIntAttr(input.getValue().sext(destWidth), getContext());
  }

  if (getType().getWidth() == input().getType().getIntOrFloatBitWidth())
    return input();

  return {};
}

LogicalResult SExtOp::canonicalize(SExtOp op, PatternRewriter &rewriter) {
  // If the sign bit is known, we can fold this to a simpler operation.
  auto knownBits = KnownBitAnalysis::compute(op.input());
  if (knownBits.getBitsKnown().isNegative()) {
    // Ok, we know the sign bit, strength reduce this into a concat of the
    // appropriate constant.
    unsigned numBits =
        op.getType().getWidth() - op.input().getType().getIntOrFloatBitWidth();
    APInt cstBits = knownBits.ones.isNegative()
                        ? APInt::getAllOnesValue(numBits)
                        : APInt(numBits, 0);
    auto signBits = rewriter.create<hw::ConstantOp>(op.getLoc(), cstBits);
    rewriter.replaceOpWithNewOp<ConcatOp>(op, signBits, op.input());
    return success();
  }
  return failure();
}

OpFoldResult ParityOp::fold(ArrayRef<Attribute> constants) {
  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>())
    return getIntAttr(APInt(1, input.getValue().countPopulation() & 1),
                      getContext());

  return {};
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

OpFoldResult ShlOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    unsigned shift = rhs.getValue().getZExtValue();
    unsigned width = getType().getIntOrFloatBitWidth();
    if (shift == 0)
      return getOperand(0);
    if (width <= shift)
      return getIntAttr(APInt::getNullValue(width), getContext());
  }

  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.shl(b); });
}

LogicalResult ShlOp::canonicalize(ShlOp op, PatternRewriter &rewriter) {
  // ShlOp(x, cst) -> Concat(Extract(x), zeros)
  APInt value;
  if (!matchPattern(op.rhs(), m_RConstant(value)))
    return failure();

  unsigned width = op.lhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  // This case is handled by fold.
  if (width <= shift || shift == 0)
    return failure();

  auto zeros =
      rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getNullValue(shift));

  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), shift, width - shift);

  rewriter.replaceOpWithNewOp<ConcatOp>(op, extract, zeros);
  return success();
}

OpFoldResult ShrUOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    unsigned shift = rhs.getValue().getZExtValue();
    if (shift == 0)
      return getOperand(0);

    unsigned width = getType().getIntOrFloatBitWidth();
    if (width <= shift)
      return getIntAttr(APInt::getNullValue(width), getContext());
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.lshr(b); });
}

LogicalResult ShrUOp::canonicalize(ShrUOp op, PatternRewriter &rewriter) {
  // ShrUOp(x, cst) -> Concat(zeros, Extract(x))
  APInt value;
  if (!matchPattern(op.rhs(), m_RConstant(value)))
    return failure();

  unsigned width = op.lhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  // This case is handled by fold.
  if (width <= shift || shift == 0)
    return failure();

  auto zeros =
      rewriter.create<hw::ConstantOp>(op.getLoc(), APInt::getNullValue(shift));

  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), 0, width - shift);

  rewriter.replaceOpWithNewOp<ConcatOp>(op, zeros, extract);
  return success();
}

OpFoldResult ShrSOp::fold(ArrayRef<Attribute> operands) {
  if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
    if (rhs.getValue().getZExtValue() == 0)
      return getOperand(0);
  }
  return constFoldBinaryOp<IntegerAttr>(
      operands, [](APInt a, APInt b) { return a.ashr(b); });
}

LogicalResult ShrSOp::canonicalize(ShrSOp op, PatternRewriter &rewriter) {
  // ShrSOp(x, cst) -> Concat(sext(extract(x, topbit)),extract(x))
  APInt value;
  if (!matchPattern(op.rhs(), m_RConstant(value)))
    return failure();

  unsigned width = op.lhs().getType().cast<IntegerType>().getWidth();
  unsigned shift = value.getZExtValue();

  auto topbit = rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), 0, 1);
  auto sext = rewriter.create<SExtOp>(op.getLoc(),
                                      rewriter.getIntegerType(shift), topbit);

  if (width <= shift) {
    rewriter.replaceOp(op, {sext});
    return success();
  }

  auto extract =
      rewriter.create<ExtractOp>(op.getLoc(), op.lhs(), 0, width - shift);

  rewriter.replaceOpWithNewOp<ConcatOp>(op, sext, extract);
  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

OpFoldResult ExtractOp::fold(ArrayRef<Attribute> constants) {
  // If we are extracting the entire input, then return it.
  if (input().getType() == getType())
    return input();

  // Constant fold.
  if (auto input = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    unsigned dstWidth = getType().getWidth();
    return getIntAttr(input.getValue().lshr(lowBit()).trunc(dstWidth),
                      getContext());
  }
  return {};
}

// Transforms extract(lo, cat(a, b, c, d, e)) into
// cat(extract(lo1, b), c, extract(lo2, d)).
// innerCat must be the argument of the provided ExtractOp.
static LogicalResult extractConcatToConcatExtract(ExtractOp op,
                                                  ConcatOp innerCat,
                                                  PatternRewriter &rewriter) {
  auto reversedConcatArgs = llvm::reverse(innerCat.inputs());
  size_t beginOfFirstRelevantElement = 0;
  auto it = reversedConcatArgs.begin();
  size_t lowBit = op.lowBit();

  // This loop finds the first concatArg that is covered by the ExtractOp
  for (; it != reversedConcatArgs.end(); it++) {
    assert(beginOfFirstRelevantElement <= lowBit &&
           "incorrectly moved past an element that lowBit has coverage over");
    auto operand = *it;

    size_t operandWidth = operand.getType().getIntOrFloatBitWidth();
    if (lowBit < beginOfFirstRelevantElement + operandWidth) {
      // A bit other than the first bit will be used in this element.
      // ...... ........ ...
      //           ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      // Edge-case close to the end of the range.
      // ...... ........ ...
      //                 ^---(position + operandWidth)
      //               ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      // Edge-case close to the beginning of the rang
      // ...... ........ ...
      //        ^---lowBit
      //        ^---beginOfFirstRelevantElement
      //
      break;
    }

    // extraction discards this element.
    // ...... ........  ...
    // |      ^---lowBit
    // ^---beginOfFirstRelevantElement
    beginOfFirstRelevantElement += operandWidth;
  }
  assert(it != reversedConcatArgs.end() &&
         "incorrectly failed to find an element which contains coverage of "
         "lowBit");

  SmallVector<Value> reverseConcatArgs;
  size_t widthRemaining = op.getType().getWidth();
  size_t extractLo = lowBit - beginOfFirstRelevantElement;

  // Transform individual arguments of innerCat(..., a, b, c,) into
  // [ extract(a), b, extract(c) ], skipping an extract operation where
  // possible.
  for (; widthRemaining != 0 && it != reversedConcatArgs.end(); it++) {
    auto concatArg = *it;
    size_t operandWidth = concatArg.getType().getIntOrFloatBitWidth();
    size_t widthToConsume = std::min(widthRemaining, operandWidth - extractLo);

    if (widthToConsume == operandWidth && extractLo == 0) {
      reverseConcatArgs.push_back(concatArg);
    } else {
      auto resultType = IntegerType::get(rewriter.getContext(), widthToConsume);
      reverseConcatArgs.push_back(
          rewriter.create<ExtractOp>(op.getLoc(), resultType, *it, extractLo));
    }

    widthRemaining -= widthToConsume;

    // Beyond the first element, all elements are extracted from position 0.
    extractLo = 0;
  }

  if (reverseConcatArgs.size() == 1) {
    rewriter.replaceOp(op, reverseConcatArgs[0]);
  } else {
    rewriter.replaceOpWithNewOp<ConcatOp>(
        op, SmallVector<Value>(llvm::reverse(reverseConcatArgs)));
  }
  return success();
}

// Pattern matches on extract(f(a, b)), and transforms f(extract(a), extract(b))
// for some known f. This is performed by analyzing all usage sites of the
// result of f(a, b). When this transformation returns true, it mutates all the
// usage sites of outerExtractOp to reference the newly created narrowed
// operation.
static bool narrowExtractWidth(ExtractOp outerExtractOp,
                               PatternRewriter &rewriter) {
  auto *innerArg = outerExtractOp.input().getDefiningOp();

  if (!innerArg) {
    return false;
  }

  // In calls to narrowOperationWidth below, innerOp is guranteed to have at
  // least one use (ie: this extract operation). So we don't need to handle
  // innerOp with no uses.

  return llvm::TypeSwitch<Operation *, bool>(innerArg)
      // The unreferenced leading bits of Add, Sub and Mul can be stripped of,
      // but not the trailing bits. The trailing bits is used to compute the
      // results of arithmetic operations.
      .Case<AddOp, MulOp>([&](auto innerOp) {
        return narrowOperationWidth(innerOp, innerOp.inputs(),
                                    /* narrowTrailingBits= */ false, rewriter);
      })
      .Case<SubOp>([&](SubOp innerOp) {
        return narrowOperationWidth(innerOp, {innerOp.lhs(), innerOp.rhs()},
                                    /* narrowTrailingBits= */ false, rewriter);
      })
      // Bit-wise operations and muxes can be narrowed more aggressively.
      // Trailing bits that are not referenced in the use-sits can all be
      // removed.
      .Case<AndOp, OrOp, XorOp>([&](auto innerOp) {
        return narrowOperationWidth(innerOp, innerOp.inputs(),
                                    /* narrowTrailingBits= */ true, rewriter);
      })
      .Case<MuxOp>([&](MuxOp innerOp) {
        Type type = innerOp.getType();

        assert(type.isa<IntegerType>() &&
               "extract() requires input to be of type IntegerType!");

        auto cond = innerOp.cond();
        auto loc = innerOp.getLoc();
        auto createMuxOp = [&](ArrayRef<Value> values) -> MuxOp {
          assert(values.size() == 2 &&
                 "createMuxOp expects exactly two elements");
          return rewriter.create<MuxOp>(loc, cond, values[0], values[1]);
        };

        return narrowOperationWidth(
            innerOp, {innerOp.trueValue(), innerOp.falseValue()},
            /* narrowTrailingBits= */ true, rewriter, createMuxOp);
      })

      // TODO: Cautiously investigate whether this optimization can be performed
      // on other comb operators (shifts & divs), arrays, memory, or even
      // sequential operations.
      .Default([](Operation *op) { return false; });
}

LogicalResult ExtractOp::canonicalize(ExtractOp op, PatternRewriter &rewriter) {
  // extract(olo, extract(ilo, x)) = extract(olo + ilo, x)
  if (auto innerExtract =
          dyn_cast_or_null<ExtractOp>(op.input().getDefiningOp())) {
    rewriter.replaceOpWithNewOp<ExtractOp>(op, op.getType(),
                                           innerExtract.input(),
                                           innerExtract.lowBit() + op.lowBit());
    return success();
  }

  // extract(lo, cat(a, b, c, d, e)) = cat(extract(lo1, b), c, extract(lo2, d))
  if (auto innerCat = dyn_cast_or_null<ConcatOp>(op.input().getDefiningOp())) {
    return extractConcatToConcatExtract(op, innerCat, rewriter);
  }

  // extract(f(a, b)) = f(extract(a), extract(b)). This is performed only when
  // the number of bits to operation f can be reduced. See documentation of
  // narrowExtractWidth for more information.
  if (narrowExtractWidth(op, rewriter))
    return success();

  return failure();
}

//===----------------------------------------------------------------------===//
// Variadic operations
//===----------------------------------------------------------------------===//

// Reduce all operands to a single value by applying the `calculate` function.
// This will fail if any of the operands are not constant.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              function_ref<ElementValueT(ElementValueT, ElementValueT)>>
static Attribute constFoldVariadicOp(ArrayRef<Attribute> operands,
                                     const CalculationT &calculate) {
  if (!operands.size())
    return {};

  if (!operands[0])
    return {};

  ElementValueT accum = operands[0].cast<AttrElementT>().getValue();
  for (auto i = operands.begin() + 1, end = operands.end(); i != end; ++i) {
    auto attr = *i;
    if (!attr)
      return {};

    auto typedAttr = attr.cast<AttrElementT>();
    calculate(accum, typedAttr.getValue());
  }
  return AttrElementT::get(operands[0].getType(), accum);
}

/// When we find a logical operation (and, or, xor) with a constant e.g.
/// `X & 42`, we want to push the constant into the computation of X if it leads
/// to simplification.
///
/// This function handles the case where the logical operation has a concat
/// operand.  We check to see if we can simplify the concat, e.g. when it has
/// constant operands.
///
/// This returns true when a simplification happens.
static bool canonicalizeLogicalCstWithConcat(Operation *logicalOp,
                                             size_t concatIdx, const APInt &cst,
                                             PatternRewriter &rewriter) {
  auto concatOp = logicalOp->getOperand(concatIdx).getDefiningOp<ConcatOp>();
  assert((isa<AndOp, OrOp, XorOp>(logicalOp) && concatOp));

  // Check to see if any operands can be simplified by pushing the logical op
  // into all parts of the concat.
  bool canSimplify =
      llvm::any_of(concatOp->getOperands(), [&](Value operand) -> bool {
        auto *operandOp = operand.getDefiningOp();
        if (!operandOp)
          return false;

        // If the concat has a constant operand then we can transform this.
        if (isa<hw::ConstantOp>(operandOp))
          return true;
        // If the concat has the same logical operation and that operation has
        // a constant operation than we can fold it into that suboperation.
        return operandOp->getName() == logicalOp->getName() &&
               operandOp->hasOneUse() && operandOp->getNumOperands() != 0 &&
               operandOp->getOperands().back().getDefiningOp<hw::ConstantOp>();
      });

  if (!canSimplify)
    return false;

  // Create a new instance of the logical operation.  We have to do this the
  // hard way since we're generic across a family of different ops.
  auto createLogicalOp = [&](ArrayRef<Value> operands) -> Value {
    OperationState state(logicalOp->getLoc(), logicalOp->getName());
    state.addOperands(operands);
    state.addTypes(operands[0].getType());
    return rewriter.createOperation(state)->getResult(0);
  };

  // Ok, let's do the transformation.  We do this by slicing up the constant
  // for each unit of the concat and duplicate the operation into the
  // sub-operand.
  SmallVector<Value> newConcatOperands;
  newConcatOperands.reserve(concatOp->getNumOperands());

  // Work from MSB to LSB.
  size_t nextOperandBit = concatOp.getType().getIntOrFloatBitWidth();
  for (Value operand : concatOp->getOperands()) {
    size_t operandWidth = operand.getType().getIntOrFloatBitWidth();
    nextOperandBit -= operandWidth;
    // Take a slice of the constant.
    auto eltCst = rewriter.create<hw::ConstantOp>(
        logicalOp->getLoc(), cst.lshr(nextOperandBit).trunc(operandWidth));

    newConcatOperands.push_back(createLogicalOp({operand, eltCst}));
  }

  // Create the concat, and the rest of the logical op if we need it.
  Value newResult =
      rewriter.create<ConcatOp>(concatOp.getLoc(), newConcatOperands);

  // If we had a variadic logical op on the top level, then recreate it with the
  // new concat and without the constant operand.
  if (logicalOp->getNumOperands() > 2) {
    auto origOperands = logicalOp->getOperands();
    SmallVector<Value> operands;
    // Take any stuff before the concat.
    operands.append(origOperands.begin(), origOperands.begin() + concatIdx);
    // Take any stuff after the concat but before the constant.
    operands.append(origOperands.begin() + concatIdx + 1,
                    origOperands.begin() + (origOperands.size() - 1));
    // Include the new concat.
    operands.push_back(newResult);
    newResult = createLogicalOp(operands);
  }

  rewriter.replaceOp(logicalOp, newResult);
  return true;
}

OpFoldResult AndOp::fold(ArrayRef<Attribute> constants) {
  APInt value = APInt::getAllOnesValue(getType().getWidth());

  // and(x, 01, 10) -> 00 -- annulment.
  for (auto operand : constants) {
    if (!operand)
      continue;
    value &= operand.cast<IntegerAttr>().getValue();
    if (value.isNullValue())
      return getIntAttr(value, getContext());
  }

  // and(x, -1) -> x.
  if (constants.size() == 2 && constants[1] &&
      constants[1].cast<IntegerAttr>().getValue().isAllOnesValue())
    return inputs()[0];

  // and(x, x, x) -> x.  This also handles and(x) -> x.
  if (llvm::all_of(inputs(), [&](auto in) { return in == this->inputs()[0]; }))
    return inputs()[0];

  // and(..., x, ..., ~x, ...) -> 0
  for (Value arg : inputs())
    if (auto xorOp = dyn_cast_or_null<XorOp>(arg.getDefiningOp()))
      if (xorOp.isBinaryNot()) {
        // isBinaryOp checks for the constant on operand 0.
        auto srcVal = xorOp.getOperand(0);
        for (Value arg2 : inputs())
          if (arg2 == srcVal)
            return getIntAttr(APInt(getType().getWidth(), 0), getContext());
      }

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a &= b; });
}

LogicalResult AndOp::canonicalize(AndOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands, `fold` should handle this");

  // and(..., x, ..., x) -> and(..., x, ...) -- idempotent
  // Trivial and(x), and(x, x) cases are handled by [AndOp::fold] above.
  if (inputs.size() > 2) {
    llvm::DenseSet<mlir::Value> dedupedArguments;
    SmallVector<Value, 4> newOperands;

    for (const auto input : inputs) {
      auto insertionResult = dedupedArguments.insert(input);
      if (insertionResult.second) {
        newOperands.push_back(input);
      }
    }

    if (newOperands.size() < inputs.size()) {
      rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
      return success();
    }
  }

  // Patterns for and with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_RConstant(value))) {
    // and(..., '1) -> and(...) -- identity
    if (value.isAllOnesValue()) {
      rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), inputs.drop_back());
      return success();
    }

    // TODO: Combine multiple constants together even if they aren't at the
    // end. and(..., c1, c2) -> and(..., c3) where c3 = c1 & c2 -- constant
    // folding
    APInt value2;
    if (matchPattern(inputs[size - 2], m_RConstant(value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value & value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      rewriter.replaceOpWithNewOp<AndOp>(op, op.getType(), newOperands);
      return success();
    }

    // Handle 'and' with a single bit constant on the RHS.
    if (size == 2 && value.isPowerOf2()) {
      // If the LHS is a sign extend from a single bit, we can 'concat' it
      // into place.  e.g.:
      //   `sext(x) & 4` -> `concat(zeros, x, zeros)`
      if (auto sext = inputs[0].getDefiningOp<SExtOp>()) {
        auto sextOperand = sext.getOperand();
        if (sextOperand.getType().isInteger(1)) {
          unsigned resultWidth = op.getType().getIntOrFloatBitWidth();
          auto trailingZeros = value.countTrailingZeros();

          // We have to dance around the top and bottom bits because we can't
          // make zero bit wide APInts.
          SmallVector<Value, 3> concatOperands;
          if (trailingZeros != resultWidth - 1) {
            auto highZeros = rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt(resultWidth - trailingZeros - 1, 0));
            concatOperands.push_back(highZeros);
          }
          concatOperands.push_back(sextOperand);
          if (trailingZeros != 0) {
            auto lowZeros = rewriter.create<hw::ConstantOp>(
                op.getLoc(), APInt(trailingZeros, 0));
            concatOperands.push_back(lowZeros);
          }
          rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(),
                                                concatOperands);
          return success();
        }
      }
    }

    // and(concat(x, cst1), a, b, c, cst2)
    //    ==> and(a, b, c, concat(and(x,cst2'), and(cst1,cst2'')).
    // We do this for even more multi-use concats since they are "just wiring".
    for (size_t i = 0; i < size - 1; ++i) {
      if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();
    }
  }

  // and(x, and(...)) -> and(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  /// TODO: and(..., x, not(x)) -> and(..., 0) -- complement
  return failure();
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> constants) {
  APInt value(/*numBits=*/getType().getWidth(), 0, /*isSigned=*/false);

  // or(x, 10, 01) -> 11
  for (auto operand : constants) {
    if (!operand)
      continue;
    value |= operand.cast<IntegerAttr>().getValue();
    if (value.isAllOnesValue())
      return getIntAttr(value, getContext());
  }

  // or(x, 0) -> x
  if (constants.size() == 2 && constants[1] &&
      constants[1].cast<IntegerAttr>().getValue().isNullValue())
    return inputs()[0];

  // or(x, x, x) -> x.  This also handles or(x) -> x
  if (llvm::all_of(inputs(), [&](auto in) { return in == this->inputs()[0]; }))
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a |= b; });
}

LogicalResult OrOp::canonicalize(OrOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // or(..., x, x) -> or(..., x) -- idempotent
  if (inputs[size - 1] == inputs[size - 2]) {
    rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
    return success();
  }

  // Patterns for and with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_RConstant(value))) {
    // or(..., '0) -> or(...) -- identity
    if (value.isNullValue()) {
      rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), inputs.drop_back());
      return success();
    }

    // or(..., c1, c2) -> or(..., c3) where c3 = c1 | c2 -- constant folding
    APInt value2;
    if (matchPattern(inputs[size - 2], m_RConstant(value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value | value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      rewriter.replaceOpWithNewOp<OrOp>(op, op.getType(), newOperands);
      return success();
    }

    // or(concat(x, cst1), a, b, c, cst2)
    //    ==> or(a, b, c, concat(or(x,cst2'), or(cst1,cst2'')).
    // We do this for even more multi-use concats since they are "just wiring".
    for (size_t i = 0; i < size - 1; ++i) {
      if (auto concat = inputs[i].getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();
    }
  }

  // or(x, or(...)) -> or(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  /// TODO: or(..., x, not(x)) -> or(..., '1) -- complement
  return failure();
}

OpFoldResult XorOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // xor(x) -> x -- noop
  if (size == 1)
    return inputs()[0];

  // xor(x, x) -> 0 -- idempotent
  if (size == 2 && inputs()[0] == inputs()[1])
    return IntegerAttr::get(getType(), 0);

  // xor(x, 0) -> x
  if (constants.size() == 2 && constants[1] &&
      constants[1].cast<IntegerAttr>().getValue().isNullValue())
    return inputs()[0];

  // xor(xor(x,1),1) -> x
  if (isBinaryNot()) {
    XorOp arg = dyn_cast_or_null<XorOp>(inputs()[0].getDefiningOp());
    if (arg && arg.isBinaryNot())
      return arg.inputs()[0];
  }

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a ^= b; });
}

// xor(icmp, a, b, 1) -> xor(icmp, a, b) if icmp has one user.
static void canonicalizeXorIcmpTrue(XorOp op, unsigned icmpOperand,
                                    PatternRewriter &rewriter) {
  auto icmp = op.getOperand(icmpOperand).getDefiningOp<ICmpOp>();
  auto negatedPred = ICmpOp::getNegatedPredicate(icmp.predicate());

  Value result = rewriter.create<ICmpOp>(
      icmp.getLoc(), negatedPred, icmp.getOperand(0), icmp.getOperand(1));

  // If the xor had other operands, rebuild it.
  if (op.getNumOperands() > 2) {
    SmallVector<Value, 4> newOperands(op.getOperands());
    newOperands.pop_back();
    newOperands.erase(newOperands.begin() + icmpOperand);
    newOperands.push_back(result);
    result = rewriter.create<XorOp>(op.getLoc(), newOperands);
  }

  rewriter.replaceOp(op, result);
}

LogicalResult XorOp::canonicalize(XorOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // xor(..., x, x) -> xor (...) -- idempotent
  if (inputs[size - 1] == inputs[size - 2]) {
    assert(size > 2 &&
           "expected idempotent case for 2 elements handled already.");
    rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(),
                                       inputs.drop_back(/*n=*/2));
    return success();
  }

  // Patterns for xor with a constant on RHS.
  APInt value;
  if (matchPattern(inputs.back(), m_RConstant(value))) {
    // xor(..., 0) -> xor(...) -- identity
    if (value.isNullValue()) {
      rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), inputs.drop_back());
      return success();
    }

    // xor(..., c1, c2) -> xor(..., c3) where c3 = c1 ^ c2.
    APInt value2;
    if (matchPattern(inputs[size - 2], m_RConstant(value2))) {
      auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value ^ value2);
      SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
      newOperands.push_back(cst);
      rewriter.replaceOpWithNewOp<XorOp>(op, op.getType(), newOperands);
      return success();
    }

    bool isSingleBit = value.getBitWidth() == 1;

    // Check for subexpressions that we can simplify.
    for (size_t i = 0; i < size - 1; ++i) {
      Value operand = inputs[i];

      // xor(concat(x, cst1), a, b, c, cst2)
      //    ==> xor(a, b, c, concat(xor(x,cst2'), xor(cst1,cst2'')).
      // We do this for even more multi-use concats since they are "just
      // wiring".
      if (auto concat = operand.getDefiningOp<ConcatOp>())
        if (canonicalizeLogicalCstWithConcat(op, i, value, rewriter))
          return success();

      // xor(icmp, a, b, 1) -> xor(icmp, a, b) if icmp has one user.
      if (isSingleBit && operand.hasOneUse()) {
        assert(value == 1 && "single bit constant has to be one if not zero");
        if (auto icmp = operand.getDefiningOp<ICmpOp>())
          return canonicalizeXorIcmpTrue(op, i, rewriter), success();
      }
    }
  }

  // xor(x, xor(...)) -> xor(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  return failure();
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> constants) {
  APInt value;
  // sub(x - 0) -> x
  if (matchPattern(rhs(), m_RConstant(value)) && value.isNullValue())
    return lhs();

  // sub(x - x) -> 0
  if (rhs() == lhs())
    return getIntAttr(APInt(lhs().getType().getIntOrFloatBitWidth(), 0),
                      getContext());

  // Constant fold
  return constFoldBinaryOp<IntegerAttr>(constants,
                                        [](APInt a, APInt b) { return a - b; });
}

LogicalResult SubOp::canonicalize(SubOp op, PatternRewriter &rewriter) {
  // sub(x, cst) -> add(x, -cst)
  APInt value;
  if (matchPattern(op.rhs(), m_RConstant(value))) {
    auto negCst = rewriter.create<hw::ConstantOp>(op.getLoc(), -value);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.lhs(), negCst);
    return success();
  }

  return failure();
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // add(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a += b; });
}

LogicalResult AddOp::canonicalize(AddOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // add(..., 0) -> add(...) -- identity
  if (matchPattern(inputs.back(), m_RConstant(value)) && value.isNullValue()) {
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), inputs.drop_back());
    return success();
  }

  // add(..., c1, c2) -> add(..., c3) where c3 = c1 + c2 -- constant folding
  if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
      matchPattern(inputs[size - 2], m_RConstant(value2))) {
    auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value + value2);
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(cst);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  // add(..., x, x) -> add(..., shl(x, 1))
  if (inputs[size - 1] == inputs[size - 2]) {
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));

    auto one = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(), 1);
    auto shiftLeftOp =
        rewriter.create<comb::ShlOp>(op.getLoc(), inputs.back(), one);

    newOperands.push_back(shiftLeftOp);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  auto shlOp = inputs[size - 1].getDefiningOp<comb::ShlOp>();
  // add(..., x, shl(x, c)) -> add(..., mul(x, (1 << c) + 1))
  if (shlOp && shlOp.lhs() == inputs[size - 2] &&
      matchPattern(shlOp.rhs(), m_RConstant(value))) {

    APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
    auto rhs =
        rewriter.create<hw::ConstantOp>(op.getLoc(), (one << value) + one);

    std::array<Value, 2> factors = {shlOp.lhs(), rhs};
    auto mulOp = rewriter.create<comb::MulOp>(op.getLoc(), factors);

    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(mulOp);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  auto mulOp = inputs[size - 1].getDefiningOp<comb::MulOp>();
  // add(..., x, mul(x, c)) -> add(..., mul(x, c + 1))
  if (mulOp && mulOp.inputs().size() == 2 &&
      mulOp.inputs()[0] == inputs[size - 2] &&
      matchPattern(mulOp.inputs()[1], m_RConstant(value))) {

    APInt one(/*numBits=*/value.getBitWidth(), 1, /*isSigned=*/false);
    auto rhs = rewriter.create<hw::ConstantOp>(op.getLoc(), value + one);
    std::array<Value, 2> factors = {mulOp.inputs()[0], rhs};
    auto newMulOp = rewriter.create<comb::MulOp>(op.getLoc(), factors);

    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(newMulOp);
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), newOperands);
    return success();
  }

  // add(x, add(...)) -> add(x, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  return failure();
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> constants) {
  auto size = inputs().size();

  // mul(x) -> x -- noop
  if (size == 1u)
    return inputs()[0];

  auto width = getType().getWidth();
  APInt value(/*numBits=*/width, 1, /*isSigned=*/false);

  // mul(x, 0, 1) -> 0 -- annulment
  for (auto operand : constants) {
    if (!operand)
      continue;
    value *= operand.cast<IntegerAttr>().getValue();
    if (value.isNullValue())
      return getIntAttr(value, getContext());
  }

  // Constant fold
  return constFoldVariadicOp<IntegerAttr>(
      constants, [](APInt &a, const APInt &b) { a *= b; });
}

LogicalResult MulOp::canonicalize(MulOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  APInt value, value2;

  // mul(x, c) -> shl(x, log2(c)), where c is a power of two.
  if (size == 2 && matchPattern(inputs.back(), m_RConstant(value)) &&
      value.isPowerOf2()) {
    auto shift = rewriter.create<hw::ConstantOp>(op.getLoc(), op.getType(),
                                                 value.exactLogBase2());
    auto shlOp = rewriter.create<comb::ShlOp>(op.getLoc(), inputs[0], shift);

    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(),
                                       ArrayRef<Value>(shlOp));
    return success();
  }

  // mul(..., 1) -> mul(...) -- identity
  if (matchPattern(inputs.back(), m_RConstant(value)) && (value == 1u)) {
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), inputs.drop_back());
    return success();
  }

  // mul(..., c1, c2) -> mul(..., c3) where c3 = c1 * c2 -- constant folding
  if (matchPattern(inputs[size - 1], m_RConstant(value)) &&
      matchPattern(inputs[size - 2], m_RConstant(value2))) {
    auto cst = rewriter.create<hw::ConstantOp>(op.getLoc(), value * value2);
    SmallVector<Value, 4> newOperands(inputs.drop_back(/*n=*/2));
    newOperands.push_back(cst);
    rewriter.replaceOpWithNewOp<MulOp>(op, op.getType(), newOperands);
    return success();
  }

  // mul(a, mul(...)) -> mul(a, ...) -- flatten
  if (tryFlatteningOperands(op, rewriter))
    return success();

  return failure();
}

template <class Op, bool isSigned>
static OpFoldResult foldDiv(Op op, ArrayRef<Attribute> constants) {
  if (auto rhs_value = constants[1].dyn_cast_or_null<IntegerAttr>()) {
    // divu(x, 1) -> x, divs(x, 1) -> x
    if (rhs_value.getValue() == 1)
      return op.lhs();

    // If the divisor is zero, do not fold for now.
    if (rhs_value.getValue().isNullValue())
      return {};
  }

  return constFoldBinaryOp<IntegerAttr>(constants, [](APInt a, APInt b) {
    return isSigned ? a.sdiv(b) : a.udiv(b);
  });
}

OpFoldResult DivUOp::fold(ArrayRef<Attribute> constants) {
  return foldDiv<DivUOp, /*isSigned=*/false>(*this, constants);
}

OpFoldResult DivSOp::fold(ArrayRef<Attribute> constants) {
  return foldDiv<DivSOp, /*isSigned=*/true>(*this, constants);
}

template <class Op, bool isSigned>
static OpFoldResult foldMod(Op op, ArrayRef<Attribute> constants) {
  if (auto rhs_value = constants[1].dyn_cast_or_null<IntegerAttr>()) {
    // modu(x, 1) -> 0, mods(x, 1) -> 0
    if (rhs_value.getValue() == 1)
      return getIntAttr(APInt(op.getType().getIntOrFloatBitWidth(), 0),
                        op.getContext());

    // If the divisor is zero, do not fold for now.
    if (rhs_value.getValue().isNullValue())
      return {};
  }

  return constFoldBinaryOp<IntegerAttr>(constants, [](APInt a, APInt b) {
    return isSigned ? a.srem(b) : a.urem(b);
  });
}

OpFoldResult ModUOp::fold(ArrayRef<Attribute> constants) {
  return foldMod<ModUOp, /*isSigned=*/false>(*this, constants);
}

OpFoldResult ModSOp::fold(ArrayRef<Attribute> constants) {
  return foldMod<ModSOp, /*isSigned=*/true>(*this, constants);
}
//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// Constant folding
OpFoldResult ConcatOp::fold(ArrayRef<Attribute> constants) {
  if (getNumOperands() == 1)
    return getOperand(0);

  // If all the operands are constant, we can fold.
  for (auto attr : constants)
    if (!attr || !attr.isa<IntegerAttr>())
      return {};

  // If we got here, we can constant fold.
  unsigned resultWidth = getType().getIntOrFloatBitWidth();
  APInt result(resultWidth, 0);

  unsigned nextInsertion = resultWidth;
  // Insert each chunk into the result.
  for (auto attr : constants) {
    auto chunk = attr.cast<IntegerAttr>().getValue();
    nextInsertion -= chunk.getBitWidth();
    result |= chunk.zext(resultWidth) << nextInsertion;
  }

  return getIntAttr(result, getContext());
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  auto inputs = op.inputs();
  auto size = inputs.size();
  assert(size > 1 && "expected 2 or more operands");

  // This function is used when we flatten neighboring operands of a
  // (variadic) concat into a new vesion of the concat.  first/last indices
  // are inclusive.
  auto flattenConcat = [&](size_t firstOpIndex, size_t lastOpIndex,
                           ValueRange replacements) -> LogicalResult {
    SmallVector<Value, 4> newOperands;
    newOperands.append(inputs.begin(), inputs.begin() + firstOpIndex);
    newOperands.append(replacements.begin(), replacements.end());
    newOperands.append(inputs.begin() + lastOpIndex + 1, inputs.end());
    if (newOperands.size() == 1)
      rewriter.replaceOp(op, newOperands[0]);
    else
      rewriter.replaceOpWithNewOp<ConcatOp>(op, op.getType(), newOperands);
    return success();
  };

  for (size_t i = 0; i != size; ++i) {
    // If an operand to the concat is itself a concat, then we can fold them
    // together.
    if (auto subConcat = inputs[i].getDefiningOp<ConcatOp>())
      return flattenConcat(i, i, subConcat->getOperands());

    // We can flatten a sext into the concat, since a sext is an extraction of
    // a sign bit plus the input value itself.
    if (auto sext = inputs[i].getDefiningOp<SExtOp>()) {
      unsigned inputWidth = sext.input().getType().getIntOrFloatBitWidth();

      auto sign = rewriter.create<ExtractOp>(op.getLoc(), rewriter.getI1Type(),
                                             sext.input(), inputWidth - 1);
      SmallVector<Value> replacements;
      unsigned signWidth = sext.getType().getIntOrFloatBitWidth() - inputWidth;
      replacements.append(signWidth, sign);
      replacements.push_back(sext.input());
      return flattenConcat(i, i, replacements);
    }

    // Check for canonicalization due to neighboring operands.
    if (i != 0) {
      // Merge neighboring constants.
      if (auto cst = inputs[i].getDefiningOp<hw::ConstantOp>()) {
        if (auto prevCst = inputs[i - 1].getDefiningOp<hw::ConstantOp>()) {
          unsigned prevWidth = prevCst.getValue().getBitWidth();
          unsigned thisWidth = cst.getValue().getBitWidth();
          auto resultCst = cst.getValue().zext(prevWidth + thisWidth);
          resultCst |= prevCst.getValue().zext(prevWidth + thisWidth)
                       << thisWidth;
          Value replacement =
              rewriter.create<hw::ConstantOp>(op.getLoc(), resultCst);
          return flattenConcat(i - 1, i, replacement);
        }
      }

      // Merge neighboring extracts of neighboring inputs, e.g.
      // {A[3], A[2]} -> A[3,2]
      if (auto extract = inputs[i].getDefiningOp<ExtractOp>()) {
        if (auto prevExtract = inputs[i - 1].getDefiningOp<ExtractOp>()) {
          if (extract.input() == prevExtract.input()) {
            auto thisWidth = extract.getType().getIntOrFloatBitWidth();
            if (prevExtract.lowBit() == extract.lowBit() + thisWidth) {
              auto prevWidth = prevExtract.getType().getIntOrFloatBitWidth();
              auto resType = rewriter.getIntegerType(thisWidth + prevWidth);
              Value replacement = rewriter.create<ExtractOp>(
                  op.getLoc(), resType, extract.input(), extract.lowBit());
              return flattenConcat(i - 1, i, replacement);
            }
          }
        }
      }
    }
  }

  // Return true if this extract is an extract of the sign bit of its input.
  auto isSignBitExtract = [](ExtractOp op) -> bool {
    if (op.getType().getIntOrFloatBitWidth() != 1)
      return false;
    return op.lowBit() == op.input().getType().getIntOrFloatBitWidth() - 1;
  };

  // Folds concat back into a sext.
  if (auto extract = inputs[0].getDefiningOp<ExtractOp>()) {
    if (extract.input() == inputs.back() && isSignBitExtract(extract)) {
      // Check intermediate bits.
      bool allMatch = true;
      // The intermediate bits are allowed to be difference instances of
      // extract (because canonicalize doesn't do CSE automatically) so long
      // as they are getting the sign bit.
      for (size_t i = 1, e = inputs.size() - 1; i != e; ++i) {
        auto extractInner = inputs[i].getDefiningOp<ExtractOp>();
        if (!extractInner || extractInner.input() != inputs.back() ||
            !isSignBitExtract(extractInner)) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        rewriter.replaceOpWithNewOp<SExtOp>(op, op.getType(), inputs.back());
        return success();
      }
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

OpFoldResult MuxOp::fold(ArrayRef<Attribute> constants) {
  // mux (c, b, b) -> b
  if (trueValue() == falseValue())
    return trueValue();

  // mux(0, a, b) -> b
  // mux(1, a, b) -> a
  if (auto pred = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    if (pred.getValue().isNullValue())
      return falseValue();
    return trueValue();
  }

  return {};
}

/// Given a mux, check to see if the "on true" value (or "on false" value if
/// isFalseSide=true) is a mux tree with the same condition.  This allows us
/// to turn things like `mux(VAL == 0, A, (mux (VAL == 1), B, C))` into
/// `array_get (array_create(A, B, C), VAL)` which is far more compact and
/// allows synthesis tools to do more interesting optimizations.
///
/// This returns false if we cannot form the mux tree (or do not want to) and
/// returns true if the mux was replaced.
static bool foldMuxChain(MuxOp rootMux, bool isFalseSide,
                         PatternRewriter &rewriter) {
  // Get the index value being compared.  Later we check to see if it is
  // compared to a constant with the right predicate.
  auto rootCmp = rootMux.cond().getDefiningOp<ICmpOp>();
  if (!rootCmp)
    return false;
  Value indexValue = rootCmp.lhs();

  // Check to see if the condition to the specified mux is an equality
  // comparison `indexValue` and a constant.  If so, return the constant, if not
  // return null.
  auto getCondConstant = [&](MuxOp mux) -> hw::ConstantOp {
    auto topLevelCmp = mux.cond().getDefiningOp<ICmpOp>();

    // TODO: We could conceivably handle things like "x < 2" as two entries
    // if there was a reason to.  We could also handle a mix of == / !=
    // comparisons if they occur.
    auto requiredPredicate =
        (isFalseSide ? ICmpPredicate::eq : ICmpPredicate::ne);
    if (!topLevelCmp || topLevelCmp.lhs() != indexValue ||
        topLevelCmp.predicate() != requiredPredicate)
      return {};
    return topLevelCmp.rhs().getDefiningOp<hw::ConstantOp>();
  };

  // Return the value to use if the equality match succeeds.
  auto getCaseValue = [&](MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(!isFalseSide));
  };

  // Return the value to use if the equality match fails.  This is the next
  // mux in the sequence or the "otherwise" value.
  auto getTreeValue = [&](MuxOp mux) -> Value {
    return mux.getOperand(1 + unsigned(isFalseSide));
  };

  // Make sure the root is a correct comparison with a constant.
  auto rootCst = getCondConstant(rootMux);
  if (!rootCst)
    return false;

  // Make sure that we're not looking at the intermediate node in a mux tree.
  if (rootMux->hasOneUse()) {
    if (auto userMux = dyn_cast<MuxOp>(*rootMux->user_begin())) {
      if (getCondConstant(userMux) &&
          getTreeValue(userMux) == rootMux.getResult())
        return false;
    }
  }

  // Start scanning the mux tree to see what we've got.  Keep track of the
  // constant comparison value and the SSA value to use when equal to it.
  SmallVector<std::pair<hw::ConstantOp, Value>, 4> valuesFound;
  SmallVector<Location> locationsFound;
  valuesFound.push_back({rootCst, getCaseValue(rootMux)});

  // Scan up the tree linearly.
  auto nextTreeValue = getTreeValue(rootMux);
  while (1) {
    auto nextMux = nextTreeValue.getDefiningOp<MuxOp>();
    if (!nextMux || !nextMux->hasOneUse())
      break;
    auto nextCst = getCondConstant(nextMux);
    if (!nextCst)
      break;
    valuesFound.push_back({nextCst, getCaseValue(nextMux)});
    locationsFound.push_back(nextMux.cond().getLoc());
    locationsFound.push_back(nextMux->getLoc());
    nextTreeValue = getTreeValue(nextMux);
  }

  // We need to have more than three values to create an array.  This is an
  // arbitrary threshold which is saying that one or two muxes together is ok,
  // but three should be folded.
  if (valuesFound.size() < 3)
    return false;

  // Next we need to see if the values are dense-ish.  We don't want to have
  // a tremendous number of replicated entries in the array.  Some sparsity is
  // ok though, so we require the table to be at least 5/8 utilized.
  uint64_t tableSize = 1ULL
                       << indexValue.getType().cast<IntegerType>().getWidth();
  if (valuesFound.size() < (tableSize * 5) / 8)
    return false; // Not dense enough.

  // Ok, we're going to do the transformation, start by building the table
  // filled with the "otherwise" value.
  SmallVector<Value, 8> table(tableSize, nextTreeValue);

  // Fill in entries in the table from the leaf to the root of the expression.
  // This ensures that any duplicate matches end up with the ultimate value,
  // which is the one closer to the root.
  for (auto &elt : llvm::reverse(valuesFound)) {
    uint64_t idx = elt.first.getValue().getZExtValue();
    assert(idx < table.size() && "constant should be same bitwidth as index");
    table[idx] = elt.second;
  }

  // The hw.array_create operation has the operand list in unintuitive order
  // with a[0] stored as the last element, not the first.
  std::reverse(table.begin(), table.end());

  // Build the array_create and the array_get.
  auto fusedLoc = rewriter.getFusedLoc(locationsFound);
  auto array = rewriter.create<hw::ArrayCreateOp>(fusedLoc, table);
  rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(rootMux, array, indexValue);
  return true;
}

LogicalResult MuxOp::canonicalize(MuxOp op, PatternRewriter &rewriter) {
  APInt value;

  // mux(a, 1, b) -> or(a, b) for single-bit values.
  if (matchPattern(op.trueValue(), m_RConstant(value)) &&
      value.getBitWidth() == 1 && value.isAllOnesValue()) {
    rewriter.replaceOpWithNewOp<OrOp>(op, op.cond(), op.falseValue());
    return success();
  }

  // mux(a, b, 0) -> and(a, b) for single-bit values.
  if (matchPattern(op.falseValue(), m_RConstant(value)) &&
      value.isNullValue() && value.getBitWidth() == 1) {
    rewriter.replaceOpWithNewOp<AndOp>(op, op.cond(), op.trueValue());
    return success();
  }

  // mux(!a, b, c) -> mux(a, c, b)
  if (auto xorOp = dyn_cast_or_null<XorOp>(op.cond().getDefiningOp())) {
    if (xorOp.isBinaryNot()) {
      rewriter.replaceOpWithNewOp<MuxOp>(op, op.getType(), xorOp.inputs()[0],
                                         op.falseValue(), op.trueValue());
      return success();
    }
  }

  if (auto falseMux =
          dyn_cast_or_null<MuxOp>(op.falseValue().getDefiningOp())) {
    // mux(selector, x, mux(selector, y, z) = mux(selector, x, z)
    if (op.cond() == falseMux.cond()) {
      rewriter.replaceOpWithNewOp<MuxOp>(op, op.cond(), op.trueValue(),
                                         falseMux.falseValue());
      return success();
    }

    // Check to see if we can fold a mux tree into an array_create/get pair.
    if (foldMuxChain(op, /*isFalse*/ true, rewriter))
      return success();
  }

  if (auto trueMux = dyn_cast_or_null<MuxOp>(op.trueValue().getDefiningOp())) {
    // mux(selector, mux(selector, a, b), c) = mux(selector, a, c)
    if (op.cond() == trueMux.cond()) {
      rewriter.replaceOpWithNewOp<MuxOp>(op, op.cond(), trueMux.trueValue(),
                                         op.falseValue());
      return success();
    }

    // Check to see if we can fold a mux tree into an array_create/get pair.
    if (foldMuxChain(op, /*isFalseSide*/ false, rewriter))
      return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ICmpOp
//===----------------------------------------------------------------------===//

// Calculate the result of a comparison when the LHS and RHS are both
// constants.
static bool applyCmpPredicate(ICmpPredicate predicate, const APInt &lhs,
                              const APInt &rhs) {
  switch (predicate) {
  case ICmpPredicate::eq:
    return lhs.eq(rhs);
  case ICmpPredicate::ne:
    return lhs.ne(rhs);
  case ICmpPredicate::slt:
    return lhs.slt(rhs);
  case ICmpPredicate::sle:
    return lhs.sle(rhs);
  case ICmpPredicate::sgt:
    return lhs.sgt(rhs);
  case ICmpPredicate::sge:
    return lhs.sge(rhs);
  case ICmpPredicate::ult:
    return lhs.ult(rhs);
  case ICmpPredicate::ule:
    return lhs.ule(rhs);
  case ICmpPredicate::ugt:
    return lhs.ugt(rhs);
  case ICmpPredicate::uge:
    return lhs.uge(rhs);
  }
  llvm_unreachable("unknown comparison predicate");
}

// Returns the result of applying the predicate when the LHS and RHS are the
// exact same value.
static bool applyCmpPredicateToEqualOperands(ICmpPredicate predicate) {
  switch (predicate) {
  case ICmpPredicate::eq:
  case ICmpPredicate::sle:
  case ICmpPredicate::sge:
  case ICmpPredicate::ule:
  case ICmpPredicate::uge:
    return true;
  case ICmpPredicate::ne:
  case ICmpPredicate::slt:
  case ICmpPredicate::sgt:
  case ICmpPredicate::ult:
  case ICmpPredicate::ugt:
    return false;
  }
  llvm_unreachable("unknown comparison predicate");
}

OpFoldResult ICmpOp::fold(ArrayRef<Attribute> constants) {
  // gt a, a -> false
  // gte a, a -> true
  if (lhs() == rhs()) {
    auto val = applyCmpPredicateToEqualOperands(predicate());
    return IntegerAttr::get(getType(), val);
  }

  // gt 1, 2 -> false
  if (auto lhs = constants[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = constants[1].dyn_cast_or_null<IntegerAttr>()) {
      auto val = applyCmpPredicate(predicate(), lhs.getValue(), rhs.getValue());
      return IntegerAttr::get(getType(), val);
    }
  }
  return {};
}

// Given a range of operands, computes the number of matching prefix and
// suffix elements. This does not perform cross-element matching.
template <typename Range>
static size_t computeCommonPrefixLength(const Range &a, const Range &b) {
  size_t commonPrefixLength = 0;
  auto ia = a.begin();
  auto ib = b.begin();

  for (; ia != a.end() && ib != b.end(); ia++, ib++, commonPrefixLength++) {
    if (*ia != *ib) {
      break;
    }
  }

  return commonPrefixLength;
}

static size_t getTotalWidth(const OperandRange &range) {
  size_t totalWidth = 0;
  for (auto operand : range) {
    // getIntOrFloatBitWidth should never raise, since all arguments to
    // ConcatOp are integers.
    ssize_t width = operand.getType().getIntOrFloatBitWidth();
    assert(width >= 0);
    totalWidth += width;
  }
  return totalWidth;
}

// Reduce the strength icmp(concat(...), concat(...)) by doing a element-wise
// comparison on common prefix and suffixes. Returns success() if a rewriting
// happens.
static LogicalResult matchAndRewriteCompareConcat(ICmpOp op, ConcatOp lhs,
                                                  ConcatOp rhs,
                                                  PatternRewriter &rewriter) {
  // It is safe to assume that [{lhsOperands, rhsOperands}.size() > 0] and
  // all elements have non-zero length. Both these invariants are verified
  // by the ConcatOp verifier.
  auto lhsOperands = lhs.getOperands();
  auto rhsOperands = rhs.getOperands();
  size_t numElements = std::min<size_t>(lhsOperands.size(), rhsOperands.size());

  size_t commonPrefixLength =
      computeCommonPrefixLength(lhsOperands, rhsOperands);
  size_t commonSuffixLength = computeCommonPrefixLength(
      llvm::reverse(lhsOperands.drop_front(commonPrefixLength)),
      llvm::reverse(rhsOperands.drop_front(commonPrefixLength)));

  auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, constant));
    return success();
  };

  auto directOrCat = [&](const OperandRange &range) -> Value {
    assert(range.size() > 0);
    if (range.size() == 1) {
      return range.front();
    }

    return rewriter.create<ConcatOp>(op.getLoc(), range);
  };

  auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                         Value rhs) -> LogicalResult {
    rewriter.replaceOpWithNewOp<ICmpOp>(op, predicate, lhs, rhs);
    return success();
  };

  if (commonPrefixLength == numElements) {
    // cat(a, b, c) == cat(a, b, c) -> 1
    return replaceWithConstantI1(
        applyCmpPredicateToEqualOperands(op.predicate()));
  }

  size_t commonPrefixTotalWidth =
      getTotalWidth(lhsOperands.take_front(commonPrefixLength));
  size_t commonSuffixTotalWidth =
      getTotalWidth(lhsOperands.take_back(commonSuffixLength));
  auto lhsOnly =
      lhsOperands.drop_front(commonPrefixLength).drop_back(commonSuffixLength);
  auto rhsOnly =
      rhsOperands.drop_front(commonPrefixLength).drop_back(commonSuffixLength);

  auto replaceWithoutReplicatingSignBit = [&]() {
    auto newLhs = directOrCat(lhsOnly);
    auto newRhs = directOrCat(rhsOnly);
    return replaceWith(op.predicate(), newLhs, newRhs);
  };

  auto replaceWithReplicatingSignBit = [&]() {
    auto firstNonEmptyValue = lhsOperands[0];
    auto firstNonEmptyElemWidth =
        firstNonEmptyValue.getType().getIntOrFloatBitWidth();
    Value signBit;

    // Skip creating an ExtractOp where possible.
    if (firstNonEmptyElemWidth == 1) {
      signBit = firstNonEmptyValue;
    } else {
      signBit = rewriter.create<ExtractOp>(
          op.getLoc(), IntegerType::get(rewriter.getContext(), 1),
          firstNonEmptyValue, firstNonEmptyElemWidth - 1);
    }

    auto newLhs = rewriter.create<ConcatOp>(op.getLoc(), signBit, lhsOnly);
    auto newRhs = rewriter.create<ConcatOp>(op.getLoc(), signBit, rhsOnly);
    return replaceWith(op.predicate(), newLhs, newRhs);
  };

  if (ICmpOp::isPredicateSigned(op.predicate())) {
    // scmp(cat(..x, b), cat(..y, b)) == scmp(cat(..x), cat(..y))
    if (commonPrefixTotalWidth == 0 && commonSuffixTotalWidth > 0)
      return replaceWithoutReplicatingSignBit();

    // scmp(cat(a, ..x, b), cat(a, ..y, b)) == scmp(cat(sgn(a), ..x),
    // cat(sgn(b), ..y)) Note that we cannot perform this optimization if
    // [width(b) = 0 && width(a) <= 1]. since that common prefix is the sign
    // bit. Doing the rewrite can result in an infinite loop.
    if (commonPrefixTotalWidth > 1 || commonSuffixTotalWidth > 0)
      return replaceWithReplicatingSignBit();

  } else if (commonPrefixTotalWidth > 0 || commonSuffixTotalWidth > 0) {
    // ucmp(cat(a, ..x, b), cat(a, ..y, b)) = ucmp(cat(..x), cat(..y))
    return replaceWithoutReplicatingSignBit();
  }

  return failure();
}

/// Given an equality comparison with a constant value and some operand that has
/// known bits, simplify the comparison to check only the unknown bits of the
/// input.
///
/// One simple example of this is that `concat(0, stuff) == 0` can be simplified
/// to `stuff == 0`, or `and(x, 3) == 0` can be simplified to
/// `extract x[1:0] == 0`
static void combineEqualityICmpWithKnownBitsAndConstant(
    ICmpOp cmpOp, const KnownBitAnalysis &bitAnalysis, const APInt &rhsCst,
    PatternRewriter &rewriter) {

  // If any of the known bits disagree with any of the comparison bits, then
  // we can constant fold this comparison right away.
  APInt bitsKnown = bitAnalysis.getBitsKnown();
  if ((bitsKnown & rhsCst) != bitAnalysis.ones) {
    // If we discover a mismatch then we know an "eq" comparison is false
    // and a "ne" comparison is true!
    bool result = cmpOp.predicate() == ICmpPredicate::ne;
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(cmpOp, APInt(1, result));
    return;
  }

  // Check to see if we can prove the result entirely of the comparison (in
  // which we bail out early), otherwise build a list of values to concat and a
  // smaller constant to compare against.
  SmallVector<Value> newConcatOperands;
  APInt newConstant;

  // Ok, some (maybe all) bits are known and some others may be unknown.
  // Extract out segments of the operand and compare against the
  // corresponding bits.
  unsigned knownMSB = bitsKnown.countLeadingOnes();

  Value operand = cmpOp.lhs();

  // Ok, some bits are known but others are not.  Extract out sequences of
  // bits that are unknown and compare just those bits.  We work from MSB to
  // LSB.
  while (knownMSB != bitsKnown.getBitWidth()) {
    // Drop any high bits that are known.
    if (knownMSB)
      bitsKnown = bitsKnown.trunc(bitsKnown.getBitWidth() - knownMSB);

    // Find the span of unknown bits, and extract it.
    unsigned unknownBits = bitsKnown.countLeadingZeros();
    unsigned lowBit = bitsKnown.getBitWidth() - unknownBits;
    auto spanOperand = rewriter.createOrFold<ExtractOp>(
        operand.getLoc(), operand, /*lowBit=*/lowBit,
        /*bitWidth=*/unknownBits);
    auto spanConstant = rhsCst.lshr(lowBit).trunc(unknownBits);

    // Add this info to the concat we're generating.
    newConcatOperands.push_back(spanOperand);
    newConstant = (newConstant.zext(newConstant.getBitWidth() + unknownBits)
                   << unknownBits);
    newConstant |= spanConstant.zext(newConstant.getBitWidth());

    // Drop the unknown bits in prep for the next chunk.  Dance around APInts
    // limitation of not being able to do zero bit things.
    unsigned newWidth = bitsKnown.getBitWidth() - unknownBits;
    if (newWidth == 0)
      break;
    bitsKnown = bitsKnown.trunc(newWidth);
    knownMSB = bitsKnown.countLeadingOnes();
  }

  // If all the operands to the concat are foldable then we have an identity
  // situation where all the sub-elements equal each other.  This implies that
  // the overall result is foldable.
  if (newConcatOperands.empty()) {
    bool result = cmpOp.predicate() == ICmpPredicate::eq;
    rewriter.replaceOpWithNewOp<hw::ConstantOp>(cmpOp, APInt(1, result));
    return;
  }

  // If we have a single operand remaining, use it, otherwise form a concat.
  Value concatResult =
      rewriter.createOrFold<ConcatOp>(operand.getLoc(), newConcatOperands);

  // The default APInt constructor adds an extra bit to the top of the APInt
  // constructor, trim it off now.
  newConstant =
      newConstant.trunc(concatResult.getType().getIntOrFloatBitWidth());

  // Form the comparison against the smaller constant.
  auto newConstantOp = rewriter.create<hw::ConstantOp>(
      cmpOp.getOperand(1).getLoc(), newConstant);

  rewriter.replaceOpWithNewOp<ICmpOp>(cmpOp, cmpOp.predicate(), concatResult,
                                      newConstantOp);
}

// Simplify icmp eq(xor(a,b,cst1), cst2) -> icmp eq(xor(a,b), cst1^cst2).
static void combineEqualityICmpWithXorOfConstant(ICmpOp cmpOp, XorOp xorOp,
                                                 const APInt &rhs,
                                                 PatternRewriter &rewriter) {
  auto xorRHS = xorOp.getOperands().back().getDefiningOp<hw::ConstantOp>();
  auto newRHS = rewriter.create<hw::ConstantOp>(xorRHS->getLoc(),
                                                xorRHS.getValue() ^ rhs);
  Value newLHS;
  switch (xorOp.getNumOperands()) {
  case 1:
    // This isn't common but is defined so we need to handle it.
    newLHS = rewriter.create<hw::ConstantOp>(xorOp.getLoc(),
                                             APInt(rhs.getBitWidth(), 0));
    break;
  case 2:
    // The binary case is the most common.
    newLHS = xorOp.getOperand(0);
    break;
  default:
    // The general case forces us to form a new xor with the remaining operands.
    SmallVector<Value> newOperands(xorOp.getOperands());
    newOperands.pop_back();
    newLHS = rewriter.create<XorOp>(xorOp.getLoc(), newOperands);
    break;
  }

  bool xorMultipleUses = !xorOp->hasOneUse();

  // If the xor has multiple uses (not just the compare, then we need/want to
  // replace them as well.
  if (xorMultipleUses) {
    auto newXor = rewriter.create<XorOp>(xorOp.getLoc(), newLHS, xorRHS);
    rewriter.replaceOp(xorOp, newXor->getResult(0));
  }

  // Replace the comparison.
  rewriter.replaceOpWithNewOp<ICmpOp>(cmpOp, cmpOp.predicate(), newLHS, newRHS);
}

// Canonicalizes a ICmp with a single constant
LogicalResult ICmpOp::canonicalize(ICmpOp op, PatternRewriter &rewriter) {
  APInt lhs, rhs;

  // icmp 1, x -> icmp x, 1
  if (matchPattern(op.lhs(), m_RConstant(lhs))) {
    assert(!matchPattern(op.rhs(), m_RConstant(rhs)) && "Should be folded");
    rewriter.replaceOpWithNewOp<ICmpOp>(
        op, ICmpOp::getFlippedPredicate(op.predicate()), op.rhs(), op.lhs());
    return success();
  }

  // Canonicalize with RHS constant
  if (matchPattern(op.rhs(), m_RConstant(rhs))) {
    auto getConstant = [&](APInt constant) -> Value {
      return rewriter.create<hw::ConstantOp>(op.getLoc(), std::move(constant));
    };

    auto replaceWith = [&](ICmpPredicate predicate, Value lhs,
                           Value rhs) -> LogicalResult {
      rewriter.replaceOpWithNewOp<ICmpOp>(op, predicate, lhs, rhs);
      return success();
    };

    auto replaceWithConstantI1 = [&](bool constant) -> LogicalResult {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, APInt(1, constant));
      return success();
    };

    switch (op.predicate()) {
    case ICmpPredicate::slt:
      // x < max -> x != max
      if (rhs.isMaxSignedValue())
        return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
      // x < min -> false
      if (rhs.isMinSignedValue())
        return replaceWithConstantI1(0);
      // x < min+1 -> x == min
      if ((rhs - 1).isMinSignedValue())
        return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs - 1));
      break;
    case ICmpPredicate::sgt:
      // x > min -> x != min
      if (rhs.isMinSignedValue())
        return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
      // x > max -> false
      if (rhs.isMaxSignedValue())
        return replaceWithConstantI1(0);
      // x > max-1 -> x == max
      if ((rhs + 1).isMaxSignedValue())
        return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs + 1));
      break;
    case ICmpPredicate::ult:
      // x < max -> x != max
      if (rhs.isMaxValue())
        return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
      // x < min -> false
      if (rhs.isMinValue())
        return replaceWithConstantI1(0);
      // x < min+1 -> x == min
      if ((rhs - 1).isMinValue())
        return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs - 1));
      break;
    case ICmpPredicate::ugt:
      // x > min -> x != min
      if (rhs.isMinValue())
        return replaceWith(ICmpPredicate::ne, op.lhs(), op.rhs());
      // x > max -> false
      if (rhs.isMaxValue())
        return replaceWithConstantI1(0);
      // x > max-1 -> x == max
      if ((rhs + 1).isMaxValue())
        return replaceWith(ICmpPredicate::eq, op.lhs(), getConstant(rhs + 1));
      break;
    case ICmpPredicate::sle:
      // x <= max -> true
      if (rhs.isMaxSignedValue())
        return replaceWithConstantI1(1);
      // x <= c -> x < (c+1)
      return replaceWith(ICmpPredicate::slt, op.lhs(), getConstant(rhs + 1));
    case ICmpPredicate::sge:
      // x >= min -> true
      if (rhs.isMinSignedValue())
        return replaceWithConstantI1(1);
      // x >= c -> x > (c-1)
      return replaceWith(ICmpPredicate::sgt, op.lhs(), getConstant(rhs - 1));
    case ICmpPredicate::ule:
      // x <= max -> true
      if (rhs.isMaxValue())
        return replaceWithConstantI1(1);
      // x <= c -> x < (c+1)
      return replaceWith(ICmpPredicate::ult, op.lhs(), getConstant(rhs + 1));
    case ICmpPredicate::uge:
      // x >= min -> true
      if (rhs.isMinValue())
        return replaceWithConstantI1(1);
      // x >= c -> x > (c-1)
      return replaceWith(ICmpPredicate::ugt, op.lhs(), getConstant(rhs - 1));
    case ICmpPredicate::eq:
      if (rhs.getBitWidth() == 1) {
        if (rhs.isNullValue()) {
          // x == 0 -> x ^ 1
          rewriter.replaceOpWithNewOp<XorOp>(op, op.lhs(),
                                             getConstant(APInt(1, 1)));
          return success();
        }
        if (rhs.isAllOnesValue()) {
          // x == 1 -> x
          rewriter.replaceOp(op, op.lhs());
          return success();
        }
      }
      break;
    case ICmpPredicate::ne:
      if (rhs.getBitWidth() == 1) {
        if (rhs.isNullValue()) {
          // x != 0 -> x
          rewriter.replaceOp(op, op.lhs());
          return success();
        }
        if (rhs.isAllOnesValue()) {
          // x != 1 -> x ^ 1
          rewriter.replaceOpWithNewOp<XorOp>(op, op.lhs(),
                                             getConstant(APInt(1, 1)));
          return success();
        }
      }
      break;
    }

    // We have some specific optimizations for comparison with a constant that
    // are only supported for equality comparisons.
    if (op.predicate() == ICmpPredicate::eq ||
        op.predicate() == ICmpPredicate::ne) {
      // Simplify `icmp(value_with_known_bits, rhscst)` into some extracts
      // with a smaller constant.  We only support equality comparisons for
      // this.
      auto knownBits = KnownBitAnalysis::compute(op.lhs());
      if (knownBits.areAnyKnown())
        return combineEqualityICmpWithKnownBitsAndConstant(op, knownBits, rhs,
                                                           rewriter),
               success();

      // Simplify icmp eq(xor(a,b,cst1), cst2) -> icmp eq(xor(a,b),
      // cst1^cst2).
      if (auto xorOp = op.lhs().getDefiningOp<XorOp>())
        if (xorOp.getOperands().back().getDefiningOp<hw::ConstantOp>())
          return combineEqualityICmpWithXorOfConstant(op, xorOp, rhs, rewriter),
                 success();
    }
  }

  // icmp(cat(prefix, a, b, suffix), cat(prefix, c, d, suffix)) => icmp(cat(a,
  // b), cat(c, d)). contains special handling for sign bit in signed
  // compressions.
  if (auto lhs = dyn_cast_or_null<ConcatOp>(op.lhs().getDefiningOp())) {
    if (auto rhs = dyn_cast_or_null<ConcatOp>(op.rhs().getDefiningOp())) {
      auto rewriteResult = matchAndRewriteCompareConcat(op, lhs, rhs, rewriter);
      if (rewriteResult.succeeded()) {
        return success();
      }
    }
  }

  return failure();
}
