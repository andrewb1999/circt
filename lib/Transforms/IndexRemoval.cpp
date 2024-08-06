//===- IndexRemoval.cpp ---------------------------------------------------===//
//
// Partially adapted from Dynamatic https://github.com/EPFL-LAP/dynamatic
//
//-------------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace circt;
using namespace circt::loopschedule;

namespace {
struct IndexRemoval : public circt::IndexRemovalBase<IndexRemoval> {
  using IndexRemovalBase<IndexRemoval>::IndexRemovalBase;
  void runOnOperation() override;
};
} // namespace

struct ReplaceConstantOpAttr : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  ReplaceConstantOpAttr(MLIRContext *ctx, unsigned width)
      : OpRewritePattern<arith::ConstantOp>(ctx), width(width) {}

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    if (!isa<IndexType>(constantOp.getValueAttr().getType()))
      return failure();

    auto newAttr =
        IntegerAttr::get(IntegerType::get(getContext(), width),
                         constantOp.getValue().cast<IntegerAttr>().getInt());
    constantOp.setValueAttr(newAttr);

    return success();
  }

private:
  /// The width to concretize IndexType's with.
  unsigned width;
};

/// Replaces an index cast with an equivalent truncation/extension operations
/// (or with nothing if widths happen to match).
template <typename Op, typename OpExt>
struct ReplaceIndexCast : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op indexCastOp,
                                PatternRewriter &rewriter) const override {
    Value fromVal = indexCastOp.getOperand();
    Value toVal = indexCastOp.getResult();
    unsigned fromWidth = fromVal.getType().getIntOrFloatBitWidth();
    unsigned toWidth = toVal.getType().getIntOrFloatBitWidth();

    if (fromWidth == toWidth)
      // Simply bypass the cast operation if widths match
      rewriter.replaceOp(indexCastOp, fromVal);
    else {
      // Insert an explicit truncation/extension operation to replace the
      // index cast
      rewriter.setInsertionPoint(indexCastOp);
      Operation *castOp;
      if (fromWidth < toWidth)
        castOp = rewriter.create<OpExt>(indexCastOp.getLoc(), toVal.getType(),
                                        fromVal);
      else
        castOp = rewriter.create<arith::TruncIOp>(indexCastOp.getLoc(),
                                                  toVal.getType(), fromVal);
      rewriter.replaceOp(indexCastOp, castOp->getResult(0));
      // inheritBB(indexCastOp, castOp);
    }

    return success();
  }
};

/// Replaces all IndexType arguments/results in a function's
/// signature.
struct ReplaceFuncSignature : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  ReplaceFuncSignature(MLIRContext *ctx, unsigned width)
      : OpRewritePattern<func::FuncOp>(ctx), width(width) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto isNotIndexType = [](Type type) { return !isa<IndexType>(type); };

    auto sameOrIndexToInt = [&](Type type) -> Type {
      if (isNotIndexType(type))
        return type;
      return IntegerType::get(type.getContext(), width);
    };

    // Check if there is any index type in the function signature
    if (llvm::all_of(funcOp.getArgumentTypes(), isNotIndexType) &&
        llvm::all_of(funcOp.getResultTypes(), isNotIndexType))
      return failure();

    // Recreate a list of function arguments with index types replaced
    SmallVector<Type, 8> argTypes;
    for (auto &argType : funcOp.getArgumentTypes())
      argTypes.push_back(sameOrIndexToInt(argType));

    // Recreate a list of function results with index types replaced
    SmallVector<Type, 8> resTypes;
    for (auto resType : funcOp.getResultTypes())
      resTypes.push_back(sameOrIndexToInt(resType));

    // Replace the function's signature
    rewriter.modifyOpInPlace(funcOp, [&] {
      auto funcType = rewriter.getFunctionType(argTypes, resTypes);
      funcOp.setFunctionType(funcType);
    });
    return success();
  }

private:
  /// The width to concretize IndexType's with.
  unsigned width;
};

void IndexRemoval::runOnOperation() {
  auto &context = getContext();
  auto indexWidthInt = IntegerType::get(&context, 64);

  // Change the type of all SSA values with an IndexType
  WalkResult walkRes = getOperation().walk([&](Operation *op) {
    for (Value operand : op->getOperands())
      if (isa<IndexType>(operand.getType()))
        operand.setType(indexWidthInt);
    for (OpResult result : op->getResults())
      if (isa<IndexType>(result.getType()))
        result.setType(indexWidthInt);
    for (auto &region : op->getRegions()) {
      for (auto arg : region.getArguments()) {
        if (isa<IndexType>(arg.getType()))
          arg.setType(indexWidthInt);
      }
    }


    return WalkResult::advance();
  });

  if (walkRes.wasInterrupted())
    return signalPassFailure();

  auto *ctx = &context;
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  RewritePatternSet patterns{ctx};
  patterns.add<ReplaceConstantOpAttr>(ctx, 64);
  patterns.add<ReplaceIndexCast<arith::IndexCastOp, arith::ExtSIOp>,
               ReplaceIndexCast<arith::IndexCastUIOp, arith::ExtUIOp>>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                          config)))
    return signalPassFailure();
}

namespace circt {
std::unique_ptr<mlir::Pass> createIndexRemovalPass() {
  return std::make_unique<IndexRemoval>();
}
} // namespace circt
