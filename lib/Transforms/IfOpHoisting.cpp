//===- IfOpHoisting.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <cstdint>
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::loopschedule;

namespace {
struct IfOpHoisting : public circt::IfOpHoistingBase<IfOpHoisting> {
  using IfOpHoistingBase<IfOpHoisting>::IfOpHoistingBase;
  void runOnOperation() override;
};
} // namespace

struct IfOpHoistingPattern : OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.elseBlock()) {
      return failure();
    }

    rewriter.modifyOpInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        // rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        // rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
        for (auto &innerOp :
             llvm::make_early_inc_range(op.thenBlock()->without_terminator())) {
          auto res = innerOp.walk([](Operation *op) {
            if (hasEffect<MemoryEffects::Write>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          });
          if (!res.wasInterrupted())
            rewriter.moveOpBefore(&innerOp, op);
        }
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        // rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        // rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
        for (auto &innerOp :
             llvm::make_early_inc_range(op.elseBlock()->without_terminator())) {
          auto res = innerOp.walk([](Operation *op) {
            if (hasEffect<MemoryEffects::Write>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          });
          if (!res.wasInterrupted())
            rewriter.moveOpBefore(&innerOp, op);
        }
      }
    });

    if (!op.thenBlock()->without_terminator().empty() || !op.elseBlock()) {
      return success();
    }

    if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
      return success();
    }

    auto thenOperands = op.thenYield().getOperands();
    auto elseOperands = op.elseYield().getOperands();

    SmallVector<Value> newVals;
    for (auto val : llvm::zip(thenOperands, elseOperands)) {
      SmallVector<Value> operands;
      operands.push_back(op.getCondition());
      operands.push_back(std::get<0>(val));
      operands.push_back(std::get<1>(val));
      auto selectOp = rewriter.create<arith::SelectOp>(op.getLoc(), operands);
      newVals.push_back(selectOp.getResult());
    }

    rewriter.replaceOp(op, newVals);

    return success();
  }
};

static bool ifOpLegalityCallback(scf::IfOp op) {
  auto res = op.getThenRegion().walk([&](Operation *op) {
    if (isa<scf::YieldOp>(op))
      return WalkResult::advance();
    if (!hasEffect<MemoryEffects::Write>(op))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadInterface>(op)) {
      if (load.isDynamic())
        return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  bool notOnlyWriteAndYield = res.wasInterrupted();

  res = op.getThenRegion().walk([&](Operation *op) {
    if (hasEffect<MemoryEffects::Write>(op))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadInterface>(op)) {
      if (load.isDynamic())
        return WalkResult::advance();
    }
    return WalkResult::advance();
  });
  bool hasWrite = res.wasInterrupted();

  if (notOnlyWriteAndYield && !hasWrite)
    return false;

  if (op.elseBlock()) {
    auto res = op.getElseRegion().walk([&](Operation *op) {
      if (isa<scf::YieldOp>(op))
        return WalkResult::advance();
      if (!hasEffect<MemoryEffects::Write>(op))
        return WalkResult::interrupt();
      if (auto load = dyn_cast<LoadInterface>(op)) {
        if (load.isDynamic())
          return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    bool notOnlyWriteAndYield = res.wasInterrupted();

    res = op.getElseRegion().walk([&](Operation *op) {
      if (hasEffect<MemoryEffects::Write>(op))
        return WalkResult::interrupt();
      if (auto load = dyn_cast<LoadInterface>(op)) {
        if (load.isDynamic())
          return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    bool hasWrite = res.wasInterrupted();

    if (notOnlyWriteAndYield && !hasWrite)
      return false;
  }

  return true;
}

void IfOpHoisting::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         affine::AffineDialect, memref::MemRefDialect,
                         func::FuncDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<scf::IfOp>(ifOpLegalityCallback);

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<IfOpHoistingPattern>(ctx);

  auto op = getOperation();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
std::unique_ptr<mlir::Pass> createIfOpHoistingPass() {
  return std::make_unique<IfOpHoisting>();
}
} // namespace circt
