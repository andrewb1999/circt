//===- IfOpHoisting.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
    rewriter.modifyOpInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
      }
    });

    return success();
  }
};

struct IfToSelectPattern : OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.thenBlock()->without_terminator().empty() || !op.elseBlock()) {
      return failure();
    }

    if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
      return failure();
    }

    auto thenOperands = op.thenYield().getOperands();
    auto elseOperands = op.elseYield().getOperands();

    for (auto iv : llvm::enumerate(llvm::zip(thenOperands, elseOperands))) {
      auto i = iv.index();
      auto v = iv.value();
      SmallVector<Value> operands;
      operands.push_back(op.getCondition());
      operands.push_back(std::get<0>(v));
      operands.push_back(std::get<1>(v));
      auto selectOp = rewriter.create<arith::SelectOp>(op.getLoc(), operands);
      auto ifRes = op.getResult(i);
      rewriter.replaceAllUsesWith(ifRes, selectOp.getResult());
    }

    rewriter.eraseOp(op);

    return success();
  }
};

static bool ifOpLegalityCallback(scf::IfOp op) {
  auto resThen = op.getThenRegion().walk([&](Operation *op) {
    if (hasEffect<MemoryEffects::Write>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (resThen.wasInterrupted())
    return true;

  if (op.elseBlock()) {
    auto resElse = op.getElseRegion().walk([&](Operation *op) {
      if (hasEffect<MemoryEffects::Write>(op))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (resElse.wasInterrupted())
      return true;
  } else {
    if (op.thenBlock()->without_terminator().empty())
      return true;
  }

  return false;
}

void IfOpHoisting::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         affine::AffineDialect, memref::MemRefDialect>();
  target.addDynamicallyLegalOp<scf::IfOp>(ifOpLegalityCallback);

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<IfOpHoistingPattern>(ctx);
  patterns.add<IfToSelectPattern>(ctx);

  auto op = getOperation();
  // op->dump();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
  // op->dump();
  // llvm::errs() << "=============================\n";
}

namespace circt {
std::unique_ptr<mlir::Pass> createIfOpHoistingPass() {
  return std::make_unique<IfOpHoisting>();
}
} // namespace circt
