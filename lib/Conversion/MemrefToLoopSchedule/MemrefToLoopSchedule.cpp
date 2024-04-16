//===- MemrefToLoopSchedule.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/MemrefToLoopSchedule.h"
#include "../PassDetail.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "memref-to-loopschedule"

using namespace mlir;
using namespace mlir::memref;
using namespace circt;
using namespace circt::loopschedule;

namespace {

struct MemrefToLoopSchedule
    : public MemrefToLoopScheduleBase<MemrefToLoopSchedule> {
  using MemrefToLoopScheduleBase::MemrefToLoopScheduleBase;
  void runOnOperation() override;
};

} // namespace

struct ReplaceMemrefLoad : OpConversionPattern<memref::LoadOp> {
public:
  ReplaceMemrefLoad(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<LoopScheduleLoadOp>(
        op.getOperation(), op.getResult().getType(), op.getMemRef(),
        op.getIndices());
    auto accessName =
        op->getAttrOfType<StringAttr>(NameAnalysis::getAttributeName());
    newOp->setAttr(NameAnalysis::getAttributeName(), accessName);
    return success();
  }
};

struct ReplaceMemrefStore : OpConversionPattern<memref::StoreOp> {
public:
  ReplaceMemrefStore(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<LoopScheduleStoreOp>(
        op.getOperation(), op.getValueToStore(), op.getMemRef(),
        op.getIndices());
    auto accessName =
        op->getAttrOfType<StringAttr>(NameAnalysis::getAttributeName());
    newOp->setAttr(NameAnalysis::getAttributeName(), accessName);
    return success();
  }
};

void MemrefToLoopSchedule::runOnOperation() {
  auto &context = getContext();
  ConversionTarget target(context);
  target.addLegalDialect<LoopScheduleDialect>();
  target.addIllegalOp<memref::LoadOp>();
  target.addIllegalOp<memref::StoreOp>();

  auto *ctx = &context;
  RewritePatternSet patterns(ctx);

  patterns.add<ReplaceMemrefLoad>(ctx);
  patterns.add<ReplaceMemrefStore>(ctx);

  auto op = getOperation();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createMemrefToLoopSchedulePass() {
  return std::make_unique<MemrefToLoopSchedule>();
}
