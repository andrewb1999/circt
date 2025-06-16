//===- AffineToSCF.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToSCF.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

#define DEBUG_TYPE "affine-to-scf"

namespace circt {
#define GEN_PASS_DEF_AFFINETOSCF
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;
using namespace circt::loopschedule;

namespace {

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
class AffineLoadLowering : public OpRewritePattern<AffineLoadOp> {
public:
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  AffineLoadLowering(MLIRContext *ctx) : OpRewritePattern(ctx, 2){};

  LogicalResult matchAndRewrite(AffineLoadOp affineLoadOp,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(affineLoadOp.getMapOperands());
    auto resultOperands = expandAffineMap(rewriter, affineLoadOp.getLoc(),
                                          affineLoadOp.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    auto accessName = affineLoadOp->getAttrOfType<StringAttr>(
        NameAnalysis::getAttributeName());
    // Replace with simple load operation and keep correspondance between the
    // two operations
    memref::LoadOp loadOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        affineLoadOp, affineLoadOp.getMemRef(), *resultOperands);
    if (affineLoadOp->hasAttrOfType<StringAttr>(
            NameAnalysis::getAttributeName()))
      loadOp->setAttr(NameAnalysis::getAttributeName(), accessName);

    return success();
  }
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
class AffineStoreLowering : public OpRewritePattern<AffineStoreOp> {
public:
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  AffineStoreLowering(MLIRContext *ctx) : OpRewritePattern(ctx, 2){};

  LogicalResult matchAndRewrite(AffineStoreOp affineStoreOp,
                                PatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(affineStoreOp.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, affineStoreOp.getLoc(),
                        affineStoreOp.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    auto accessName = affineStoreOp->getAttrOfType<StringAttr>(
        NameAnalysis::getAttributeName());
    // Replace with simple store operation and keep correspondance between the
    // two operations
    memref::StoreOp storeOp = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        affineStoreOp, affineStoreOp.getValueToStore(),
        affineStoreOp.getMemRef(), *maybeExpandedMap);
    if (affineStoreOp->hasAttrOfType<StringAttr>(
            NameAnalysis::getAttributeName()))
      storeOp->setAttr(NameAnalysis::getAttributeName(), accessName);

    return success();
  }
};

class AffineForLowering : public OpRewritePattern<AffineForOp> {
public:
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto tripCount = getConstantTripCount(op);
    Value lowerBound = lowerAffineLowerBound(op, rewriter);
    Value upperBound = lowerAffineUpperBound(op, rewriter);
    Value step =
        rewriter.create<arith::ConstantIndexOp>(loc, op.getStepAsInt());
    auto scfForOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                step, op.getInits());

    if (tripCount.has_value()) {
      scfForOp->setAttr("loopschedule.trip_count",
                        rewriter.getI64IntegerAttr(tripCount.value()));
    }

    auto hasPipeline = op->hasAttr("hls.pipeline");
    auto unrollAttr = op->getAttrOfType<IntegerAttr>("hls.unroll");
    rewriter.eraseBlock(scfForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(),
                                scfForOp.getRegion().end());

    if (hasPipeline)
      scfForOp->setAttr("hls.pipeline", op->getAttr("hls.pipeline"));

    if (op->hasAttrOfType<IntegerAttr>("hls.unroll"))
      scfForOp->setAttr("hls.unroll", unrollAttr);

    rewriter.replaceOp(op, scfForOp.getResults());
    return success();
  }
};

class SchedulableAffineInterfaceLowering
    : public OpInterfaceRewritePattern<SchedulableAffineInterface> {
public:
  using OpInterfaceRewritePattern<
      SchedulableAffineInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(SchedulableAffineInterface schedInterface,
                                PatternRewriter &rewriter) const override {
    auto *op = schedInterface.getOperation();
    SmallVector<Value, 8> indices;
    AffineMap affineMap;
    if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      affineMap = writeOp.getAffineMap();
      indices.append(writeOp.getMapOperands().begin(),
                     writeOp.getMapOperands().end());
    } else if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
      affineMap = readOp.getAffineMap();
      indices.append(readOp.getMapOperands().begin(),
                     readOp.getMapOperands().end());
    } else {
      return failure();
    }

    // Expand affine map from 'affineReadOpInterface'.
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op->getLoc(), affineMap, indices);
    if (!maybeExpandedMap.has_value())
      return failure();

    auto accessName =
        op->getAttrOfType<StringAttr>(NameAnalysis::getAttributeName());
    // Build non-affine op from expandedMap.
    rewriter.setInsertionPoint(op);
    auto *newOp = schedInterface.createNonAffineOp(rewriter, *maybeExpandedMap);
    if (schedInterface->hasAttrOfType<StringAttr>(
            NameAnalysis::getAttributeName()))
      newOp->setAttr(NameAnalysis::getAttributeName(), accessName);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

} // namespace

namespace {
class AffineToSCFPass : public circt::impl::AffineToSCFBase<AffineToSCFPass> {
  void runOnOperation() override;
};
} // namespace

void AffineToSCFPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  populateAffineToStdConversionPatterns(patterns);
  patterns.add<AffineLoadLowering, AffineStoreLowering, AffineForLowering>(ctx,
                                                                           2);
  patterns.add<SchedulableAffineInterfaceLowering>(ctx);

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();
  target.addIllegalDialect<AffineDialect>();
  target.markUnknownOpDynamicallyLegal(
      [](Operation *op) -> std::optional<bool> {
        if (isa<SchedulableAffineInterface>(op))
          return false;
        if (isa<LoadInterface, StoreInterface>(op))
          return true;
        return std::nullopt;
      });
  if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createAffineToSCFPass() {
  return std::make_unique<AffineToSCFPass>();
}
