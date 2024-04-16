//===- BitwidthReductionForLoopSchedule.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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
struct BitwidthReductionForLoopSchedule
    : public BitwidthReductionForLoopScheduleBase<
          BitwidthReductionForLoopSchedule> {
  using BitwidthReductionForLoopScheduleBase<
      BitwidthReductionForLoopSchedule>::BitwidthReductionForLoopScheduleBase;
  void runOnOperation() override;
};
} // namespace

struct SCFForIterationReduction : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto constantLB = op.getLowerBound().getDefiningOp<ConstantOp>();
    auto constantUB = op.getUpperBound().getDefiningOp<ConstantOp>();
    auto constantStep = op.getStep().getDefiningOp<ConstantOp>();
    if (constantLB == nullptr || constantUB == nullptr ||
        constantStep == nullptr) {
      auto inductionType = op.getInductionVar().getType();
      auto bitwidth = isa<IndexType>(inductionType)
                          ? 64
                          : inductionType.getIntOrFloatBitWidth();
      auto newType = rewriter.getIntegerType(bitwidth);
      if (op.getInductionVar().getType() == newType)
        return failure();

      op.getInductionVar().setType(newType);
      rewriter.setInsertionPointToStart(&op.getRegion().front());
      auto newExt = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI64Type(), op.getInductionVar());
      rewriter.replaceAllUsesExcept(op.getInductionVar(), newExt.getOut(),
                                    newExt);
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
    auto newLB = rewriter.create<ConstantOp>(op.getLoc(), newLBAttr);
    op.setLowerBound(newLB);

    auto ubValue = cast<IntegerAttr>(constantUB.getValue()).getInt();
    auto newUBAttr = rewriter.getIntegerAttr(newType, ubValue);
    auto newUB = rewriter.create<ConstantOp>(op.getLoc(), newUBAttr);
    op.setUpperBound(newUB);

    auto stepValue = cast<IntegerAttr>(constantStep.getValue()).getInt();
    auto newStepAttr = rewriter.getIntegerAttr(newType, stepValue);
    auto newStep = rewriter.create<ConstantOp>(op.getLoc(), newStepAttr);
    op.setStep(newStep);

    induction.setType(newType);
    rewriter.setInsertionPointToStart(&op.getRegion().front());
    auto newExt = rewriter.create<arith::ExtSIOp>(
        op.getLoc(), rewriter.getI64Type(), induction);
    // auto newCast = rewriter.create<arith::IndexCastOp>(op.getLoc(),
    // rewriter.getIndexType(), newExt.getOut());
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
      newOp = rewriter.create<ExtUIOp>(op.getLoc(), outputType, extUI.getIn());
    } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
      newOp = rewriter.create<ExtSIOp>(op.getLoc(), outputType, extSI.getIn());
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
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
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
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
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
          auto newIdx =
              rewriter.create<arith::TruncIOp>(op.getLoc(), newType, idx.get());
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
          auto newIdx =
              rewriter.create<arith::TruncIOp>(op.getLoc(), newType, idx.get());
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
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
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
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
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
          auto newIdx =
              rewriter.create<arith::TruncIOp>(op.getLoc(), newType, idx.get());
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
          auto newIdx =
              rewriter.create<arith::TruncIOp>(op.getLoc(), newType, idx.get());
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

void populateIndexRemovalTypeConverter(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](Type type) -> std::optional<Type> { return type; });
  typeConverter.addConversion([](IndexType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &b, Type target, ValueRange input, Location loc) {
        return b.create<UnrealizedConversionCastOp>(loc, target, input)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &b, Type source, ValueRange input, Location loc) {
        return b.create<UnrealizedConversionCastOp>(loc, source, input)
            .getResult(0);
      });
}

struct IndexRemovalRewritePattern final : ConversionPattern {
public:
  IndexRemovalRewritePattern(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    const TypeConverter *converter = getTypeConverter();
    if (converter->isLegal(op))
      return rewriter.notifyMatchFailure(loc, "op already legal");

    OperationState newOp(loc, op->getName());
    newOp.addOperands(operands);

    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), newResultTypes)))
      return rewriter.notifyMatchFailure(loc, "couldn't convert return types");
    newOp.addTypes(newResultTypes);
    newOp.addAttributes(op->getAttrs());
    Operation *legalized = rewriter.create(newOp);
    SmallVector<Value> results = legalized->getResults();
    if (op->hasAttrOfType<StringAttr>(NameAnalysis::getAttributeName())) {
      auto accessName =
          op->getAttrOfType<StringAttr>(NameAnalysis::getAttributeName());
      legalized->setAttr(NameAnalysis::getAttributeName(), accessName);
    }
    rewriter.replaceOp(op, results);

    return success();
  }
};

struct ConstIndexRemovalRewritePattern final
    : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto val = op.getValue();
    if (!isa<IndexType>(val.getType()))
      return failure();

    auto integerAttr = dyn_cast_or_null<IntegerAttr>(val);
    if (!integerAttr)
      return failure();

    auto intVal = integerAttr.getInt();

    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            rewriter.getI64IntegerAttr(intVal));

    return success();
  }
};

struct IndexCastRemovalRewritePattern final
    : public OpConversionPattern<IndexCastOp> {
public:
  using OpConversionPattern<IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getOut().getType(), op.getIn());

    return success();
  }
};

void BitwidthReductionForLoopSchedule::runOnOperation() {
  auto op = getOperation();
  auto &context = getContext();

  // Minimize SCFFor iteration argument bitwidth to enable further bitwidth
  // reduction
  RewritePatternSet patterns(&context);
  patterns.add<SCFForIterationReduction>(&context);

  GreedyRewriteConfig config;
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  // Remove index types to enable bitwidth reduction of indices
  TypeConverter typeConverter;
  populateIndexRemovalTypeConverter(typeConverter);
  ConversionTarget target(context);
  target.addDynamicallyLegalDialect<ArithDialect>(
      [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalDialect<LoopScheduleDialect>(
      [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  // target.addDynamicallyLegalDialect<scf::SCFDialect>(
  //     [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal(
      [&typeConverter](Operation *op) -> std::optional<bool> {
        if (!isa<LoadInterface>(op) && !isa<StoreInterface>(op))
          return std::nullopt;
        return typeConverter.isLegal(op);
      });
  target.addIllegalOp<arith::IndexCastOp>();
  patterns.clear();
  patterns.add<IndexRemovalRewritePattern>(typeConverter, &context);
  patterns.add<ConstIndexRemovalRewritePattern>(typeConverter, &context);
  patterns.add<IndexCastRemovalRewritePattern>(typeConverter, &context);
  populateReconcileUnrealizedCastsPatterns(patterns);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  // Apply the core integer narrowing pass
  patterns.clear();
  SmallVector<unsigned> bitwidthsSupported;
  for (unsigned i = 1; i <= 128; ++i) {
    bitwidthsSupported.push_back(i);
  }
  populateArithIntNarrowingPatterns(
      patterns, ArithIntNarrowingOptions{bitwidthsSupported});
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<SCFForCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);
  patterns.add<LoadAddressNarrowingPattern>(&context);
  patterns.add<StoreAddressNarrowingPattern>(&context);
  patterns.add<LoadInterfaceCleanupPattern>(&context);
  patterns.add<StoreInterfaceCleanupPattern>(&context);
  patterns.add<LoadInterfaceAddressNarrowingPattern>(&context);
  patterns.add<StoreInterfaceAddressNarrowingPattern>(&context);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    signalPassFailure();
  }

  auto res = op->walk([](Operation *op) {
    if (isa<UnrealizedConversionCastOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    op->emitOpError(
        "Bitwidth minimization failed to remove UnrealizedConversionCastOps");
    signalPassFailure();
  }

  // Perform dead code elimination again before scheduling
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
