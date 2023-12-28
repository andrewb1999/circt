//===- BitwidthMinimization.cpp - Minimize bitwidth of registers *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print the module hierarchy.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;
using namespace loopschedule;

namespace {
struct BitwidthMinimizationPass
    : public circt::loopschedule::BitwidthMinimizationBase<BitwidthMinimizationPass> {
  void runOnOperation() override;

};
} // end anonymous namespace

struct IndexCastRemoval : OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op,
                  PatternRewriter &rewriter) const override {

    for (auto &use : llvm::make_early_inc_range(op.getOut().getUses())) {
      if (auto reg = dyn_cast<LoopScheduleRegisterOp>(use.getOwner())) {
        auto operandNum = use.getOperandNumber();
        auto phase = cast<PhaseInterface>(reg->getParentOp());
        auto result = phase->getResult(operandNum);
        for (auto &use : llvm::make_early_inc_range(result.getUses())) {
          rewriter.setInsertionPoint(use.getOwner());
          IRMapping valueMap;
          valueMap.map(op.getIn(), result);
          auto *newOp = rewriter.clone(*op.getOperation(), valueMap);
          auto newCast = cast<arith::IndexCastOp>(newOp);
          rewriter.updateRootInPlace(use.getOwner(),
                                     [&]() { use.set(newCast.getOut()); });
        }
        rewriter.updateRootInPlace(phase, [&]() { result.setType(op.getIn().getType()); });
      }

      rewriter.updateRootInPlace(use.getOwner(),
                                 [&]() { use.set(op.getIn()); });
    }

    rewriter.eraseOp(op.getOperation());

    return success();
  }
};

struct AddressSizeReduction : public RewritePattern {
public:
  AddressSizeReduction(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  PatternRewriter &rewriter) const override {
    if (!isa<LoopScheduleLoadOp, LoopScheduleStoreOp>(op)) {
      return failure();
    }

    // uint32_t numElems = 0;
    llvm::ArrayRef<int64_t> memrefShape;
    std::optional<MutableOperandRange> mutIndices;

    bool replaced = false;
    if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
      auto memrefType = load.getMemRefType();
      memrefShape = memrefType.getShape();
      mutIndices = load.getIndicesMutable();
    } else if (auto store = dyn_cast<LoopScheduleStoreOp>(op)) {
      auto memrefType = store.getMemRefType();
      memrefShape = memrefType.getShape();
      mutIndices = store.getIndicesMutable();
    }

    for (auto it : llvm::enumerate(*mutIndices)) {
      auto i = it.index();
      auto &index = it.value();
      uint32_t numBits = llvm::Log2_32_Ceil(memrefShape[i]);
      auto newType = rewriter.getIntegerType(numBits);
      if (index.get().getType() != newType) {
        replaced = true;
        rewriter.setInsertionPoint(op);
        auto newIndex = rewriter.create<arith::TruncIOp>(op->getLoc(), newType, index.get());
        index.set(newIndex);
      }
    }


    if (replaced)
      return success();

    return failure();
  }
};

struct TruncMoving : OpRewritePattern<arith::TruncIOp> {
  using OpRewritePattern<arith::TruncIOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op,
                  PatternRewriter &rewriter) const override {
    auto *definingOp = op.getIn().getDefiningOp();
    if (!isa<LoopSchedulePipelineStageOp, LoopScheduleStepOp>(definingOp)) {
      return failure();
    }
    op->getParentOfType<ModuleOp>().dump();

    auto result = cast<OpResult>(op.getIn());
    if (auto stage = dyn_cast<LoopSchedulePipelineStageOp>(definingOp)) {
      auto regOperand = stage.getRegisterOp().getOperand(result.getResultNumber());
      IRMapping valueMap;
      valueMap.map(op.getIn(), regOperand);
      rewriter.setInsertionPoint(stage.getRegisterOp());
      auto newOp = cast<arith::TruncIOp>(rewriter.clone(*op.getOperation(), valueMap));
      stage.getRegisterOp().setOperand(result.getResultNumber(), newOp.getOut());
      rewriter.updateRootInPlace(stage, [&]() { result.setType(newOp.getOut().getType()); });
    }

    rewriter.replaceAllUsesWith(op.getOut(), op.getIn());
    rewriter.eraseOp(op.getOperation());

    return success();
  }
};

void BitwidthMinimizationPass::runOnOperation() {
  auto *context = &getContext();
  auto op = getOperation();

  RewritePatternSet patterns(context);
  patterns.add<IndexCastRemoval>(context);
  patterns.add<AddressSizeReduction>(context);
  patterns.add<TruncMoving>(context);

  GreedyRewriteConfig config;
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config)))
    op->emitOpError("Failed to perform bitwidth minimization conversions");
}

std::unique_ptr<mlir::Pass> circt::loopschedule::createBitwidthMinimization() {
  return std::make_unique<BitwidthMinimizationPass>();
}
