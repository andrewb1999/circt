//===- ArithStrengthReduction.cpp
//---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include <cassert>
#include <cstdint>
#include <limits>

namespace circt {
#define GEN_PASS_DEF_ARITHSTRENGTHREDUCTION
#define GEN_PASS_DEF_ARITHSTRENGTHREDUCTIONCALLS
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

namespace {
struct ArithStrengthReduction
    : public circt::impl::ArithStrengthReductionBase<ArithStrengthReduction> {
  using ArithStrengthReductionBase::ArithStrengthReductionBase;
  void runOnOperation() override;
};
} // namespace

struct MulStrengthReduction : OpConversionPattern<MulIOp> {
  using OpConversionPattern<MulIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getRhs()))
      return failure();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto log = val.getValue().exactLogBase2();
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), log);
        auto shift = arith::ConstantOp::create(rewriter, op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(),
                                                   shift.getResult());
        return success();
      }

      if (val.getInt() == -1) {
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), 1);
        auto constOne = arith::ConstantOp::create(rewriter, op.getLoc(), attr);
        auto xorOp = arith::XOrIOp::create(rewriter, op.getLoc(), op.getLhs(),
                                           op.getRhs());
        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, xorOp.getResult(),
                                                   constOne);
        return success();
      }
    }

    return failure();
  }
};

struct RemUIStrengthReduction : OpConversionPattern<RemUIOp> {
  using OpConversionPattern<RemUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getRhs()))
      return failure();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto shifted = val.getValue() - 1;
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), shifted);
        auto shift = arith::ConstantOp::create(rewriter, op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(),
                                                   shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

struct RemSIStrengthReduction : OpConversionPattern<RemSIOp> {
  using OpConversionPattern<RemSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, op.getLhs(), op.getRhs());

    return success();
  }
};

struct DivSIStrengthReduction : OpConversionPattern<DivSIOp> {
  using OpConversionPattern<DivSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getRhs()))
      return failure();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto log = val.getValue().exactLogBase2();
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), log);
        auto shift = arith::ConstantOp::create(rewriter, op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, op.getLhs(),
                                                    shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

static bool mulLegalityCallback(Operation *op) {
  if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    if (isa<BlockArgument>(mulOp.getRhs()))
      return true;
    auto *rhsDef = mulOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
      if (cast<IntegerAttr>(constOp.getValue()).getInt() == -1) {
        return false;
      }
    }
  }
  return true;
}

static bool divSIOpLegalityCallback(Operation *op) {
  if (auto divOp = dyn_cast<arith::DivSIOp>(op)) {
    if (isa<BlockArgument>(divOp.getRhs()))
      return true;
    auto *rhsDef = divOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool remUILegalityCallback(Operation *op) {
  if (auto remOp = dyn_cast<arith::RemUIOp>(op)) {
    if (isa<BlockArgument>(remOp.getRhs()))
      return true;
    auto *rhsDef = remOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool remSILegalityCallback(Operation *op) {
  if (auto remOp = dyn_cast<arith::RemSIOp>(op)) {
    if (isa<BlockArgument>(remOp.getRhs()))
      return true;
    auto *rhsDef = remOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto rhsValue = cast<IntegerAttr>(constOp.getValue());
      if (rhsValue.getValue().exactLogBase2() != -1) {
        if (rhsValue.getInt() >= 0)
          return false;
      }
    }
  }
  return true;
}

void ArithStrengthReduction::runOnOperation() {
  auto &context = getContext();
  ConversionTarget target(context);
  target.addLegalDialect<AffineDialect, ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();

  auto *ctx = &context;
  RewritePatternSet patterns(ctx);

  patterns.add<MulStrengthReduction>(ctx);
  patterns.add<DivSIStrengthReduction>(ctx);
  patterns.add<RemUIStrengthReduction>(ctx);
  patterns.add<RemSIStrengthReduction>(ctx);

  target.addDynamicallyLegalOp<MulIOp>(mulLegalityCallback);
  target.addDynamicallyLegalOp<DivSIOp>(divSIOpLegalityCallback);
  target.addDynamicallyLegalOp<RemUIOp>(remUILegalityCallback);
  target.addDynamicallyLegalOp<RemSIOp>(remSILegalityCallback);

  auto op = getOperation();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
std::unique_ptr<mlir::Pass> createArithStrengthReductionPass() {
  return std::make_unique<ArithStrengthReduction>();
}
} // namespace circt
