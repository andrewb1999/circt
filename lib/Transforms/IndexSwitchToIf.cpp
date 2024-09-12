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

namespace {
struct IndexSwitchToIf : public circt::IndexSwitchToIfBase<IndexSwitchToIf> {
  using IndexSwitchToIfBase<IndexSwitchToIf>::IndexSwitchToIfBase;
  void runOnOperation() override;
};
} // namespace

struct IndexSwitchToIfPattern : OpConversionPattern<scf::IndexSwitchOp> {
  using OpConversionPattern<scf::IndexSwitchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> comparisons;
    for (unsigned int i = 0; i < op.getNumCases(); ++i) {
      auto constOp = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(op.getCases()[i]));
      auto cmpiOp = rewriter.create<arith::CmpIOp>(op.getLoc(), arith::CmpIPredicate::eq, op.getArg(), constOp.getResult());
      comparisons.push_back(cmpiOp.getResult());
    }

    scf::IfOp outerIfOp;
    for (unsigned int i = 0; i < op.getNumCases(); ++i) {
      bool lastCase = i == op.getNumCases() - 1;
      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(), comparisons[i], false, false);
      rewriter.inlineRegionBefore(op.getCaseRegions()[i], ifOp.getThenRegion(), ifOp.getThenRegion().end());
      if (lastCase) {
        rewriter.inlineRegionBefore(op.getDefaultRegion(), ifOp.getElseRegion(), ifOp.getElseRegion().end());
      } else {
        rewriter.createBlock(&ifOp.getElseRegion());
      }

      if (i == 0) {
        outerIfOp = ifOp;
      } else {
        rewriter.setInsertionPointAfter(ifOp);
        rewriter.create<scf::YieldOp>(op.getLoc(), ifOp.getResults());
      }
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
    }

    if (outerIfOp == nullptr)
      return failure();

    rewriter.replaceOp(op, outerIfOp);

    return success();
  }
};

void IndexSwitchToIf::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                         affine::AffineDialect, memref::MemRefDialect>();
  target.addIllegalOp<scf::IndexSwitchOp>();

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<IndexSwitchToIfPattern>(ctx);

  auto op = getOperation();
  // op->dump();
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
  // op->dump();
  // llvm::errs() << "=============================\n";
}

namespace circt {
std::unique_ptr<mlir::Pass> createIndexSwitchToIfPass() {
  return std::make_unique<IndexSwitchToIf>();
}
} // namespace circt
