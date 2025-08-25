//===- IndexSwitchToIf.cpp - Index switch to if-else pass ---*-C++-*-===//
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

namespace circt {
#define GEN_PASS_DEF_INDEXSWITCHTOIF
#define GEN_PASS_DEF_INDEXSWITCHTOIFCALLS
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::affine;

namespace {
struct IndexSwitchToIf
    : public circt::impl::IndexSwitchToIfBase<IndexSwitchToIf> {
  using IndexSwitchToIfBase<IndexSwitchToIf>::IndexSwitchToIfBase;
  void runOnOperation() override;
};
} // namespace

struct IndexSwitchToIfPattern : OpConversionPattern<scf::IndexSwitchOp> {
  using OpConversionPattern<scf::IndexSwitchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp switchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = switchOp.getLoc();

    Region &defaultRegion = switchOp.getDefaultRegion();
    bool hasResults = !switchOp.getResultTypes().empty();

    Value finalResult;
    scf::IfOp prevIfOp = nullptr;

    rewriter.setInsertionPointAfter(switchOp);
    auto switchCases = switchOp.getCases();
    for (size_t i = 0; i < switchCases.size(); i++) {
      auto caseValueInt = switchCases[i];
      if (prevIfOp)
        rewriter.setInsertionPointToStart(&prevIfOp.getElseRegion().front());

      Value caseValue =
          arith::ConstantIndexOp::create(rewriter, loc, caseValueInt);
      Value cond =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                switchOp.getOperand(), caseValue);

      auto ifOp = scf::IfOp::create(rewriter, loc, switchOp.getResultTypes(),
                                    cond, /*hasElseRegion=*/true);

      Region &caseRegion = switchOp.getCaseRegions()[i];
      rewriter.eraseBlock(&ifOp.getThenRegion().front());
      rewriter.inlineRegionBefore(caseRegion, ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());

      if (i + 1 == switchCases.size()) {
        rewriter.eraseBlock(&ifOp.getElseRegion().front());
        rewriter.inlineRegionBefore(defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
      }

      if (prevIfOp && hasResults) {
        rewriter.setInsertionPointToEnd(&prevIfOp.getElseRegion().front());
        scf::YieldOp::create(rewriter, loc, ifOp.getResult(0));
      }

      if (i == 0 && hasResults)
        finalResult = ifOp.getResult(0);

      prevIfOp = ifOp;
    }

    if (hasResults)
      rewriter.replaceOp(switchOp, finalResult);
    else
      rewriter.eraseOp(switchOp);

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
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

namespace circt {
std::unique_ptr<mlir::Pass> createIndexSwitchToIfPass() {
  return std::make_unique<IndexSwitchToIf>();
}
} // namespace circt
