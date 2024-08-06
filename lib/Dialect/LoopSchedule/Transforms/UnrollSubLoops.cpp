//===- UnrollSubLoops.cpp - Dependencies pass ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the UnrollSubLoops pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace loopschedule;
using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct UnrollSubLoopsPass : public UnrollSubLoopsBase<UnrollSubLoopsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

LogicalResult unrollSubLoops(scf::ForOp &forOp) {
  auto result = forOp.getBody()->walk<WalkOrder::PostOrder>([](scf::ForOp op) {
    std::optional<int64_t> lbCstOp = getConstantIntValue(op.getLowerBound());
    std::optional<int64_t> ubCstOp = getConstantIntValue(op.getUpperBound());
    std::optional<int64_t> stepCstOp = getConstantIntValue(op.getStep());
    if (!lbCstOp || !ubCstOp || !stepCstOp) {
      return WalkResult::interrupt();
    }
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = llvm::divideCeilSigned(ubCst - lbCst, stepCst);
    if (loopUnrollByFactor(op, tripCount).failed())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    forOp.emitOpError("Could not unroll sub loops");
    return failure();
  }

  return success();
}

void UnrollSubLoopsPass::runOnOperation() {
  auto funcOp = getOperation();

  // Collect loops to pipeline and work on them.
  SmallVector<scf::ForOp> loops;

  auto hasPipelinedParent = [](Operation *op) {
    Operation *currentOp = op;

    while (!isa<ModuleOp>(currentOp->getParentOp())) {
      if (currentOp->getParentOp()->hasAttr("hls.pipeline"))
        return true;
      currentOp = currentOp->getParentOp();
    }

    return false;
  };

  funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<scf::ForOp>(op) || !op->hasAttr("hls.pipeline"))
      return;

    if (hasPipelinedParent(op))
      return;

    loops.push_back(cast<scf::ForOp>(op));
  });

  // Unroll loops within this loop to make pipelining possible
  for (auto loop : llvm::make_early_inc_range(loops)) {
    if (failed(unrollSubLoops(loop)))
      return signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::loopschedule::createUnrollSubLoopsPass() {
  return std::make_unique<UnrollSubLoopsPass>();
}
