//===- RemoveGroups.cpp - Remove Groups Pass --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Remove Groups pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HLS/HLSPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace circt;
using namespace hls;
using namespace mlir::affine;

namespace {

struct UnrollMarkedLoopsPass
    : public UnrollMarkedLoopsBase<UnrollMarkedLoopsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void UnrollMarkedLoopsPass::runOnOperation() {
  WalkResult result = getOperation().walk([](AffineForOp loop) {
    if (!loop->hasAttr("hls.unroll"))
      return WalkResult::advance();

    auto attr = loop->getAttr("hls.unroll");
    loop->removeAttr("hls.unroll");
    if (isa<StringAttr>(attr)) {
      auto val = cast<StringAttr>(attr).getValue();
      if (val != "full")
        return WalkResult::interrupt();

      if (loopUnrollFull(loop).failed())
        return WalkResult::interrupt();
    } else if (isa<IntegerAttr>(attr)) {
      auto val = cast<IntegerAttr>(attr).getInt();

      if (loopUnrollByFactor(loop, val).failed())
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::hls::createUnrollMarkedLoopsPass() {
  return std::make_unique<UnrollMarkedLoopsPass>();
}
