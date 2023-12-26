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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace loopschedule;

namespace {
struct BitwidthMinimizationPass
    : public circt::loopschedule::BitwidthMinimizationBase<BitwidthMinimizationPass> {
  void runOnOperation() override;

};
} // end anonymous namespace

void BitwidthMinimizationPass::runOnOperation() {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<LoopScheduleDialect, arith::ArithDialect>();
}

std::unique_ptr<mlir::Pass> circt::loopschedule::createBitwidthMinimization() {
  return std::make_unique<BitwidthMinimizationPass>();
}
