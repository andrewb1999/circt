//===- LoopScheduleBinding.cpp - Hardware instance binding ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Bare-bones pass that walks each loopschedule func-level container and
// (eventually) runs the Binding framework to assign hardware instance ids
// to scheduled ops. Today it is a no-op stub: it iterates the expected
// containers so downstream plumbing (registration, pipeline insertion, lit
// tests) can land ahead of the actual binding logic.
//
//===----------------------------------------------------------------------===//

#include "circt/Binding/Algorithms.h"
#include "circt/Binding/Problems.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"

namespace circt {
namespace loopschedule {
#define GEN_PASS_DEF_LOOPSCHEDULEBINDING
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"
} // namespace loopschedule
} // namespace circt

using namespace circt;
using namespace circt::loopschedule;
using namespace mlir;

namespace {
struct LoopScheduleBindingPass
    : public circt::loopschedule::impl::LoopScheduleBindingBase<
          LoopScheduleBindingPass> {
  void runOnOperation() override;

  /// Run binding on a single scheduled container. Currently a no-op; the
  /// per-op resource-linking and `bindLeftEdge` call will land here.
  LogicalResult bindFunction(Operation *funcOp);
};
} // namespace

LogicalResult LoopScheduleBindingPass::bindFunction(Operation *funcOp) {
  (void)funcOp;
  return success();
}

void LoopScheduleBindingPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  SmallVector<Operation *> containers;
  moduleOp.walk([&](LoopScheduleFuncSequentialOp f) {
    containers.push_back(f.getOperation());
  });
  moduleOp.walk([&](LoopScheduleFuncPipelineOp f) {
    containers.push_back(f.getOperation());
  });
  for (auto *fn : containers)
    if (failed(bindFunction(fn))) {
      signalPassFailure();
      return;
    }
}

std::unique_ptr<Pass>
circt::loopschedule::createLoopScheduleBindingPass() {
  return std::make_unique<LoopScheduleBindingPass>();
}
