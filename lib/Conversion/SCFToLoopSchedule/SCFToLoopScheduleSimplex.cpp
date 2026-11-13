//===- SCFToLoopScheduleSimplex.cpp - SCF to LoopSchedule, simplex --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// convert-scf-to-loopschedule: each region scheduled on its own by the
// simplex, each op on the operator allocation chose.
//
//===----------------------------------------------------------------------===//

#include "SCFToLoopScheduleLowering.h"
#include "circt/Conversion/SCFToLoopSchedule.h"
#include "circt/Dialect/OpLib/OpLibDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace circt {
#define GEN_PASS_DEF_SCFTOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::scheduling;
using namespace circt::scftoloopschedule;

namespace {

class SimplexLowering : public SCFToLoopScheduleLowering {
public:
  using SCFToLoopScheduleLowering::SCFToLoopScheduleLowering;

protected:
  LogicalResult scheduleModulo(ChainingModuloProblem &problem,
                               Operation *anchor, Operation *owner,
                               scf::WhileOp pipelinedLoop,
                               std::optional<int32_t> ii, Value condValue,
                               const InputDelays &delays) override {
    if (succeeded(simplexModulo(problem, anchor, ii, delays)))
      return success();
    if (isa<func::FuncOp>(owner))
      return owner->emitOpError(
          "failed to solve ChainingModuloProblem for pipelined func");
    return failure();
  }
  LogicalResult scheduleSequential(ChainingSharedOperatorsProblem &problem,
                                   Operation *anchor, Operation *owner,
                                   const InputDelays &delays) override {
    return simplexSequential(problem, anchor, delays);
  }
};

struct SCFToLoopSchedulePass
    : public circt::impl::SCFToLoopScheduleBase<SCFToLoopSchedulePass> {
  using SCFToLoopScheduleBase::SCFToLoopScheduleBase;
  void runOnOperation() override {
    LoweringConfig config;
    config.cycleTimeNs = cycleTimeNs;
    config.recordChainSlack = recordChainSlack;
    config.shareMuxDelays.assign(shareMuxDelays.begin(), shareMuxDelays.end());
    config.shareMuxNorDelays.assign(shareMuxNorDelays.begin(),
                                    shareMuxNorDelays.end());
    config.shareMuxDelays64.assign(shareMuxDelays64.begin(),
                                   shareMuxDelays64.end());
    config.shareMuxNorDelays64.assign(shareMuxNorDelays64.begin(),
                                      shareMuxNorDelays64.end());
    SimplexLowering lowering(std::move(config), getAnalysisManager());
    if (failed(lowering.run(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
circt::createSCFToLoopSchedulePass(const SCFToLoopScheduleOptions &options) {
  return std::make_unique<SCFToLoopSchedulePass>(options);
}
