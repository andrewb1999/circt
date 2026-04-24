//===- LoopScheduleBinding.cpp - Hardware instance binding ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Walks each loopschedule func-level container, runs the Binding framework
// on its scheduled ops, and stamps `loopschedule.binding = K : i64` on each
// op that competes for a limited hardware resource.
//
// Every op that carries `loopschedule.operator = @name` is considered for
// binding. If the operator's `oplib.operator` entry declares a `limit`,
// the op enters the binding problem and is assigned an instance id; if
// not, the op is left alone (no cap to enforce). Memory ops go through
// the same path — `OperatorAllocation` emits a per-memref `oplib.operator`
// with `limit = 2` and tags loads/stores accordingly, so memory and
// arith binding share one code path.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/LoopScheduleStartTimeAnalysis.h"
#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Binding/Algorithms.h"
#include "circt/Binding/Problems.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

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

  /// Build a `BindingProblem` for \p funcOp's memory accesses and run the
  /// greedy binder. Stamps `loopschedule.binding = K` on each participating
  /// op.
  LogicalResult bindFunction(Operation *funcOp);
};
} // namespace

LogicalResult LoopScheduleBindingPass::bindFunction(Operation *funcOp) {
  analysis::LoopScheduleStartTimeAnalysis stAnalysis(funcOp);
  analysis::OperatorLibraryAnalysis opLibAnalysis(funcOp);
  binding::BindingProblem prob(funcOp);

  // Resource type per operator symbol. Only operators with a declared
  // `limit` on their `oplib.operator` entry enter the binding problem —
  // uncapped operators are free to duplicate hardware per op, which the
  // binder can't constrain anyway. Memory accesses arrive here through
  // the same channel: `OperatorAllocation` attaches a per-memref
  // `oplib.operator` with `limit = 2` and stamps loads/stores with the
  // matching `loopschedule.operator`.
  DenseMap<StringRef, binding::BindingProblem::ResourceType> oprRsrc;
  auto getOprRsrc = [&](StringRef oprName, unsigned limit)
      -> binding::BindingProblem::ResourceType {
    auto it = oprRsrc.find(oprName);
    if (it != oprRsrc.end())
      return it->second;
    auto rsrc = prob.getOrInsertResourceType(oprName);
    prob.setInstanceLimit(rsrc, limit);
    oprRsrc[oprName] = rsrc;
    return rsrc;
  };

  for (Operation *op : stAnalysis.getBindableOps()) {
    auto oprAttr = op->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    if (!oprAttr)
      continue;
    auto oprName = oprAttr.getLeafReference().getValue();
    auto limit = opLibAnalysis.getOperatorLimit(oprName);
    if (!limit)
      continue; // no cap declared — don't bind this op

    prob.insertOperation(op);
    prob.setStartTime(op, *stAnalysis.getStartTime(op));
    prob.setLatency(op, 1);
    if (auto period = stAnalysis.getPeriod(op); period && *period > 0)
      prob.setPeriod(op, *period);
    if (auto count = stAnalysis.getCount(op); count && *count > 1)
      prob.setCount(op, *count);
    if (auto *group = stAnalysis.getConcurrencyGroup(op))
      prob.setConcurrencyGroup(op, group);
    prob.setLinkedResourceType(op, getOprRsrc(oprName, *limit));
  }

  if (prob.getOperations().empty())
    return success();

  if (failed(binding::bindGreedy(prob)))
    return failure();

  auto *ctx = funcOp->getContext();
  auto i64 = IntegerType::get(ctx, 64);
  for (Operation *op : prob.getOperations())
    if (auto inst = prob.getInstance(op))
      op->setAttr("loopschedule.binding", IntegerAttr::get(i64, *inst));
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
