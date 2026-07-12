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
// op that shares (or competes for) a hardware resource.
//
// Two kinds of operations enter the binding problem:
//
//  * CAPPED resources — ops whose `oplib.operator` entry declares a
//    `limit` (memory accesses arrive this way: `OperatorAllocation` emits
//    a per-memref operator with `limit = 2` and tags loads/stores). The
//    binder packs them into the declared pool and fails if the schedule
//    genuinely needs more instances than the cap.
//
//  * SHARED operators — ops whose operator is backed by an extern
//    hw.instance (a physical unit, e.g. a pipelined multiplier) but
//    declares no limit. These bind in "share mode": the pool is sized to
//    the op count, so the greedy binder simply assigns the MINIMAL number
//    of instances consistent with the schedule — ops whose occupancy
//    windows can never overlap (disjoint concurrency groups, or disjoint
//    AP-of-intervals within a group) coalesce onto one instance, and
//    genuinely concurrent ops keep their own. Zero schedule impact by
//    construction. Comb-lowered operators (plain adders etc.) have no
//    instance to share and are left untagged.
//
// The downstream FSM lowering keys extern-instance creation on
// (operator, binding id) within each generated hw.module, muxing each
// user's operands in under its activation gate — see
// `emitHwInstanceFromOperator` in LoopScheduleToFSM.
//
// Occupancy latency is 1 for compute operators: every extern operator in
// the current library is fully pipelined (accepts new operands every
// enabled cycle), so only ISSUE-cycle collisions matter — in-flight
// computations coexist in the unit's internal pipe. A future non-pipelined
// operator (e.g. a sequential divider) must pass its full occupancy here.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/LoopScheduleStartTimeAnalysis.h"
#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Binding/Algorithms.h"
#include "circt/Binding/Problems.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "circt/Dialect/OpLib/OpLibOps.h"

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
  using circt::loopschedule::impl::LoopScheduleBindingBase<
      LoopScheduleBindingPass>::LoopScheduleBindingBase;
  void runOnOperation() override;

  /// Build a `BindingProblem` for \p funcOp's scheduled ops and run the
  /// greedy binder. Stamps `loopschedule.binding = K` on each
  /// participating op.
  LogicalResult bindFunction(Operation *funcOp);
};

/// Is \p op a memory access (as opposed to a compute op)? Memory port
/// assignment interacts with the AMC port-group binding flow, so pipeline
/// integrations may exclude it via the `bind-memories` option.
bool isMemoryAccess(Operation *op) {
  return isa<LoopScheduleLoadOp, LoopScheduleStoreOp>(op) ||
         isa<LoadInterface, StoreInterface>(op);
}

/// Does \p oprAttr's hw_match instantiate an extern module? Only such
/// operators have a physical instance that share-mode binding can
/// coalesce; comb-matched operators lower to inline logic per use.
bool isExternInstanceOperator(analysis::OperatorLibraryAnalysis &ola,
                              SymbolRefAttr oprAttr) {
  StringRef opName = ola.getOperatorBySymbol(oprAttr);
  oplib::HwMatchOp hwMatch = ola.getHwMatchOp(opName);
  if (!hwMatch)
    return false;
  for (auto &op : *hwMatch.getBodyBlock())
    if (isa<oplib::HwInstanceOp>(op))
      return true;
  return false;
}
} // namespace

LogicalResult LoopScheduleBindingPass::bindFunction(Operation *funcOp) {
  analysis::LoopScheduleStartTimeAnalysis stAnalysis(funcOp);
  analysis::OperatorLibraryAnalysis opLibAnalysis(funcOp);
  binding::BindingProblem prob(funcOp);

  // Phase 1 — collect candidates. Share-mode resources size their pool to
  // the op count, so counts must be known before limits are set.
  struct Candidate {
    Operation *op;
    StringRef oprName;
    std::optional<unsigned> declaredLimit;
  };
  SmallVector<Candidate> candidates;
  llvm::StringMap<unsigned> oprOpCount;
  for (Operation *op : stAnalysis.getBindableOps()) {
    auto oprAttr = op->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    if (!oprAttr)
      continue;
    auto oprName = oprAttr.getLeafReference().getValue();
    auto limit = opLibAnalysis.getOperatorLimit(oprName);
    if (isMemoryAccess(op)) {
      // Memory accesses only bind against a declared port cap.
      if (!bindMemories || !limit)
        continue;
    } else if (!limit) {
      // No cap declared: bind in share mode when the operator is backed
      // by a physical instance; leave comb-lowered operators alone.
      if (!shareOperators ||
          !isExternInstanceOperator(opLibAnalysis, oprAttr))
        continue;
    }
    candidates.push_back({op, oprName, limit});
    ++oprOpCount[oprName];
  }

  if (candidates.empty())
    return success();

  // Phase 2 — populate the problem.
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

  for (auto &cand : candidates) {
    Operation *op = cand.op;
    unsigned limit =
        cand.declaredLimit ? *cand.declaredLimit : oprOpCount[cand.oprName];
    prob.insertOperation(op);
    prob.setStartTime(op, *stAnalysis.getStartTime(op));
    // Fully-pipelined units: only the issue cycle occupies the instance
    // (see the file header). Non-pipelined multi-cycle operators would
    // need their real occupancy here.
    prob.setLatency(op, 1);
    if (auto period = stAnalysis.getPeriod(op); period && *period > 0)
      prob.setPeriod(op, *period);
    if (auto count = stAnalysis.getCount(op); count && *count > 1)
      prob.setCount(op, *count);
    if (auto *group = stAnalysis.getConcurrencyGroup(op))
      prob.setConcurrencyGroup(op, group);
    prob.setLinkedResourceType(op, getOprRsrc(cand.oprName, limit));
  }

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

std::unique_ptr<Pass> circt::loopschedule::createLoopScheduleBindingPass(
    const LoopScheduleBindingOptions &options) {
  return std::make_unique<LoopScheduleBindingPass>(options);
}
