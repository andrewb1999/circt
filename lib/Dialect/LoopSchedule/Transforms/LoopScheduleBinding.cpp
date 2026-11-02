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
// Occupancy latency is 1 for compute operators: every shared extern
// operator is fully pipelined (accepts new operands every enabled cycle),
// so only ISSUE-cycle collisions matter — in-flight computations coexist
// in the unit's internal pipe.
//
// Sharing puts a mux on each user's operands (a one-hot AND-OR of the
// users' operands, LoopScheduleToFSM), on the chain the scheduler placed
// the op on: the whole path for a combinational operator, the input path
// for a pipelined one. With `share-mux-delays`, an op shares an instance
// only if its `loopschedule.chain_slack` takes the mux that instance's
// user count needs; the users that do not are given instances of their own
// (ids past the shared ones) and the rest are bound again, until every
// shared user's slack holds. An operator that has to see the cycles its
// op issues (a `valid` hw_match argument: one with state, or one that takes
// an operand only every so many cycles) is its op's alone and never bound.
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
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <map>

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

/// Does \p oprAttr's hw_match take the cycles its op issues (a `valid`
/// argument)? Such an operator takes one step per execution of its op
/// (an accumulator a beat; a core that takes an operand only every so many
/// cycles, an operation): another op's executions would step it too, and
/// the FSM lowering drives `valid` from one op's gate.
bool takesIssueCycles(analysis::OperatorLibraryAnalysis &ola,
                      SymbolRefAttr oprAttr) {
  StringRef opName = ola.getOperatorBySymbol(oprAttr);
  oplib::HwMatchOp hwMatch = ola.getHwMatchOp(opName);
  return hwMatch && llvm::any_of(hwMatch.getArgRoles(), [](Attribute role) {
           return cast<StringAttr>(role).getValue() == "valid";
         });
}
} // namespace

LogicalResult LoopScheduleBindingPass::bindFunction(Operation *funcOp) {
  analysis::LoopScheduleStartTimeAnalysis stAnalysis(funcOp);
  analysis::OperatorLibraryAnalysis opLibAnalysis(funcOp);

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
    } else if (!isMemoryEffectFree(op) ||
               takesIssueCycles(opLibAnalysis, oprAttr)) {
      // An operator with state (an accumulator), or one that takes an
      // operand only every so many cycles, is its op's alone: another op's
      // executions would step it too.
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

  // The delay of the mux an instance shared by `users` ops puts on their
  // operands: AND-OR levels of 6-input LUTs, 3 users to a LUT, then ORs
  // of 6.
  auto muxDelay = [&](unsigned users) -> double {
    unsigned levels = 1;
    for (unsigned luts = (users + 2) / 3; luts > 1; luts = (luts + 5) / 6)
      ++levels;
    size_t given = shareMuxDelays.size();
    if (levels <= given)
      return shareMuxDelays[levels - 1];
    double step = given > 1 ? shareMuxDelays[given - 1] -
                                  shareMuxDelays[given - 2]
                            : shareMuxDelays[given - 1];
    return shareMuxDelays[given - 1] + step * (levels - given);
  };

  // Share-mode ops whose chain slack does not take their instance's mux:
  // each keeps an instance of its own.
  DenseSet<Operation *> alone;
  DenseMap<Operation *, int64_t> instanceOf;
  while (true) {
    // Phase 2 — populate the problem.
    binding::BindingProblem prob(funcOp);
    llvm::StringMap<unsigned> shareCount;
    for (auto &cand : candidates)
      if (!alone.contains(cand.op))
        ++shareCount[cand.oprName];
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
      if (alone.contains(op))
        continue;
      unsigned limit =
          cand.declaredLimit ? *cand.declaredLimit : shareCount[cand.oprName];
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

    instanceOf.clear();
    for (Operation *op : prob.getOperations())
      if (auto inst = prob.getInstance(op))
        instanceOf[op] = *inst;
    if (shareMuxDelays.empty())
      break;

    // The users of each shared instance; of those whose slack does not
    // take its mux, the one with the least leaves it, one an instance a
    // round (a smaller mux may then fit the rest).
    using InstanceKey = std::pair<std::string, int64_t>;
    std::map<InstanceKey, unsigned> users;
    for (auto &cand : candidates)
      if (!cand.declaredLimit)
        if (auto it = instanceOf.find(cand.op); it != instanceOf.end())
          ++users[{cand.oprName.str(), it->second}];
    std::map<InstanceKey, std::pair<double, Operation *>> tightest;
    for (auto &cand : candidates) {
      if (cand.declaredLimit)
        continue;
      auto it = instanceOf.find(cand.op);
      auto slack =
          cand.op->getAttrOfType<FloatAttr>("loopschedule.chain_slack");
      if (it == instanceOf.end() || !slack)
        continue;
      InstanceKey key{cand.oprName.str(), it->second};
      unsigned n = users[key];
      double s = slack.getValueAsDouble();
      if (n < 2 || s >= muxDelay(n))
        continue;
      auto [t, inserted] = tightest.try_emplace(key, s, cand.op);
      if (!inserted && s < t->second.first)
        t->second = {s, cand.op};
    }
    if (tightest.empty())
      break;
    for (auto &[key, worst] : tightest)
      alone.insert(worst.second);
  }

  // An op that stays off a shared instance gets an instance id of its own,
  // past the operator's shared ones: every share-mode op is bound, alone or
  // not, which the FSM lowering keys its instances and its result captures
  // on (a bound combinational instance's result is captured in its cycle,
  // not read live by the next one's ops, as the schedule has it).
  llvm::StringMap<int64_t> nextInstance;
  for (auto &cand : candidates)
    if (auto it = instanceOf.find(cand.op); it != instanceOf.end()) {
      int64_t &next = nextInstance[cand.oprName];
      next = std::max(next, it->second + 1);
    }
  for (auto &cand : candidates)
    if (alone.contains(cand.op))
      instanceOf[cand.op] = nextInstance[cand.oprName]++;

  auto *ctx = funcOp->getContext();
  auto i64 = IntegerType::get(ctx, 64);
  for (auto [op, inst] : instanceOf)
    op->setAttr("loopschedule.binding", IntegerAttr::get(i64, inst));
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
  // The scheduler's chain slack is for this pass alone.
  moduleOp.walk(
      [](Operation *op) { op->removeAttr("loopschedule.chain_slack"); });
}

std::unique_ptr<Pass>
circt::loopschedule::createLoopScheduleBindingPass() {
  return std::make_unique<LoopScheduleBindingPass>();
}

std::unique_ptr<Pass> circt::loopschedule::createLoopScheduleBindingPass(
    const LoopScheduleBindingOptions &options) {
  return std::make_unique<LoopScheduleBindingPass>(options);
}
