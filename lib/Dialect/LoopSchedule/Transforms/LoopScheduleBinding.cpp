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
//    hw.instance (a physical unit, e.g. a pipelined multiplier). One that
//    declares a limit had the scheduler hold each region to it, and its
//    delays price the mux sharing puts on its operands (resource-constrained
//    sharing, AMC's operator allocation): it binds in share mode too, never
//    to fewer instances than the schedule runs at once (two regions that
//    each use the limit and overlap get more), with no slack check. These bind in "share mode": the pool is sized to
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

#include "mlir/Dialect/Arith/IR/Arith.h"
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

/// The cycle of `op` in an iteration of `pipeline`: its stage's offset,
/// through `if`s; none for an op in any other nesting.
std::optional<uint64_t> cycleInPipeline(Operation *op,
                                        LoopSchedulePipelineOp pipeline) {
  for (Operation *p = op; p; p = p->getParentOp()) {
    if (auto at = dyn_cast<LoopScheduleAtOp>(p)) {
      if (at->getParentOp() != pipeline.getOperation())
        return std::nullopt;
      return at.getOffset();
    }
    if (p != op && !isa<LoopScheduleIfOp>(p))
      return std::nullopt;
  }
  return std::nullopt;
}

/// Does the next iteration of a pipeline read a result of `op`, a
/// multi-cycle operator of `latency` cycles, after the cycle it is ready?
/// A result the pipeline carries (an iter_arg_update's value, through
/// bitcasts and the yields of `at`s and `if`s) is read there from the
/// instance's output, which holds it only while the op's operands do (a
/// stage register's); a shared instance takes another user's operands in
/// between. In the ready cycle the output is the op's, whatever the other
/// users do: an accumulator at II = latency (`acc = acc + x`, read by the
/// next iteration's add as it issues) shares. A read after it (deriche's
/// filter at II 2 read `y1` two cycles late), the iter-arg's value after
/// the loop (a terminator result), or one passed along to another iter-arg
/// (`ym2 = ym1`) keeps the op on an instance of its own.
bool readsCarriedPastReady(Operation *op, unsigned latency) {
  auto pipeline = op->getParentOfType<LoopSchedulePipelineOp>();
  if (!pipeline)
    return false;
  Operation *terminator = pipeline.getStagesBlock().getTerminator();
  std::optional<uint64_t> issue = cycleInPipeline(op, pipeline);
  SmallVector<Value> worklist(op->result_begin(), op->result_end());
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (auto update = dyn_cast<LoopScheduleIterArgUpdateOp>(user)) {
        if (!issue)
          return true;
        uint64_t ready = *issue + latency;
        for (OpOperand &read : update.getIterArg().getUses()) {
          Operation *reader = read.getOwner();
          if (auto other = dyn_cast<LoopScheduleIterArgUpdateOp>(reader)) {
            if (&read == &other.getIterArgMutable())
              continue;
            return true;
          }
          if (reader == terminator)
            return true;
          std::optional<uint64_t> cycle = cycleInPipeline(reader, pipeline);
          if (!cycle || *cycle + pipeline.getII() != ready)
            return true;
        }
        continue;
      }
      if (isa<arith::BitcastOp>(user))
        worklist.append(user->result_begin(), user->result_end());
      else if (isa<LoopScheduleYieldOp>(user) &&
               isa<LoopScheduleAtOp, LoopScheduleIfOp>(user->getParentOp()))
        worklist.push_back(
            user->getParentOp()->getResult(use.getOperandNumber()));
    }
  }
  return false;
}
/// Can `pipeline` stall: does it wait on a dynamic-latency access (a
/// launched load, an expect, a dynamic port)?
bool canStall(LoopSchedulePipelineOp pipeline) {
  return pipeline
      ->walk([](Operation *op) {
        bool dynamic = isa<LoopScheduleLaunchOp, LoopScheduleExpectOp>(op);
        if (auto load = dyn_cast<LoadInterface>(op))
          dynamic |= load.isDynamic();
        if (auto store = dyn_cast<StoreInterface>(op))
          dynamic |= store.isDynamic();
        return dynamic ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

/// A binding problem that sees its users' stalls. An instance's clock
/// enable is the AND of its users' not-stalls, so a user that stalls
/// freezes the other users' operations in it, and its issues fall out of
/// step with theirs. A frame's ops and those of a pipeline it launched
/// share a concurrency group (the pipeline lowers inline): the frame's ops
/// issue by the launch, and the frame counts plain cycles to their results.
/// Against a pipeline that can stall, a frame op holds its instance from
/// its issue to its ready cycle, its operation in the instance (`flight`
/// cycles, its latency); two pipelines launched together share none.
class StallAwareBindingProblem : public binding::BindingProblem {
public:
  using BindingProblem::BindingProblem;

  /// \p op uses an operator instance, its operation in it \p cycles.
  void setFlight(Operation *op, unsigned cycles) { flight[op] = cycles; }

  bool conflicts(Operation *a, Operation *b) override {
    if (BindingProblem::conflicts(a, b))
      return true;
    if (a == b || !areConcurrent(a, b) || !flight.count(a) ||
        !flight.count(b))
      return false; // not both operator instance users (ports have limits)
    auto pa = a->getParentOfType<LoopSchedulePipelineOp>();
    auto pb = b->getParentOfType<LoopSchedulePipelineOp>();
    if (pa == pb)
      return false; // one stall domain
    if (pa && pb)
      return stalls(pa) || stalls(pb);
    Operation *frameOp = pa ? b : a, *pipeOp = pa ? a : b;
    if (!stalls(pa ? pa : pb))
      return false;
    // The frame op's operation is in the instance in [s, s + flight - 1];
    // the pipeline issues at t + j * II, j < count.
    unsigned f = flight.lookup(frameOp);
    if (f <= 1)
      return false; // its issue cycle alone: the check above
    int64_t s = getStartTime(frameOp).value_or(0);
    int64_t t = getStartTime(pipeOp).value_or(0);
    int64_t ii = getPeriod(pipeOp).value_or(0);
    int64_t count = getCount(pipeOp).value_or(1);
    int64_t last = s + f - 1;
    if (t >= s)
      return t <= last;
    if (ii == 0)
      return false;
    int64_t j = (s - t + ii - 1) / ii; // the first issue at or after s
    return j < count && t + j * ii <= last;
  }

private:
  bool stalls(LoopSchedulePipelineOp pipeline) {
    auto [it, inserted] = canStallCache.try_emplace(pipeline, false);
    if (inserted)
      it->second = canStall(pipeline);
    return it->second;
  }
  DenseMap<Operation *, unsigned> flight;
  DenseMap<Operation *, bool> canStallCache;
};
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
    // The operator's delays already include its sharing mux.
    bool muxPriced = false;
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
    } else if (isExternInstanceOperator(opLibAnalysis, oprAttr)) {
      // A physical unit with a declared limit: the schedule keeps each
      // region within it and its delays price the mux. Share mode.
      if (!shareOperators)
        continue;
      candidates.push_back({op, oprName, std::nullopt, /*muxPriced=*/true});
      ++oprOpCount[oprName];
      continue;
    }
    candidates.push_back({op, oprName, limit});
    ++oprOpCount[oprName];
  }

  // Bindings the scheduler made: kept, each op off its instance where it
  // issues with another user or its carried result needs the instance alone.
  // The rest bind below, past them.
  llvm::StringMap<int64_t> keptPast;
  if (keepBindings) {
    SmallVector<Candidate> rest;
    std::map<std::pair<std::string, int64_t>, SmallVector<Operation *>> groups;
    StallAwareBindingProblem prob(funcOp);
    for (auto &cand : candidates) {
      auto id = cand.op->getAttrOfType<IntegerAttr>("loopschedule.binding");
      if (!id || cand.declaredLimit) {
        rest.push_back(cand);
        continue;
      }
      Operation *op = cand.op;
      prob.insertOperation(op);
      prob.setStartTime(op, *stAnalysis.getStartTime(op));
      prob.setLatency(op, 1);
      prob.setFlight(op, opLibAnalysis.getOperatorLatency(cand.oprName));
      if (auto period = stAnalysis.getPeriod(op); period && *period > 0)
        prob.setPeriod(op, *period);
      if (auto count = stAnalysis.getCount(op); count && *count > 1)
        prob.setCount(op, *count);
      if (auto *group = stAnalysis.getConcurrencyGroup(op))
        prob.setConcurrencyGroup(op, group);
      groups[{cand.oprName.str(), id.getInt()}].push_back(op);
      int64_t &past = keptPast[cand.oprName];
      past = std::max(past, id.getInt() + 1);
    }
    auto *ctx = funcOp->getContext();
    auto i64 = IntegerType::get(ctx, 64);
    for (auto &[key, ops] : groups) {
      SmallVector<Operation *> stay;
      for (Operation *op : ops) {
        unsigned latency = opLibAnalysis.getOperatorLatency(key.first);
        bool conflict =
            (ops.size() > 1 && latency && readsCarriedPastReady(op, latency)) ||
            llvm::any_of(stay, [&](Operation *o) { return prob.conflicts(o, op); });
        if (!conflict) {
          stay.push_back(op);
          continue;
        }
        op->emitRemark("moves off the instance the scheduler bound it to (")
            << key.second << "): it would issue with another user, or its "
            << "carried result needs the instance alone";
        op->setAttr("loopschedule.binding",
                    IntegerAttr::get(i64, keptPast[key.first]++));
      }
    }
    candidates = std::move(rest);
  }

  if (candidates.empty())
    return success();

  // The delay of the mux an instance shared by `users` ops puts on their
  // operands: shareMuxDelays[users - 2], past the last given extrapolated by
  // its last step.
  auto muxDelay = [&](unsigned users) -> double {
    if (users < 2)
      return 0.0;
    size_t i = users - 2, given = shareMuxDelays.size();
    if (i < given)
      return shareMuxDelays[i];
    double step = given > 1 ? shareMuxDelays[given - 1] -
                                  shareMuxDelays[given - 2]
                            : 0.0;
    return shareMuxDelays[given - 1] + step * (i - (given - 1));
  };

  // Share-mode ops whose chain slack does not take their instance's mux,
  // and multi-cycle ones whose result a pipeline's next iteration reads
  // after its ready cycle (readsCarriedPastReady): each keeps an instance
  // of its own.
  DenseSet<Operation *> alone;
  for (auto &cand : candidates)
    if (!cand.declaredLimit)
      if (unsigned latency = opLibAnalysis.getOperatorLatency(cand.oprName))
        if (readsCarriedPastReady(cand.op, latency))
          alone.insert(cand.op);
  DenseMap<Operation *, int64_t> instanceOf;
  while (true) {
    // Phase 2 — populate the problem.
    StallAwareBindingProblem prob(funcOp);
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
      if (!isMemoryAccess(op))
        prob.setFlight(op, opLibAnalysis.getOperatorLatency(cand.oprName));
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
      if (cand.declaredLimit || cand.muxPriced)
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
  for (auto [op, inst] : instanceOf) {
    auto oprAttr = op->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    int64_t past =
        oprAttr ? keptPast.lookup(oprAttr.getLeafReference().getValue()) : 0;
    op->setAttr("loopschedule.binding", IntegerAttr::get(i64, inst + past));
  }
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
