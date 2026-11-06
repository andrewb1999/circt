//===- CPSATChaining.cpp - Allocation, scheduling and binding in CP-SAT ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See CPSATChaining.h for the model.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/CPSATChaining.h"
#include "circt/Scheduling/Utilities.h"

#include "mlir/IR/Operation.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <vector>

#define DEBUG_TYPE "cpsat-chaining"

using namespace circt;
using namespace circt::scheduling;
using namespace operations_research;
using namespace operations_research::sat;

using llvm::dbgs;

float scheduling::cpsatMuxDelay(ArrayRef<float> table, unsigned users) {
  if (users < 2 || table.empty())
    return 0.0f;
  size_t i = users - 2, given = table.size();
  if (i < given)
    return table[i];
  float step = given > 1 ? table[given - 1] - table[given - 2] : 0.0f;
  return table[given - 1] + step * (i - (given - 1));
}

namespace {

/// Nanoseconds to the model's picoseconds.
int64_t ps(float ns) { return std::llround(double(ns) * 1000.0); }

/// One operation's candidate operator, as the model sees it.
struct Choice {
  Problem::OperatorType opr;
  int64_t latency;
  int64_t inc, out; // ps, at most the cycle
  bool shareable;
  double area;
  BoolVar chosen; // valid when the operation has several choices
};

/// One instance an operation may take: one of the pool's, or a slot this
/// region may open.
struct Slot {
  Problem::OperatorType opr;
  std::optional<unsigned> poolIndex; // none for a new slot
  unsigned existingUsers = 0;
  int64_t rawSlack = 0; // ps; the pool's
  bool nor = false;     // the slower table
  BoolVar used;         // a new slot: opened
  IntVar users;
  // Users of regions not scheduled yet the instance keeps mux capacity for.
  std::optional<IntVar> reserve;
  IntVar mux; // ps
  // The operations that may take it, with their binding literals.
  SmallVector<std::pair<Operation *, BoolVar>, 8> takers;
  // users >= 2.
  BoolVar shared;
};

struct OpVars {
  IntVar c, z, mux, ready;
  std::optional<IntVar> mod; // c mod II
  SmallVector<Choice, 4> choices;
};

class ChainingModel {
public:
  ChainingModel(Problem &prob, ChainingProblem &chain, SharedOperatorsProblem &shr,
          CyclicProblem *cyclic, Operation *lastOp,
          const CPSATChainingConfig &config, const CPSATChainingInput &input)
      : prob(prob), chain(chain), shr(shr), cyclic(cyclic), lastOp(lastOp),
        config(config), input(input) {}

  LogicalResult run(CPSATChainingResult &result);

private:
  unsigned distance(Problem::Dependence dep) {
    return cyclic ? cyclic->getDistance(dep).value_or(0) : 0;
  }
  /// The operation's latency by its choice; at least 1 with `registered`
  /// (a value carried to a later iteration goes through a register: a
  /// combinational one is visible the cycle after it is computed).
  LinearExpr latencyOf(OpVars &v, bool registered = false) {
    auto of = [&](const Choice &ch) {
      return registered ? std::max<int64_t>(ch.latency, 1) : ch.latency;
    };
    if (v.choices.size() == 1)
      return LinearExpr(of(v.choices.front()));
    LinearExpr e;
    for (auto &ch : v.choices)
      e += of(ch) * LinearExpr(ch.chosen);
    return e;
  }
  /// The operation keeps an instance to itself on this choice.
  bool isAlone(Operation *op, const Choice &ch) {
    return input.alone.contains(op) ||
           (ch.latency > 0 && input.aloneMultiCycle.contains(op));
  }
  void build();
  LinearExpr areaExpr();
  bool solve(const LinearExpr &objective, CpSolverResponse &response);
  void hintFrom(const CpSolverResponse &response);
  void extract(const CpSolverResponse &response, CPSATChainingResult &result);

  Problem &prob;
  ChainingProblem &chain;
  SharedOperatorsProblem &shr;
  CyclicProblem *cyclic;
  Operation *lastOp;
  const CPSATChainingConfig &config;
  const CPSATChainingInput &input;

  CpModelBuilder cp;
  int64_t K = 0, H = 0;
  unsigned ii = 0;
  llvm::MapVector<Operation *, OpVars> vars;
  std::vector<Slot> slots;
  // Area terms, in LUT equivalents scaled to integers.
  LinearExpr area;
  // Every variable, for hints between phases.
  std::vector<IntVar> allInts;
  std::vector<BoolVar> allBools;
};

void ChainingModel::build() {
  K = ps(config.cycleTime);
  ii = cyclic ? cyclic->getInitiationInterval().value_or(0) : 0;

  // The horizon: every operation in a cycle of its own, past its longest
  // latency; or the hint's length, if longer.
  H = 1;
  int64_t maxLatAll = 0;
  for (Operation *op : prob.getOperations()) {
    int64_t maxLat = *prob.getLatency(*prob.getLinkedOperatorType(op));
    if (auto it = input.candidates.find(op); it != input.candidates.end())
      for (auto opr : it->second)
        maxLat = std::max<int64_t>(maxLat, *prob.getLatency(opr));
    H += maxLat + 1;
    maxLatAll = std::max(maxLatAll, maxLat);
  }
  for (auto [op, start] : input.hint)
    H = std::max<int64_t>(H, start + 1);

  auto newInt = [&](int64_t lo, int64_t hi) {
    IntVar v = cp.NewIntVar(Domain(lo, hi));
    allInts.push_back(v);
    return v;
  };
  auto newBool = [&]() {
    BoolVar b = cp.NewBoolVar();
    allBools.push_back(b);
    return b;
  };

  // Operations and their choices.
  int64_t maxMux = 0;
  for (float d : config.muxDelays)
    maxMux = std::max(maxMux, ps(d));
  for (float d : config.muxNorDelays)
    maxMux = std::max(maxMux, ps(d));
  maxMux = std::min(maxMux, K);
  for (Operation *op : prob.getOperations()) {
    OpVars v;
    v.c = newInt(0, H);
    v.z = newInt(0, K);
    v.mux = newInt(0, maxMux);
    v.ready = newInt(0, (H + maxLatAll + 1) * K);
    if (ii > 1) {
      v.mod = newInt(0, ii - 1);
      cp.AddModuloEquality(*v.mod, v.c, ii);
    }
    SmallVector<Problem::OperatorType, 4> oprs;
    if (auto it = input.candidates.find(op); it != input.candidates.end())
      oprs.assign(it->second.begin(), it->second.end());
    else
      oprs.push_back(*prob.getLinkedOperatorType(op));
    for (auto opr : oprs) {
      Choice ch;
      ch.opr = opr;
      ch.latency = *prob.getLatency(opr);
      ch.inc = std::min(ps(*chain.getIncomingDelay(opr)), K);
      ch.out = std::min(ps(*chain.getOutgoingDelay(opr)), K);
      auto info = input.operatorInfo.lookup(opr);
      ch.shareable = info.shareable;
      ch.area = info.area;
      if (oprs.size() > 1)
        ch.chosen = newBool();
      v.choices.push_back(ch);
    }
    if (oprs.size() > 1) {
      std::vector<BoolVar> lits;
      for (auto &ch : v.choices)
        lits.push_back(ch.chosen);
      cp.AddExactlyOne(lits);
    }
    // Fit and readiness, per choice.
    LinearExpr start = K * LinearExpr(v.c) + v.z;
    for (auto &ch : v.choices) {
      std::vector<BoolVar> when;
      if (oprs.size() > 1)
        when.push_back(ch.chosen);
      cp.AddLessOrEqual(LinearExpr(v.z) + v.mux + ch.inc, K)
          .OnlyEnforceIf(when);
      if (ch.latency == 0)
        cp.AddEquality(v.ready, start + v.mux + ch.out).OnlyEnforceIf(when);
      else
        cp.AddEquality(v.ready, K * LinearExpr(v.c) + K * ch.latency + ch.out)
            .OnlyEnforceIf(when);
    }
    vars.insert({op, std::move(v)});
  }

  // Dependences.
  for (Operation *op : prob.getOperations()) {
    OpVars &dst = vars[op];
    for (auto dep : prob.getDependences(op)) {
      Operation *srcOp = dep.getSource();
      if (!vars.count(srcOp))
        continue;
      OpVars &src = vars[srcOp];
      int64_t d = distance(dep) * ii;
      // In cycles: the source's latency, or the edge's own (an ordering
      // edge that needs only the source to have started). To a later
      // iteration, at least a cycle: through the iter arg's register.
      LinearExpr lat = latencyOf(src, /*registered=*/d > 0);
      if (dep.isAuxiliary()) {
        unsigned depLat = *prob.getDependenceLatency(dep);
        unsigned linkedLat =
            *prob.getLatency(*prob.getLinkedOperatorType(srcOp));
        if (depLat != linkedLat)
          lat = LinearExpr(depLat);
      }
      cp.AddGreaterOrEqual(LinearExpr(dst.c) + d, LinearExpr(src.c) + lat);
      if (dep.isAuxiliary())
        continue;
      // In time: the operands arrive by the start. A multi-cycle result
      // carried to a later iteration is read off the core in its ready
      // cycle, a combinational one from the register (at the cycle's start,
      // which the cycles above already order).
      if (d == 0) {
        cp.AddGreaterOrEqual(K * LinearExpr(dst.c) + dst.z,
                             LinearExpr(src.ready));
        continue;
      }
      for (auto &ch : src.choices) {
        if (ch.latency == 0)
          continue;
        std::vector<BoolVar> when;
        if (src.choices.size() > 1)
          when.push_back(ch.chosen);
        cp.AddGreaterOrEqual(K * LinearExpr(dst.c) + dst.z + d * K,
                             LinearExpr(src.ready))
            .OnlyEnforceIf(when);
      }
    }
  }

  // Limited resources (memory ports and the like): per cycle, or per
  // congruence class in a pipeline.
  llvm::MapVector<Problem::ResourceType, SmallVector<Operation *>> byResource;
  for (Operation *op : prob.getOperations())
    if (auto rsrcs = prob.getLinkedResourceTypes(op))
      for (auto rsrc : *rsrcs)
        if (shr.getLimit(rsrc))
          byResource[rsrc].push_back(op);
  for (auto &[rsrc, ops] : byResource) {
    int64_t limit = *shr.getLimit(rsrc);
    if (ii == 1) {
      if ((int64_t)ops.size() > limit)
        cp.AddLessOrEqual(LinearExpr((int64_t)ops.size()), limit);
      continue;
    }
    CumulativeConstraint cumul = cp.AddCumulative(limit);
    for (Operation *op : ops) {
      OpVars &v = vars[op];
      LinearExpr at = ii > 1 ? LinearExpr(*v.mod) : LinearExpr(v.c);
      cumul.AddDemand(cp.NewFixedSizeIntervalVar(at, 1), 1);
    }
  }

  // Binding. The slots of each shareable operator: the pool's instances,
  // then one new one per operation that may take the operator.
  llvm::MapVector<Problem::OperatorType, SmallVector<Operation *>> takersOf;
  for (auto &[op, v] : vars)
    for (auto &ch : v.choices)
      if (ch.shareable && !isAlone(op, ch))
        takersOf[ch.opr].push_back(op);
  for (auto &[opr, ops] : takersOf) {
    unsigned first = slots.size();
    for (auto [i, inst] : llvm::enumerate(input.pool)) {
      if (!(inst.opr == opr) || inst.closed)
        continue;
      Slot s;
      s.opr = opr;
      s.poolIndex = i;
      s.existingUsers = inst.users;
      s.rawSlack = ps(inst.rawSlack);
      s.nor = inst.pipelineUser || config.pipelined;
      slots.push_back(s);
    }
    unsigned firstNew = slots.size();
    for (size_t k = 0; k < ops.size(); ++k) {
      Slot s;
      s.opr = opr;
      s.nor = config.pipelined;
      s.used = newBool();
      slots.push_back(s);
    }
    // The binding literals: operation `i` of `ops` may open only new slots
    // up to its own index (symmetry).
    for (auto [i, op] : llvm::enumerate(ops)) {
      OpVars &v = vars[op];
      BoolVar takes = cp.TrueVar();
      for (auto &ch : v.choices)
        if (ch.opr == opr && v.choices.size() > 1)
          takes = ch.chosen;
      std::vector<BoolVar> lits;
      for (unsigned s = first; s < slots.size(); ++s) {
        if (s >= firstNew && s - firstNew > i)
          break;
        BoolVar b = newBool();
        slots[s].takers.push_back({op, b});
        lits.push_back(b);
        cp.AddImplication(b, takes);
      }
      // Taking the operator is taking one of its slots.
      LinearExpr sum;
      for (BoolVar b : lits)
        sum += b;
      cp.AddEquality(sum, takes);
    }
    // New slots open in order, and are open when taken.
    for (unsigned s = firstNew; s < slots.size(); ++s) {
      std::vector<BoolVar> takers;
      for (auto &[op, b] : slots[s].takers) {
        cp.AddImplication(b, slots[s].used);
        takers.push_back(b);
      }
      takers.push_back(slots[s].used.Not());
      cp.AddBoolOr(takers);
      if (s > firstNew)
        cp.AddImplication(slots[s].used, slots[s - 1].used);
    }
  }

  // Each slot's users, conflicts, mux and its users' chains.
  DenseMap<Problem::OperatorType, LinearExpr> reservedOf;
  for (Slot &s : slots) {
    unsigned future = input.futureUsers.lookup(s.opr);
    unsigned maxUsers = s.existingUsers + s.takers.size() + future;
    s.users = newInt(0, maxUsers);
    LinearExpr users(s.existingUsers);
    for (auto &[op, b] : s.takers)
      users += b;
    // The mux is sized for its users and those it reserves for; an instance
    // reserves only while open.
    if (future) {
      s.reserve = newInt(0, future);
      if (!s.poolIndex)
        cp.AddLessOrEqual(*s.reserve, future * LinearExpr(s.used));
      users += *s.reserve;
      reservedOf[s.opr] += *s.reserve;
    }
    cp.AddEquality(s.users, users);
    // Two of this region's users issue in different cycles.
    if (s.takers.size() > 1) {
      if (ii == 1) {
        LinearExpr n;
        for (auto &[op, b] : s.takers)
          n += b;
        cp.AddLessOrEqual(n, 1);
      } else {
        std::vector<IntervalVar> issues;
        for (auto &[op, b] : s.takers) {
          OpVars &v = vars[op];
          LinearExpr at = ii > 1 ? LinearExpr(*v.mod) : LinearExpr(v.c);
          issues.push_back(cp.NewOptionalFixedSizeIntervalVar(at, 1, b));
        }
        cp.AddNoOverlap(issues);
      }
    }
    // The mux: a step per user count, as the table goes.
    ArrayRef<float> table = s.nor && !config.muxNorDelays.empty()
                                ? config.muxNorDelays
                                : config.muxDelays;
    LinearExpr mux;
    int64_t previous = 0;
    s.shared = newBool();
    for (unsigned t = 2; t <= maxUsers; ++t) {
      int64_t delay = std::min(ps(cpsatMuxDelay(table, t)), K);
      BoolVar atLeast = t == 2 ? s.shared : newBool();
      cp.AddGreaterOrEqual(s.users, t).OnlyEnforceIf(atLeast);
      cp.AddLessThan(s.users, t).OnlyEnforceIf(atLeast.Not());
      if (delay > previous)
        mux += (delay - previous) * LinearExpr(atLeast);
      previous = std::max(previous, delay);
    }
    if (maxUsers < 2)
      cp.AddEquality(s.shared, 0);
    s.mux = newInt(0, K);
    cp.AddEquality(s.mux, mux);
    // The pool's users take it within their slack (they already take the
    // mux of their own count).
    if (s.poolIndex) {
      int64_t current =
          std::min(ps(cpsatMuxDelay(table, s.existingUsers)), K);
      cp.AddLessOrEqual(s.mux, std::max(s.rawSlack, current));
    }
    for (auto &[op, b] : s.takers)
      cp.AddEquality(vars[op].mux, s.mux).OnlyEnforceIf(b);
    // A multi-cycle carried result on an instance with other users is read
    // in its ready cycle (a combinational one goes into the iter arg's
    // register in its own cycle).
    for (auto &[op, b] : s.takers) {
      auto it = input.carriedReaders.find(op);
      if (it == input.carriedReaders.end())
        continue;
      OpVars &v = vars[op];
      for (auto &ch : v.choices) {
        if (!(ch.opr == s.opr) || ch.latency == 0)
          continue;
        for (Operation *reader : it->second) {
          if (!vars.count(reader))
            continue;
          cp.AddEquality(LinearExpr(vars[reader].c) + ii,
                         LinearExpr(v.c) + ch.latency)
              .OnlyEnforceIf({b, s.shared});
        }
      }
    }
  }
  // An operator's future users are reserved for once.
  for (auto &[opr, reserved] : reservedOf)
    cp.AddLessOrEqual(reserved, input.futureUsers.lookup(opr));
  // An operation on no shared instance has no mux.
  for (auto &[op, v] : vars) {
    std::vector<BoolVar> onShared;
    for (Slot &s : slots)
      for (auto &[taker, b] : s.takers)
        if (taker == op)
          onShared.push_back(b);
    if (onShared.empty())
      cp.AddEquality(v.mux, 0);
  }

  // Area: each opened slot, each unshared choice, each user of a mux.
  for (auto &[op, v] : vars)
    for (auto &ch : v.choices) {
      bool inSlots = ch.shareable && !isAlone(op, ch);
      if (inSlots)
        continue; // paid per opened slot
      int64_t a = std::llround(ch.area);
      if (a == 0)
        continue;
      if (v.choices.size() > 1)
        area += a * LinearExpr(ch.chosen);
      else
        area += LinearExpr(a);
    }
  for (Slot &s : slots) {
    auto info = input.operatorInfo.lookup(s.opr);
    if (!s.poolIndex)
      area += std::llround(info.area) * LinearExpr(s.used);
    int64_t perUser = std::llround(info.muxAreaPerUser);
    if (perUser)
      for (auto &[op, b] : s.takers)
        area += perUser * LinearExpr(b);
    // A user reserved for saves a later region the instance it would open
    // otherwise, less its mux.
    auto saved = input.futureCredit.find(s.opr);
    int64_t credit =
        (saved != input.futureCredit.end() ? std::llround(saved->second) : 0) -
        perUser;
    if (s.reserve && credit > 0)
      area += -credit * LinearExpr(*s.reserve);
  }
}

bool ChainingModel::solve(const LinearExpr &objective, CpSolverResponse &response) {
  cp.ClearObjective();
  cp.Minimize(objective);
  Model model;
  SatParameters params;
  // One worker searches the same way every run (the default: the same input
  // schedules the same way); more run CP-SAT's parallel portfolio, whose
  // solution within the limit may vary. (Its interleaved deterministic mode
  // fails an internal check on these models in OR-Tools 9.15.) The limit is
  // in deterministic time, with a wall-clock limit as a backstop.
  params.set_num_workers(std::max(1, config.numWorkers));
  params.set_max_deterministic_time(config.timeLimitS);
  params.set_max_time_in_seconds(config.timeLimitS * 4 + 1);
  params.set_random_seed(0);
  model.Add(NewSatParameters(params));
  response = SolveCpModel(cp.Build(), &model);
  LLVM_DEBUG(dbgs() << "cpsat: status " << response.status() << " objective "
                    << response.objective_value() << " bound "
                    << response.best_objective_bound() << " in "
                    << response.wall_time() << " s\n");
  return response.status() == CpSolverStatus::OPTIMAL ||
         response.status() == CpSolverStatus::FEASIBLE;
}

void ChainingModel::hintFrom(const CpSolverResponse &response) {
  cp.ClearHints();
  for (IntVar v : allInts)
    cp.AddHint(v, SolutionIntegerValue(response, v));
  for (BoolVar b : allBools)
    cp.AddHint(b, SolutionBooleanValue(response, b));
}

void ChainingModel::extract(const CpSolverResponse &response,
                      CPSATChainingResult &result) {
  // Operator choices, start times and the times operands reach the core.
  DenseMap<Operation *, unsigned> chosenOf;
  for (auto &[op, v] : vars) {
    unsigned k = 0;
    if (v.choices.size() > 1)
      for (auto [i, ch] : llvm::enumerate(v.choices))
        if (SolutionBooleanValue(response, ch.chosen))
          k = i;
    chosenOf[op] = k;
    prob.setLinkedOperatorType(op, v.choices[k].opr);
    prob.setStartTime(op, SolutionIntegerValue(response, v.c));
  }
  // The arrival of each operation's operands as the solution places them
  // (`z` may sit later than they need): the latest ready of a source in the
  // same cycle, by the solution's choices.
  DenseMap<Operation *, int64_t> arrival, readyAt, muxOf;
  for (auto &[op, v] : vars)
    muxOf[op] = SolutionIntegerValue(response, v.mux);
  (void)handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    OpVars &v = vars[op];
    int64_t c = SolutionIntegerValue(response, v.c);
    int64_t a = 0;
    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary() || distance(dep) != 0)
        continue;
      Operation *src = dep.getSource();
      auto it = readyAt.find(src);
      if (it == readyAt.end())
        return failure();
      int64_t r = it->second - c * K;
      if (r > 0)
        a = std::max(a, r);
    }
    arrival[op] = a;
    const Choice &ch = v.choices[chosenOf[op]];
    readyAt[op] = ch.latency == 0 ? c * K + a + muxOf[op] + ch.out
                                  : (c + ch.latency) * K + ch.out;
    chain.setStartTimeInCycle(op, float(a + muxOf[op]) / 1000.0f);
    return success();
  });
  // Each operation's slack before its mux: the cycle less the latest end of
  // a chain through it, its own mux taken out (that is what a larger one
  // would replace).
  DenseMap<Operation *, SmallVector<Operation *, 2>> sameCycleUsers;
  for (auto &[op, v] : vars)
    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary() || distance(dep) != 0)
        continue;
      Operation *src = dep.getSource();
      if (*prob.getStartTime(src) == *prob.getStartTime(op))
        sameCycleUsers[src].push_back(op);
    }
  DenseMap<Operation *, int64_t> chainEnd;
  std::function<int64_t(Operation *)> endOf = [&](Operation *op) -> int64_t {
    if (auto it = chainEnd.find(op); it != chainEnd.end())
      return it->second;
    const Choice &ch = vars[op].choices[chosenOf[op]];
    int64_t end = arrival.lookup(op) + muxOf[op] +
                  (ch.latency == 0 ? ch.out : ch.inc);
    chainEnd[op] = end;
    if (ch.latency == 0)
      for (Operation *user : sameCycleUsers.lookup(op))
        end = std::max(end, endOf(user));
    return chainEnd[op] = end;
  };
  for (auto &[op, v] : vars) {
    result.rawSlack[op] = float(K - endOf(op) + muxOf[op]) / 1000.0f;
    result.muxDelay[op] = float(muxOf[op]) / 1000.0f;
  }
  // Instances: the pool's ids, then new ones in slot order past them.
  DenseMap<Problem::OperatorType, int64_t> nextId;
  for (auto &inst : input.pool) {
    int64_t &next = nextId[inst.opr];
    next = std::max(next, inst.id + 1);
  }
  for (Slot &s : slots) {
    std::optional<int64_t> id;
    if (s.poolIndex)
      id = input.pool[*s.poolIndex].id;
    for (auto &[op, b] : s.takers) {
      if (!SolutionBooleanValue(response, b))
        continue;
      if (!id)
        id = nextId[s.opr]++;
      result.instance[op] = *id;
    }
  }
  // An operation alone on a shareable operator takes an instance of its own.
  for (auto &[op, v] : vars) {
    const Choice &ch = v.choices[chosenOf[op]];
    if (ch.shareable && isAlone(op, ch)) {
      result.instance[op] = nextId[ch.opr]++;
      result.closes.insert(op);
    }
  }
  // A carried result read past its ready cycle closes its instance (alone,
  // by the model) to later regions too.
  for (auto &[op, readers] : input.carriedReaders) {
    if (!result.instance.count(op))
      continue;
    const Choice &ch = vars[op].choices[chosenOf[op]];
    if (ch.latency == 0)
      continue;
    int64_t ready = *prob.getStartTime(op) + ch.latency;
    for (Operation *reader : readers)
      if (vars.count(reader) && *prob.getStartTime(reader) + ii != ready)
        result.closes.insert(op);
  }
}

LogicalResult ChainingModel::run(CPSATChainingResult &result) {
  if (!prob.hasOperation(lastOp))
    return prob.getContainingOp()->emitError(
        "problem does not include last operation");
  LLVM_DEBUG({
    for (auto &inst : input.pool)
      dbgs() << "cpsat: pool " << inst.opr.getValue() << " #" << inst.id
             << " users " << inst.users << " slack " << inst.rawSlack
             << (inst.closed ? " closed" : "")
             << (inst.pipelineUser ? " pipeline" : "") << "\n";
    for (auto &[opr, n] : input.futureUsers)
      dbgs() << "cpsat: future " << opr.getValue() << " " << n << "\n";
  });
  build();
  LLVM_DEBUG(dbgs() << "cpsat: " << vars.size() << " operations, "
                    << slots.size() << " slots, horizon " << H << ", II "
                    << ii << "\n");

  // Start from the hint: its start times, each operation on its linked
  // operator.
  for (auto &[op, v] : vars) {
    if (auto it = input.hint.find(op); it != input.hint.end())
      cp.AddHint(v.c, it->second);
    if (v.choices.size() > 1) {
      auto linked = *prob.getLinkedOperatorType(op);
      for (auto &ch : v.choices)
        cp.AddHint(ch.chosen, ch.opr == linked);
    }
  }

  // Phase 1: the length.
  LinearExpr length(vars[lastOp].c);
  CpSolverResponse response;
  if (!solve(length, response))
    return failure();
  int64_t bestLength = SolutionIntegerValue(response, length);
  result.optimal = response.status() == CpSolverStatus::OPTIMAL;
  CpSolverResponse best = response;

  if (!config.feasibilityOnly) {
    // Phase 2: area, within the length's slack.
    if (config.lengthSlack)
      cp.AddLessOrEqual(length, bestLength + *config.lengthSlack);
    hintFrom(best);
    LinearExpr cost = area;
    if (int64_t perCycle = std::llround(config.cycleCost))
      cost += perCycle * length;
    if (solve(cost, response)) {
      best = response;
      result.optimal &= response.status() == CpSolverStatus::OPTIMAL;
      // Phase 3: the earliest starts at that area.
      cp.AddLessOrEqual(area, SolutionIntegerValue(response, area));
      LinearExpr starts;
      for (auto &[op, v] : vars)
        starts += v.c;
      hintFrom(best);
      if (solve(starts, response))
        best = response;
    }
  }

  extract(best, result);
  result.length = SolutionIntegerValue(best, length);
  result.area = double(SolutionIntegerValue(best, area));
  return success();
}

} // namespace

LogicalResult scheduling::scheduleCPSATChaining(
    ChainingSharedOperatorsProblem &prob, Operation *lastOp,
    const CPSATChainingConfig &config, const CPSATChainingInput &input,
    CPSATChainingResult &result) {
  ChainingModel model(prob, prob, prob, /*cyclic=*/nullptr, lastOp, config, input);
  return model.run(result);
}

LogicalResult scheduling::scheduleCPSATChaining(
    ChainingModuloProblem &prob, Operation *lastOp,
    const CPSATChainingConfig &config, const CPSATChainingInput &input,
    CPSATChainingResult &result) {
  if (!prob.getInitiationInterval())
    return prob.getContainingOp()->emitError(
        "cpsat chaining: a pipeline needs its II");
  ChainingModel model(prob, prob, prob, &prob, lastOp, config, input);
  return model.run(result);
}
