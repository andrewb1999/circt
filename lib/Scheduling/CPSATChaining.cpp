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

#include "google/protobuf/text_format.h"
#include "ortools/sat/cp_model.h"
#include "ortools/sat/sat_parameters.pb.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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


class FunctionModel {
public:
  FunctionModel(const CPSATFunctionModel &m, const CPSATChainingConfig &config)
      : m(m), config(config) {}
  LogicalResult run(CPSATFunctionSolution &solution, std::string &why);

private:
  using FM = CPSATFunctionModel;
  struct OpV {
    IntVar c, z, mux, ready;
    std::optional<IntVar> mod;
    SmallVector<BoolVar, 4> chosen; // per candidate, when several
    bool onSlot = false;
  };
  struct Taker {
    unsigned region, op;
    BoolVar b;
  };
  struct SlotV {
    unsigned opr;
    BoolVar used, nor, shared;
    IntVar users, md;
    SmallVector<Taker, 8> takers;
  };

  const FM::Operator &opr(unsigned k) { return m.oprs[k]; }
  int64_t lat(unsigned k) { return m.oprs[k].latency; }
  int64_t inc(unsigned k) { return std::min(ps(m.oprs[k].inc), K); }
  int64_t out(unsigned k) { return std::min(ps(m.oprs[k].out), K); }
  std::vector<BoolVar> when(unsigned r, unsigned i, unsigned j) {
    if (m.regions[r].ops[i].oprs.size() == 1)
      return {};
    return {vars[r][i].chosen[j]};
  }
  LinearExpr latencyOf(unsigned r, unsigned i, bool registered) {
    auto &op = m.regions[r].ops[i];
    auto of = [&](unsigned k) {
      return registered ? std::max<int64_t>(lat(k), 1) : lat(k);
    };
    if (op.oprs.size() == 1)
      return LinearExpr(of(op.oprs[0]));
    LinearExpr e;
    for (auto [j, k] : llvm::enumerate(op.oprs))
      e += of(k) * LinearExpr(vars[r][i].chosen[j]);
    return e;
  }
  bool isAlone(const FM::Op &op, unsigned k) {
    return op.alone || (lat(k) > 0 && op.aloneMultiCycle);
  }
  /// A pipeline's II: a variable where the solver chooses it.
  LinearExpr iiOf(unsigned r) {
    if (iiVar[r])
      return LinearExpr(*iiVar[r]);
    return LinearExpr(int64_t(m.regions[r].ii));
  }
  bool fixedII1(unsigned r) {
    return !iiVar[r] && m.regions[r].ii == 1;
  }
  /// The mux tables at an operand width.
  ArrayRef<float> delays(bool nor, unsigned width) {
    if (width > 32) {
      auto t = nor ? config.muxNorDelays64 : config.muxDelays64;
      if (!t.empty())
        return t;
    }
    return nor && !config.muxNorDelays.empty() ? config.muxNorDelays
                                               : config.muxDelays;
  }
  IntVar newInt(int64_t lo, int64_t hi) {
    IntVar v = cp.NewIntVar(Domain(lo, hi));
    ints.push_back(v);
    return v;
  }
  BoolVar newBool() {
    BoolVar b = cp.NewBoolVar();
    bools.push_back(b);
    return b;
  }
  void build();
  /// The least and most \p e can be, by its variables' domains.
  std::pair<int64_t, int64_t> bounds(const LinearExpr &e);
  /// A variable at least `logScale * ln(x)` for x = \p e in [lo, hi] (and
  /// minimized, equal to it at the breakpoints): piecewise linear over
  /// geometric breakpoints, one segment chosen.
  LinearExpr logOf(const LinearExpr &e, StringRef what);
  bool solve(StringRef phase, const LinearExpr &objective,
             CpSolverResponse &response, std::string &why);
  void hint();
  /// The solver's settings for a solve of \p limit (deterministic) seconds.
  bool parameters(double limit, SatParameters &params, std::string &why);
  /// The hint is the regions' schedules alone: their starts, operator
  /// choices and IIs, a small part of the model's variables. Solve the
  /// model with those fixed (the offsets in a cycle, the instances, the
  /// registers left, which is quick), and hint its solution: complete, a
  /// schedule the search starts from. A large function left a partial
  /// hint found no schedule in its limit (sequential f64 deriche: 25,000
  /// variables, a thousand hinted).
  void completeHint(const LinearExpr &objective);
  /// An instance's users in regions that run at once (a child loop, and a
  /// sibling launched with it or its parent's op in the instance then)
  /// kept apart.
  void concurrency(SlotV &s);
  void extract(const CpSolverResponse &response,
               CPSATFunctionSolution &solution);
  /// Add \p x times an operator's own resources.
  void addResources(const CPSATOperatorInfo &info, const LinearExpr &x);
  /// The area objective over the resources (`config.bottleneck`).
  void buildArea();

  const FM &m;
  const CPSATChainingConfig &config;
  CpModelBuilder cp;
  int64_t K = 0;
  std::vector<std::vector<OpV>> vars;
  std::vector<std::optional<IntVar>> iiVar;
  std::vector<IntVar> lastVar; // per region, its last cycle
  SmallVector<int64_t> horizon;
  std::vector<SlotV> slots;
  // The function's resources, the area objective over them, and its cycles.
  LinearExpr lutUse, dspUse, ffUse, area, cycles;
  std::optional<IntVar> peak; // the bottleneck's share of the device
  // Parts of them, for the debug output.
  LinearExpr muxLutsTotal, registerFFs, registerSRLs, enableFFs;
  struct Held {
    unsigned region, op;
    int64_t bits;
    IntVar held;
    LinearExpr regs;
  };
  std::vector<Held> heldValues;
  std::vector<IntVar> ints;
  std::vector<BoolVar> bools;
};

void FunctionModel::build() {
  K = ps(config.cycleTime);
  int64_t maxMux = 0;
  for (auto table : {config.muxDelays, config.muxNorDelays,
                     config.muxDelays64, config.muxNorDelays64})
    for (float d : table)
      maxMux = std::max(maxMux, ps(d));
  maxMux = std::min(maxMux, K);

  for (auto [r, region] : llvm::enumerate(m.regions)) {
    // The horizon: every op after the last, on its slowest candidate or
    // its longest dependence latency (a serial schedule, which any resource
    // limit admits).
    SmallVector<int64_t> depLat(region.ops.size(), 0);
    for (auto &dep : region.deps)
      if (dep.latency)
        depLat[dep.src] = std::max<int64_t>(depLat[dep.src], *dep.latency);
    int64_t H = 1, maxLat = 0;
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      int64_t l = depLat[i];
      for (unsigned k : op.oprs)
        l = std::max(l, lat(k));
      H += l + 1;
      maxLat = std::max(maxLat, l);
      if (op.hint)
        H = std::max<int64_t>(H, *op.hint + 1);
    }
    horizon.push_back(H);

    // The II, a variable from the resources' bound where the solver
    // chooses it, up to the serial schedule's length (which closes any
    // recurrence).
    std::optional<IntVar> ii;
    int64_t maxII = region.ii;
    if (region.ii > 0 && region.searchII) {
      int64_t lo = cpsatResourceMII(region);
      maxII = std::max<int64_t>({int64_t(region.ii), H, lo});
      ii = newInt(lo, maxII);
    }
    iiVar.push_back(ii);
    auto &rv = vars.emplace_back();
    for (auto &op : region.ops) {
      OpV v;
      v.c = newInt(0, H);
      v.z = newInt(0, K);
      v.mux = newInt(0, maxMux);
      v.ready = newInt(0, (H + maxLat + 1) * K);
      if (region.ii > 0 && !fixedII1(r)) {
        v.mod = newInt(0, maxII - 1);
        cp.AddModuloEquality(*v.mod, v.c, iiOf(r));
      }
      if (op.oprs.size() > 1) {
        std::vector<BoolVar> lits;
        for (size_t j = 0; j < op.oprs.size(); ++j) {
          v.chosen.push_back(newBool());
          lits.push_back(v.chosen.back());
        }
        cp.AddExactlyOne(lits);
        // An operator slower than the cycle, its delays clamped to it (so
        // it would seem to fit), only where none of the op's candidates
        // fits: allocation offers one where a pipelined loop's recurrence
        // needs it at the II it was asked for (there the II takes
        // precedence over the clock), but this scheduler keeps the clock
        // and chooses the II (deriche's filters took a 3.45 ns multiply-add
        // at II 2 beside a 2-cycle one that fits).
        auto slow = [&](unsigned k) { return m.oprs[k].info.overClock; };
        if (!llvm::all_of(op.oprs, slow))
          for (auto [j, k] : llvm::enumerate(op.oprs))
            if (slow(k))
              cp.AddEquality(v.chosen[j], 0);
      }
      rv.push_back(v);
    }
    // Fit and readiness, per choice.
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      OpV &v = rv[i];
      LinearExpr start = K * LinearExpr(v.c) + v.z;
      for (auto [j, k] : llvm::enumerate(op.oprs)) {
        auto w = when(r, i, j);
        cp.AddLessOrEqual(LinearExpr(v.z) + v.mux + inc(k), K).OnlyEnforceIf(w);
        if (lat(k) == 0)
          cp.AddEquality(v.ready, start + v.mux + out(k)).OnlyEnforceIf(w);
        else
          cp.AddEquality(v.ready, K * LinearExpr(v.c) + K * lat(k) + out(k))
              .OnlyEnforceIf(w);
      }
    }
    // Dependences (to a later iteration through the iter arg's register).
    for (auto &dep : region.deps) {
      OpV &src = rv[dep.src], &dst = rv[dep.dst];
      LinearExpr later = int64_t(dep.distance) * iiOf(r);
      LinearExpr l = latencyOf(r, dep.src, dep.distance > 0);
      if (dep.auxiliary && dep.latency)
        l = LinearExpr(*dep.latency);
      cp.AddGreaterOrEqual(LinearExpr(dst.c) + later, LinearExpr(src.c) + l);
      if (dep.auxiliary)
        continue;
      if (dep.distance == 0) {
        cp.AddGreaterOrEqual(K * LinearExpr(dst.c) + dst.z,
                             LinearExpr(src.ready));
        continue;
      }
      for (auto [j, k] : llvm::enumerate(region.ops[dep.src].oprs))
        if (lat(k) > 0)
          cp.AddGreaterOrEqual(K * LinearExpr(dst.c) + dst.z + K * later,
                               LinearExpr(src.ready))
              .OnlyEnforceIf(when(r, dep.src, j));
    }
    // The continuation, decided in time to gate the next launch.
    if (region.cond) {
      LinearExpr condAt(rv[*region.cond].c);
      if (region.condFirst) {
        cp.AddEquality(condAt, 0);
      } else {
        cp.AddLessOrEqual(condAt, iiOf(r));
        for (unsigned e : region.effects)
          cp.AddGreaterOrEqual(rv[e].c, condAt);
      }
    }
    // Limited resources, per cycle or congruence class.
    for (auto &rsrc : region.resources) {
      if (fixedII1(r)) {
        if (rsrc.ops.size() > rsrc.limit)
          cp.AddLessOrEqual(LinearExpr(int64_t(rsrc.ops.size())), rsrc.limit);
        continue;
      }
      CumulativeConstraint cumul = cp.AddCumulative(rsrc.limit);
      for (unsigned i : rsrc.ops) {
        LinearExpr at = region.ii > 0 ? LinearExpr(*rv[i].mod)
                                      : LinearExpr(rv[i].c);
        cumul.AddDemand(cp.NewFixedSizeIntervalVar(at, 1), 1);
      }
    }
    // Registers: a value held past the cycle it is ready, to its last
    // reader but the terminator (an iter arg's update and a loop result
    // have registers of their own, whatever the schedule). In a pipeline a
    // register keeps it until the next iteration overwrites it, II cycles
    // on, so a value held L cycles is a delay line of k = ceil(L / II) (one
    // per cycle at II 1), which synthesis builds, per bit, as k flops up to
    // 3 and otherwise a flop at each end and an SRL per 32 cycles between
    // (Vivado, measured). In a sequential region a frame holds it in one.
    SmallVector<SmallVector<unsigned, 2>> uses(region.ops.size());
    for (auto &dep : region.deps)
      if (!dep.auxiliary && dep.distance == 0 && dep.src != dep.dst &&
          dep.dst != region.anchor)
        uses[dep.src].push_back(dep.dst);
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      if (!op.width || uses[i].empty())
        continue;
      int64_t bits = op.width;
      IntVar held = newInt(0, H);
      for (unsigned u : uses[i])
        cp.AddGreaterOrEqual(held, LinearExpr(rv[u].c) -
                                       (LinearExpr(rv[i].c) +
                                        latencyOf(r, i, false)));
      if (region.ii > 0) {
        LinearExpr regs(held);
        if (iiVar[r]) {
          IntVar n = newInt(0, H), covered = newInt(0, H * maxII);
          cp.AddMultiplicationEquality(covered, {LinearExpr(n), iiOf(r)});
          cp.AddGreaterOrEqual(covered, held);
          regs = LinearExpr(n);
        } else if (region.ii > 1) {
          IntVar n = newInt(0, H);
          cp.AddGreaterOrEqual(int64_t(region.ii) * LinearExpr(n), held);
          regs = LinearExpr(n);
        }
        // The delay line's flops and SRLs per bit.
        BoolVar srl = newBool();
        cp.AddGreaterOrEqual(regs, 4).OnlyEnforceIf(srl);
        cp.AddLessOrEqual(regs, 3).OnlyEnforceIf(srl.Not());
        IntVar flops = newInt(0, 3), srls = newInt(0, H / 32 + 2);
        cp.AddGreaterOrEqual(LinearExpr(flops), regs)
            .OnlyEnforceIf(srl.Not());
        cp.AddEquality(flops, 2).OnlyEnforceIf(srl);
        cp.AddEquality(srls, 0).OnlyEnforceIf(srl.Not());
        cp.AddGreaterOrEqual(32 * LinearExpr(srls), regs - 2)
            .OnlyEnforceIf(srl);
        cp.AddGreaterOrEqual(srls, 1).OnlyEnforceIf(srl);
        ffUse += bits * LinearExpr(flops);
        lutUse += bits * LinearExpr(srls);
        registerFFs += bits * LinearExpr(flops);
        registerSRLs += bits * LinearExpr(srls);
        heldValues.push_back({unsigned(r), unsigned(i), bits, held, regs});
      } else {
        BoolVar any = newBool();
        cp.AddLessOrEqual(held, H * LinearExpr(any));
        ffUse += bits * LinearExpr(any);
        registerFFs += bits * LinearExpr(any);
        heldValues.push_back(
            {unsigned(r), unsigned(i), bits, held, LinearExpr(any)});
      }
    }
    // The region's last cycle: a pipeline's depth is its anchor's start
    // (every op precedes the anchor); a sequential region's anchor precedes
    // only its stores, so there it is the last op's start (a multi-cycle
    // op's last cycle), each a bound of its own.
    IntVar last = newInt(0, H + maxLat);
    lastVar.push_back(last);
    if (region.ii > 0) {
      cp.AddEquality(last, rv[region.anchor].c);
    } else {
      for (auto [i, op] : llvm::enumerate(region.ops)) {
        LinearExpr end(rv[i].c);
        for (auto [j, k] : llvm::enumerate(op.oprs))
          if (lat(k) > 1)
            cp.AddGreaterOrEqual(last, end + (lat(k) - 1))
                .OnlyEnforceIf(when(r, i, j));
        cp.AddGreaterOrEqual(last, end);
      }
    }
    // An enable flop per cycle: a pipeline's stage valid bits, a sequential
    // body's cycle states.
    ffUse += LinearExpr(last) + 1;
    enableFFs += LinearExpr(last) + 1;
    // The region's cycles in a run of the function.
    LinearExpr length = LinearExpr(last) + 1;
    if (region.ii > 0)
      cycles += region.runs * (length + (region.tripCount - 1) * iiOf(r));
    else
      cycles += region.runs * length;
  }

  // A child loop runs from its launch until its parent's next frame. A
  // dynamic access of the parent's in flight then, on a port the child uses
  // (an operator of its or of a loop inside it), completes with a pulse on
  // the port the child counts as its own, and the child waits for one more
  // than it gets: it is done by the launch (in its last budget cycle, which
  // the frame holds until the access completes, and the launch with it;
  // not in its own cycle, a function frame's child starting with the frame)
  // or comes after it, in a frame that awaits the child. Within the
  // parent's region (the launch knows its child's operators), so a region
  // scheduled alone keeps it too.
  for (auto [r, region] : llvm::enumerate(m.regions))
    for (auto [l, launcher] : llvm::enumerate(region.ops)) {
      if (launcher.childOprs.empty())
        continue;
      IntVar launch = vars[r][l].c;
      for (auto [a, op] : llvm::enumerate(region.ops)) {
        if (!op.dynamic || op.oprs.empty() ||
            !llvm::binary_search(launcher.childOprs, op.oprs[0]))
          continue;
        int64_t budget = std::max<int64_t>(lat(op.oprs[0]) - 1, 1);
        BoolVar after = newBool();
        cp.AddGreaterOrEqual(LinearExpr(launch),
                             LinearExpr(vars[r][a].c) + budget)
            .OnlyEnforceIf(after.Not());
        cp.AddGreaterOrEqual(vars[r][a].c, LinearExpr(launch) + 1)
            .OnlyEnforceIf(after);
      }
    }

  // Instances of each shareable operator, across all regions: as many as
  // twice the most a region could use at once, and two more (a mux may make
  // a region split its users), at most one per taker.
  for (auto [k, o] : llvm::enumerate(m.oprs)) {
    if (!o.info.shareable)
      continue;
    SmallVector<std::pair<unsigned, unsigned>> takers;
    SmallVector<unsigned> perRegion(m.regions.size(), 0);
    for (auto [r, region] : llvm::enumerate(m.regions))
      for (auto [i, op] : llvm::enumerate(region.ops))
        if (llvm::is_contained(op.oprs, k) && !isAlone(op, k)) {
          takers.push_back({r, i});
          ++perRegion[r];
        }
    if (takers.empty())
      continue;
    unsigned most = *llvm::max_element(perRegion);
    unsigned count = std::min<unsigned>(takers.size(), 2 * most + 2);
    unsigned first = slots.size();
    for (unsigned s = 0; s < count; ++s) {
      SlotV slot;
      slot.opr = k;
      slot.used = newBool();
      slots.push_back(slot);
      if (s > 0)
        cp.AddImplication(slots[first + s].used, slots[first + s - 1].used);
    }
    for (auto [t, ri] : llvm::enumerate(takers)) {
      auto [r, i] = ri;
      auto &op = m.regions[r].ops[i];
      unsigned j = llvm::find(op.oprs, k) - op.oprs.begin();
      BoolVar takes =
          op.oprs.size() > 1 ? vars[r][i].chosen[j] : cp.TrueVar();
      LinearExpr sum;
      for (unsigned s = 0; s < count && s <= t; ++s) {
        BoolVar b = newBool();
        slots[first + s].takers.push_back({unsigned(r), unsigned(i), b});
        cp.AddImplication(b, slots[first + s].used);
        sum += b;
      }
      cp.AddEquality(sum, takes);
      vars[r][i].onSlot = true;
    }
  }

  for (SlotV &s : slots) {
    const auto &info = opr(s.opr).info;
    unsigned maxUsers = s.takers.size();
    s.users = newInt(0, maxUsers);
    LinearExpr users;
    for (auto &t : s.takers)
      users += t.b;
    cp.AddEquality(s.users, users);
    std::vector<BoolVar> any;
    for (auto &t : s.takers)
      any.push_back(t.b);
    any.push_back(s.used.Not());
    cp.AddBoolOr(any);
    // A region's users of one instance issue apart.
    DenseMap<unsigned, SmallVector<Taker *, 4>> byRegion;
    for (auto &t : s.takers)
      byRegion[t.region].push_back(&t);
    bool anyPipelined = false;
    for (auto &[r, ts] : byRegion) {
      anyPipelined |= m.regions[r].ii > 0;
      if (ts.size() < 2)
        continue;
      if (fixedII1(r)) {
        LinearExpr n;
        for (Taker *t : ts)
          n += t->b;
        cp.AddLessOrEqual(n, 1);
        continue;
      }
      std::vector<IntervalVar> issues;
      for (Taker *t : ts) {
        OpV &v = vars[r][t->op];
        LinearExpr at = m.regions[r].ii > 0 ? LinearExpr(*v.mod)
                                            : LinearExpr(v.c);
        issues.push_back(cp.NewOptionalFixedSizeIntervalVar(at, 1, t->b));
      }
      cp.AddNoOverlap(issues);
    }
    concurrency(s);
    // The mux's delay a step per user count, from the table at the
    // operands' width (the slower one if a pipeline uses it).
    s.nor = newBool();
    for (auto &t : s.takers)
      if (m.regions[t.region].ii > 0)
        cp.AddImplication(t.b, s.nor);
    if (!anyPipelined)
      cp.AddEquality(s.nor, 0);
    s.shared = newBool();
    auto plainT = delays(false, info.width), norT = delays(true, info.width);
    LinearExpr plain, nor;
    int64_t plainPrev = 0, norPrev = 0;
    for (unsigned t = 2; t <= maxUsers; ++t) {
      BoolVar atLeast = t == 2 ? s.shared : newBool();
      cp.AddGreaterOrEqual(s.users, t).OnlyEnforceIf(atLeast);
      cp.AddLessThan(s.users, t).OnlyEnforceIf(atLeast.Not());
      int64_t p = std::min(ps(cpsatMuxDelay(plainT, t)), K);
      int64_t q = std::min(ps(cpsatMuxDelay(norT, t)), K);
      if (p > plainPrev)
        plain += (p - plainPrev) * LinearExpr(atLeast);
      if (q > norPrev)
        nor += (q - norPrev) * LinearExpr(atLeast);
      plainPrev = std::max(plainPrev, p);
      norPrev = std::max(norPrev, q);
    }
    if (maxUsers < 2)
      cp.AddEquality(s.shared, 0);
    s.md = newInt(0, K);
    cp.AddGreaterOrEqual(s.md, plain);
    if (anyPipelined)
      cp.AddGreaterOrEqual(s.md, nor).OnlyEnforceIf(s.nor);
    for (auto &t : s.takers) {
      cp.AddGreaterOrEqual(vars[t.region][t.op].mux, s.md).OnlyEnforceIf(t.b);
      auto &op = m.regions[t.region].ops[t.op];
      if (op.carriedReaders.empty() || lat(s.opr) == 0)
        continue;
      for (unsigned reader : op.carriedReaders)
        cp.AddEquality(LinearExpr(vars[t.region][reader].c) +
                           iiOf(t.region),
                       LinearExpr(vars[t.region][t.op].c) + lat(s.opr))
            .OnlyEnforceIf({t.b, s.shared});
    }
    // Resources: the instance's, and its mux's LUTs, per operand a price
    // for each input past the first, an input per value its users put on
    // it.
    addResources(info, LinearExpr(s.used));
    int64_t perInput =
        std::llround(config.muxInputLuts * std::max(info.width, 1u) / 32.0);
    if (perInput <= 0 || maxUsers < 2)
      continue;
    LinearExpr muxLuts;
    for (unsigned p = 0; p < info.operands; ++p) {
      LinearExpr inputs;
      llvm::MapVector<unsigned, std::vector<BoolVar>> byValue;
      for (auto &t : s.takers) {
        auto &op = m.regions[t.region].ops[t.op];
        if (p < op.inputs.size())
          byValue[op.inputs[p]].push_back(t.b);
        else
          inputs += t.b;
      }
      for (auto &[value, takers] : byValue) {
        if (takers.size() == 1) {
          inputs += takers.front();
          continue;
        }
        // The value is an input iff one of its users is on the instance.
        BoolVar fed = newBool();
        for (BoolVar b : takers)
          cp.AddImplication(b, fed);
        takers.push_back(fed.Not());
        cp.AddBoolOr(takers);
        inputs += fed;
      }
      muxLuts += perInput * (inputs - LinearExpr(s.used));
    }
    lutUse += muxLuts;
    muxLutsTotal += muxLuts;
  }
  // An op on no instance has no mux; an unshared choice pays its own area.
  for (auto [r, region] : llvm::enumerate(m.regions))
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      if (!vars[r][i].onSlot)
        cp.AddEquality(vars[r][i].mux, 0);
      for (auto [j, k] : llvm::enumerate(op.oprs)) {
        if (opr(k).info.shareable && !isAlone(op, k))
          continue;
        addResources(opr(k).info, op.oprs.size() > 1
                                      ? LinearExpr(vars[r][i].chosen[j])
                                      : LinearExpr(1));
      }
    }
  buildArea();
}

void FunctionModel::concurrency(SlotV &s) {
  unsigned n = m.regions.size();
  // The instance's users in each child loop, or in the loops inside it.
  std::vector<std::vector<BoolVar>> inTree(n);
  for (auto &t : s.takers)
    for (std::optional<unsigned> r = t.region; r; r = m.regions[*r].parent)
      if (m.regions[*r].parent)
        inTree[*r].push_back(t.b);
  std::vector<std::optional<BoolVar>> uses(n);
  auto usesOf = [&](unsigned r) -> BoolVar {
    if (uses[r])
      return *uses[r];
    if (inTree[r].size() == 1)
      return *(uses[r] = inTree[r].front());
    BoolVar u = newBool();
    std::vector<BoolVar> any;
    for (BoolVar b : inTree[r]) {
      cp.AddImplication(b, u);
      any.push_back(b);
    }
    any.push_back(u.Not());
    cp.AddBoolOr(any);
    return *(uses[r] = u);
  };
  int64_t latency = lat(s.opr);
  for (unsigned r = 0; r < n; ++r) {
    auto parent = m.regions[r].parent;
    if (!parent || inTree[r].empty())
      continue;
    unsigned launch = m.regions[r].launch;
    IntVar at = vars[*parent][launch].c;
    // The parent's users: out of the instance before the launch's cycle
    // (issued and ready; a combinational one issued), or after it, in a
    // frame that awaits it. Not in its cycle: a function's frame that
    // launches a child starts the child as the frame starts, which a
    // frame's shift (after a multi-cycle result of an earlier frame's) puts
    // before the cycle its other ops are scheduled in (deriche's frame
    // issued a multiply two cycles into a child's pipeline on it).
    int64_t busy = std::max<int64_t>(latency, 1);
    for (auto &t : s.takers) {
      if (t.region != *parent)
        continue;
      IntVar c = vars[t.region][t.op].c;
      BoolVar after = newBool();
      cp.AddLessOrEqual(LinearExpr(c) + busy, at)
          .OnlyEnforceIf({t.b, usesOf(r), after.Not()});
      cp.AddGreaterOrEqual(c, LinearExpr(at) + 1)
          .OnlyEnforceIf({t.b, usesOf(r), after});
    }
    // A sibling launched in the same cycle runs alongside; one launched
    // later is in a later frame, which awaits this one.
    for (unsigned q = r + 1; q < n; ++q) {
      if (m.regions[q].parent != parent || inTree[q].empty())
        continue;
      if (m.regions[q].launch == launch) {
        cp.AddBoolOr({usesOf(r).Not(), usesOf(q).Not()});
        continue;
      }
      cp.AddNotEqual(at, vars[*parent][m.regions[q].launch].c)
          .OnlyEnforceIf({usesOf(r), usesOf(q)});
    }
  }
}

void FunctionModel::addResources(const CPSATOperatorInfo &info,
                                 const LinearExpr &x) {
  if (int64_t n = std::llround(info.lut))
    lutUse += n * x;
  if (int64_t n = std::llround(info.dsp))
    dspUse += n * x;
  if (int64_t n = std::llround(info.ff))
    ffUse += n * x;
}

void FunctionModel::buildArea() {
  // Each resource as a share of the device's, in billionths: their sum
  // weighs each by its scarcity (on the xcv80 a DSP is 237 LUTs, an FF half
  // of one); the bottleneck is the largest, the sum its tie-break.
  auto share = [](const LinearExpr &e, double capacity) {
    return std::llround(1e9 / std::max(capacity, 1.0)) * e;
  };
  LinearExpr uLut = share(lutUse, config.lutCapacity);
  LinearExpr uDsp = share(dspUse, config.dspCapacity);
  LinearExpr uFf = share(ffUse, config.ffCapacity);
  LinearExpr sum = uLut + uDsp + uFf;
  if (!config.bottleneck) {
    area = sum;
    return;
  }
  int64_t hi = 0;
  for (const LinearExpr *u : {&uLut, &uDsp, &uFf})
    hi = std::max(hi, bounds(*u).second);
  peak = newInt(0, hi);
  for (const LinearExpr *u : {&uLut, &uDsp, &uFf})
    cp.AddGreaterOrEqual(*peak, *u);
  area = 1000 * LinearExpr(*peak) + sum;
}

std::pair<int64_t, int64_t> FunctionModel::bounds(const LinearExpr &e) {
  const CpModelProto &proto = cp.Proto();
  int64_t lo = e.constant(), hi = e.constant();
  for (auto [v, c] : llvm::zip(e.variables(), e.coefficients())) {
    const auto &dom = proto.variables(v).domain();
    int64_t vlo = dom[0], vhi = dom[dom.size() - 1];
    lo += c * (c >= 0 ? vlo : vhi);
    hi += c * (c >= 0 ? vhi : vlo);
  }
  return {lo, hi};
}

/// A natural log's units in the objective.
static constexpr double logScale = 1e5;

LinearExpr FunctionModel::logOf(const LinearExpr &e, StringRef what) {
  // x = e, shifted to at least 1 where e may be 0 (an area of nothing).
  auto [lo, hi] = bounds(e);
  int64_t shift = lo < 1 ? 1 - lo : 0;
  lo += shift;
  hi += shift;
  auto y = [](int64_t b) {
    return std::llround(logScale * std::log(double(b)));
  };
  if (lo == hi)
    return LinearExpr(y(lo));
  IntVar x = newInt(lo, hi);
  cp.AddEquality(x, e + shift);

  // Breakpoints a ratio q apart, q the largest whose chord falls at most
  // logError below the log (the chord over [a, q a] falls furthest, by
  // ln((q - 1) / ln q) - 1 + ln q / (q - 1), at a (q - 1) / ln q), and at
  // least one apart: between consecutive integers x takes no value, so
  // there the chord is exact.
  auto chordError = [](double q) {
    double l = std::log(q);
    return std::log((q - 1) / l) - 1 + l / (q - 1);
  };
  double qlo = 1.0001, qhi = 4.0;
  if (chordError(qhi) <= config.logError)
    qlo = qhi;
  for (int it = 0; it < 60 && qlo < qhi; ++it) {
    double mid = (qlo + qhi) / 2;
    (chordError(mid) <= config.logError ? qlo : qhi) = mid;
  }
  SmallVector<int64_t> b{lo};
  while (b.back() < hi) {
    int64_t next = std::max<int64_t>(
        b.back() + 1, int64_t(std::floor(double(b.back()) * qlo)));
    b.push_back(std::min(next, hi));
  }

  // L at least the chord of the segment x is in; the secant over the whole
  // range, below the log everywhere on it, also unconditionally (the
  // relaxation's bound).
  IntVar L = newInt(y(lo), y(hi));
  auto chord = [&](int64_t b0, int64_t b1) {
    // (b1 - b0) L >= (b1 - b0) y(b0) + (y(b1) - y(b0)) (x - b0)
    int64_t w = b1 - b0, dy = y(b1) - y(b0);
    return cp.AddGreaterOrEqual(w * LinearExpr(L) - dy * LinearExpr(x),
                                w * y(b0) - dy * b0);
  };
  chord(lo, hi);
  std::vector<BoolVar> segments;
  for (size_t k = 0; k + 1 < b.size(); ++k) {
    BoolVar in = newBool();
    segments.push_back(in);
    cp.AddGreaterOrEqual(x, b[k]).OnlyEnforceIf(in);
    cp.AddLessOrEqual(x, b[k + 1]).OnlyEnforceIf(in);
    chord(b[k], b[k + 1]).OnlyEnforceIf(in);
  }
  cp.AddExactlyOne(segments);
  LLVM_DEBUG(dbgs() << "cpsat: log " << what << " over [" << lo
                    << ", " << hi << "]" << (shift ? " (shifted)" : "")
                    << ": " << segments.size() << " segments, ratio " << qlo
                    << "\n");
  return LinearExpr(L);
}

bool FunctionModel::parameters(double limit, SatParameters &params,
                               std::string &why) {
  // Interleaved: every subsolver (LNS included, with one worker too) in
  // batches synchronized deterministically, so the same input and worker
  // count schedule the same way.
  params.set_num_workers(std::max(1, config.numWorkers));
  params.set_interleave_search(true);
  params.set_max_deterministic_time(limit);
  params.set_max_time_in_seconds(limit * 2 + 1);
  params.set_random_seed(0);
  if (!config.solverParameters.empty()) {
    SatParameters extra;
    if (!google::protobuf::TextFormat::ParseFromString(
            config.solverParameters, &extra)) {
      why = "cannot parse the solver parameters '" +
            config.solverParameters + "'";
      return false;
    }
    params.MergeFrom(extra);
  }
  return true;
}

void FunctionModel::completeHint(const LinearExpr &objective) {
  cp.ClearObjective();
  cp.Minimize(objective);
  CpModelProto proto = cp.Build();
  bool fixed = false;
  auto fix = [&](int index, int64_t value) {
    auto *var = proto.mutable_variables(index);
    var->clear_domain();
    var->add_domain(value);
    var->add_domain(value);
    fixed = true;
  };
  for (auto [r, region] : llvm::enumerate(m.regions)) {
    if (iiVar[r] && region.hintII)
      fix(iiVar[r]->index(), *region.hintII);
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      if (!op.hint)
        continue;
      fix(vars[r][i].c.index(), *op.hint);
      for (auto [j, chosen] : llvm::enumerate(vars[r][i].chosen))
        fix(chosen.index(), j == op.hintChoice);
    }
  }
  if (!fixed)
    return;
  Model model;
  SatParameters params;
  std::string why;
  if (!parameters(config.timeLimitS / 4, params, why))
    return;
  model.Add(NewSatParameters(params));
  CpSolverResponse response = SolveCpModel(proto, &model);
  LLVM_DEBUG(dbgs() << "cpsat: complete the hint: "
                    << CpSolverStatus_Name(response.status()) << " objective "
                    << response.objective_value() << " in "
                    << response.wall_time() << " s\n");
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE)
    return;
  cp.ClearHints();
  for (int i = 0; i < proto.variables_size(); ++i)
    cp.AddHint(cp.GetIntVarFromProtoIndex(i), response.solution(i));
}

bool FunctionModel::solve(StringRef phase, const LinearExpr &objective,
                          CpSolverResponse &response, std::string &why) {
  cp.ClearObjective();
  cp.Minimize(objective);
  Model model;
  SatParameters params;
  if (!parameters(config.timeLimitS, params, why))
    return false;
  model.Add(NewSatParameters(params));
  response = SolveCpModel(cp.Build(), &model);
  LLVM_DEBUG(dbgs() << "cpsat: " << phase << ": "
                    << CpSolverStatus_Name(response.status()) << " objective "
                    << response.objective_value() << " bound "
                    << response.best_objective_bound() << " in "
                    << response.wall_time() << " s\n");
  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE)
    return true;
  why = (llvm::Twine(phase) + ": " + CpSolverStatus_Name(response.status()) + " after " +
         std::to_string(response.wall_time()) + " s")
            .str();
  return false;
}

void FunctionModel::hint() {
  for (auto [r, region] : llvm::enumerate(m.regions)) {
    if (iiVar[r] && region.hintII)
      cp.AddHint(*iiVar[r], *region.hintII);
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      if (!op.hint)
        continue;
      cp.AddHint(vars[r][i].c, *op.hint);
      for (size_t j = 0; j < vars[r][i].chosen.size(); ++j)
        cp.AddHint(vars[r][i].chosen[j], j == op.hintChoice);
    }
  }
}

void FunctionModel::extract(const CpSolverResponse &response,
                            CPSATFunctionSolution &solution) {
  DenseMap<unsigned, int64_t> nextId;
  DenseMap<std::pair<unsigned, unsigned>, int64_t> instanceOf;
  for (SlotV &s : slots) {
    if (!SolutionBooleanValue(response, s.used))
      continue;
    int64_t id = nextId[s.opr]++;
    for (auto &t : s.takers)
      if (SolutionBooleanValue(response, t.b))
        instanceOf[{t.region, t.op}] = id;
  }
  for (auto [r, region] : llvm::enumerate(m.regions)) {
    auto &out = solution.regions.emplace_back();
    auto &rv = vars[r];
    size_t n = region.ops.size();
    out.start.resize(n);
    out.opr.resize(n);
    out.inCycle.resize(n);
    out.instance.resize(n);
    SmallVector<int64_t> readyAt(n), mux(n);
    SmallVector<unsigned> choice(n, 0);
    for (size_t i = 0; i < n; ++i) {
      auto &op = region.ops[i];
      if (op.oprs.size() > 1)
        for (size_t j = 0; j < op.oprs.size(); ++j)
          if (SolutionBooleanValue(response, rv[i].chosen[j]))
            choice[i] = j;
      out.start[i] = SolutionIntegerValue(response, rv[i].c);
      out.opr[i] = op.oprs[choice[i]];
      mux[i] = SolutionIntegerValue(response, rv[i].mux);
      readyAt[i] = SolutionIntegerValue(response, rv[i].ready);
      if (auto it = instanceOf.find({unsigned(r), unsigned(i)});
          it != instanceOf.end())
        out.instance[i] = it->second;
      else if (opr(out.opr[i]).info.shareable && isAlone(op, out.opr[i]))
        out.instance[i] = nextId[out.opr[i]]++;
    }
    SmallVector<int64_t> arrival(n, 0);
    for (auto &dep : region.deps) {
      if (dep.auxiliary || dep.distance)
        continue;
      int64_t r0 = readyAt[dep.src] - int64_t(out.start[dep.dst]) * K;
      arrival[dep.dst] = std::max(arrival[dep.dst], r0);
    }
    for (size_t i = 0; i < n; ++i)
      out.inCycle[i] = float(arrival[i] + mux[i]) / 1000.0f;
    out.length = SolutionIntegerValue(response, lastVar[r]);
    out.ii = iiVar[r] ? SolutionIntegerValue(response, *iiVar[r])
                      : region.ii;
  }
  solution.area = double(SolutionIntegerValue(response, area));
  solution.cycles = SolutionIntegerValue(response, cycles);
  solution.luts = SolutionIntegerValue(response, lutUse);
  solution.dsps = SolutionIntegerValue(response, dspUse);
  solution.ffs = SolutionIntegerValue(response, ffUse);
  LLVM_DEBUG({
    for (const Held &h : heldValues)
      if (int64_t n = SolutionIntegerValue(response, h.regs))
        dbgs() << "cpsat: held: region " << h.region << " op " << h.op << " "
               << m.regions[h.region].ops[h.op].name << " width " << h.bits
               << " held " << SolutionIntegerValue(response, h.held)
               << " regs " << n << " (II "
               << (iiVar[h.region]
                       ? SolutionIntegerValue(response, *iiVar[h.region])
                       : int64_t(m.regions[h.region].ii))
               << ")\n";
    auto pct = [](int64_t n, double capacity) {
      return 100.0 * double(n) / std::max(capacity, 1.0);
    };
    dbgs() << "cpsat: resources: " << solution.luts << " LUTs ("
           << SolutionIntegerValue(response, muxLutsTotal) << " in muxes, "
           << pct(solution.luts, config.lutCapacity) << "%), "
           << solution.dsps << " DSPs ("
           << pct(solution.dsps, config.dspCapacity) << "%), " << solution.ffs
           << " FFs (" << SolutionIntegerValue(response, registerFFs)
           << " in registers, " << SolutionIntegerValue(response, enableFFs)
           << " in enables, " << pct(solution.ffs, config.ffCapacity)
           << "%), " << SolutionIntegerValue(response, registerSRLs)
           << " of the LUTs in SRLs\n";
  });
}

LogicalResult FunctionModel::run(CPSATFunctionSolution &solution,
                                 std::string &why) {
  build();
  LLVM_DEBUG({
    size_t ops = 0;
    for (auto &r : m.regions)
      ops += r.ops.size();
    dbgs() << "cpsat: " << m.regions.size() << " regions, " << ops
           << " operations, " << slots.size() << " instances\n";
  });
  hint();

  // One objective: (1 - t) log(area) + t log(cycles), in permille of t; the
  // ends keep a thousandth of the other as a tie-break.
  LinearExpr objective;
  StringRef phase = "cycles";
  if (config.cyclesOnly) {
    objective = cycles;
  } else {
    phase = "tradeoff";
    int64_t permille = std::clamp<int64_t>(
        std::llround(std::clamp(config.tradeoff, 0.0, 1.0) * 1000), 1, 999);
    objective = (1000 - permille) * logOf(area, "area") +
                permille * logOf(cycles, "cycles");
    completeHint(objective);
  }
  CpSolverResponse response;
  if (!solve(phase, objective, response, why))
    return failure();
  extract(response, solution);
  solution.optimal = response.status() == CpSolverStatus::OPTIMAL;
  return success();
}

} // namespace

unsigned
scheduling::cpsatResourceMII(const CPSATFunctionModel::Region &region) {
  unsigned mii = 1;
  for (auto &rsrc : region.resources)
    if (rsrc.limit)
      mii = std::max<unsigned>(mii,
                               (rsrc.ops.size() + rsrc.limit - 1) / rsrc.limit);
  return mii;
}

LogicalResult scheduling::scheduleCPSATFunction(
    const CPSATFunctionModel &model, const CPSATChainingConfig &config,
    CPSATFunctionSolution &solution, std::string &why) {
  FunctionModel fm(model, config);
  return fm.run(solution, why);
}
