//===- CPSATChaining.h - Allocation, scheduling and binding in CP-SAT -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A chaining scheduler on OR-Tools' CP-SAT that also allocates (chooses each
// operation's operator among alternatives) and binds (puts operations on
// shared operator instances), for every region of a function at once.
//
// Time is a cycle and an offset in it, in picoseconds: an operation starts
// at `c * K + z`, `K` the cycle time, its operands arriving `z` into cycle
// `c`. A dependence's destination starts no earlier than its source's result
// arrives: `T_dst + distance * II * K >= R_src`, where a combinational
// operation's result arrives its outgoing delay (after the mux, if shared)
// past its start, a multi-cycle one's its outgoing delay into the cycle it
// is ready. An operation fits its cycle when `z` plus the mux plus its
// incoming delay is at most `K`: the clock is a hard constraint. In cycles,
// each dependence also holds as in the plain problem (`c_dst + distance * II
// >= c_src + latency`).
//
// Each region has its own time axis, II, dependences and resources. An
// operation's operator is one of its candidates. A shareable operator's
// instances are shared across all regions (two regions' operations never
// issue at once: they are in different frames); a region's users of one
// instance issue in different cycles (modulo the II in a pipeline). The mux
// on an instance's operands grows with its users, by the measured tables
// (the slower one if a user is in a pipeline), and every user's chain takes
// it. A pipeline's II is a variable where the solver chooses it (a unit
// `hls.pipeline`).
//
// One objective over the function's area A (its LUTs, DSPs and FFs:
// instances, muxes, registers, unshared operators, as shares of the device,
// summed or their largest) and its cycles C (each region's length, a pipeline's
// II and depth, weighted by how often it runs): minimize
//
//   (1 - t) log A + t log C,  that is,  A^(1 - t) C^t,
//
// for the tradeoff t in [0, 1] (0: area, 1: cycles; at 0.5 the area-cycles
// product). It weighs relative changes, so it needs no reference: a 1%
// change of A trades for a (1 - t) / t % change of C. The logs are
// piecewise-linear over geometric breakpoints spanning the range the model's
// own bounds give each, so they are exact at the breakpoints and within
// `logError` between them.
//
// A region is described by index, apart from any problem, so that a
// function can be described while it is lowered (on a copy) and solved
// before it is lowered for real.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_CPSATCHAINING_H
#define CIRCT_SCHEDULING_CPSATCHAINING_H

#include "circt/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace circt {
namespace scheduling {

/// What allocation and binding need to know of an operator type.
struct CPSATOperatorInfo {
  /// The resources of one instance (a shareable operator) or of one
  /// operation's own logic (any other), as measured.
  double lut = 0.0, dsp = 0.0, ff = 0.0;
  /// Instances may serve several operations.
  bool shareable = false;
  /// The width (bits) of each operand its mux selects, and how many there
  /// are: the model takes the mux's delay at that width, and its LUTs.
  unsigned width = 32;
  unsigned operands = 2;
};

struct CPSATChainingConfig {
  /// The combinational budget per cycle, in ns.
  float cycleTime = 1.0f;
  /// The solver's deterministic time limit, in seconds, its workers' work
  /// together (its wall-clock limit is twice that, plus a second).
  double timeLimitS = 10.0;
  /// The workers of its interleaved search: deterministic for a given
  /// count; more finish the same work sooner.
  int numWorkers = 8;
  /// The mux delay for 2, 3, ... users, in ns; past the end extrapolated by
  /// the last step. `muxNorDelays` is the table for an instance with a user
  /// in a pipeline.
  ArrayRef<float> muxDelays;
  ArrayRef<float> muxNorDelays;
  /// The same for operands wider than 32 bits.
  ArrayRef<float> muxDelays64, muxNorDelays64;
  /// The LUTs a shared instance's operand mux takes per input past the
  /// first, per 32 bits of operand. An input is a value: users with one
  /// value on an operand (a constant, an argument, an op's result) feed it
  /// one input (none past the first if all of them do). Fitted to Vivado's
  /// netlists (shared against unshared, PolyBench over AXI at 3.33 ns),
  /// which spread the cost past the operand mux: into the users' input
  /// logic and the memory adapters' read paths, and an add/subtract core's
  /// operation select.
  double muxInputLuts = 12.0;
  /// The exchange rate between area and cycles, 0 (area) to 1 (cycles): the
  /// solver minimizes (1 - t) log(area) + t log(cycles). The ends keep a
  /// thousandth of the other as a tie-break.
  double tradeoff = 0.5;
  /// The most a piecewise-linear log may fall below the log between two
  /// breakpoints (natural-log units: 0.005 is half a percent of the value).
  double logError = 0.005;
  /// The device's resources (the xcv80's by default). The area is each
  /// resource's share of the device: summed, which weighs each by its
  /// scarcity (an xcv80 DSP is 237 LUTs, an FF half of one), or with
  /// `bottleneck` the largest share, the sum a thousandth of it as a
  /// tie-break: as little of the device as its scarcest resource allows.
  double lutCapacity = 2574208, dspCapacity = 10848, ffCapacity = 5148416;
  bool bottleneck = false;
  /// Minimize the cycles alone, within `timeLimitS`: a schedule to lower a
  /// region with, where its area does not matter.
  bool cyclesOnly = false;
  /// SatParameters in protobuf text format, over the settings above (e.g.
  /// "num_workers: 8 interleave_search: true").
  std::string solverParameters;
};

struct CPSATFunctionModel {
  struct Operator {
    std::string name;
    unsigned latency = 0;
    float inc = 0.0f, out = 0.0f; // ns
    CPSATOperatorInfo info;
  };
  struct Dependence {
    unsigned src = 0, dst = 0, distance = 0;
    bool auxiliary = false;
    /// An auxiliary edge's own latency, where it has one.
    std::optional<unsigned> latency;
  };
  struct Resource {
    unsigned limit = 0;
    SmallVector<unsigned> ops;
  };
  struct Op {
    /// The op's name, to check a region against its description.
    std::string name;
    /// Indices into `oprs`; the first is the op's own (allocation's).
    SmallVector<unsigned, 4> oprs;
    /// Keeps a shareable operator's instance to itself: on any choice, or
    /// (`aloneMultiCycle`) on a multi-cycle one (a pipeline's carried result
    /// also read after the loop, or passed along to another iter arg: a
    /// combinational one goes into the iter arg's register in its cycle, a
    /// multi-cycle one is read off the instance).
    bool alone = false, aloneMultiCycle = false;
    /// A pipeline's carried result: the ops reading it in the next
    /// iteration. The next iteration reads the result from the instance's
    /// output, which holds it past its ready cycle only while no other user
    /// drives the instance: on an instance with other users, every read is
    /// in the ready cycle (`c_reader + II == c_op + latency`).
    SmallVector<unsigned, 2> carriedReaders;
    /// A start and a choice (an index into `oprs`) to search from, where
    /// known: a schedule of the region alone.
    std::optional<unsigned> hint;
    unsigned hintChoice = 0;
    /// The bits of its results: a register's width where one is held.
    unsigned width = 0;
    /// The value on each operand, by an id common to every op of the
    /// function reading it (where known: a constant, a function argument, a
    /// value of the region); one past the end is a value of its own.
    SmallVector<unsigned, 3> inputs;
  };
  struct Region {
    SmallVector<Op, 0> ops;
    SmallVector<Dependence, 0> deps;
    SmallVector<Resource, 0> resources;
    unsigned anchor = 0;
    /// The II of a pipeline; 0 for a sequential region.
    unsigned ii = 0;
    /// The pipeline's II is the solver's to choose (a unit `hls.pipeline`)
    /// from the resources' bound up; otherwise it is `ii`.
    bool searchII = false;
    std::optional<unsigned> hintII;
    /// A pipeline's continuation, as its lowering takes it: the op deciding
    /// it (`cond`) is at most the II, every observable effect (`effects`) no
    /// earlier than it, and at cycle 0 where a late decision has no remedy
    /// (`condFirst`: a dynamic-latency access in the body, or the loop's
    /// results used).
    std::optional<unsigned> cond;
    bool condFirst = false;
    SmallVector<unsigned> effects;
    /// How many times the region runs in a run of the function (a
    /// pipeline: is launched), and a pipeline's iterations per launch: the
    /// region's share of the cycles is runs x (length + 1), or launches x
    /// ((iterations - 1) x II + depth + 1).
    int64_t runs = 1;
    int64_t tripCount = 1;
    /// The region whose op (`launch`, an index into its ops) launches this
    /// one, a child loop; none for a function. A child runs from its launch
    /// until the parent's next frame, which awaits it: alongside a sibling
    /// launched in the same cycle, and alongside the parent's ops still in
    /// an instance then (issued by the launch, ready after it). An instance
    /// serves this region (or one inside it) and such a sibling or op
    /// never at once: the model launches the siblings apart, and keeps the
    /// parent's op out of the instance by the launch or after it.
    std::optional<unsigned> parent;
    unsigned launch = 0;
  };
  SmallVector<Operator, 0> oprs;
  SmallVector<Region, 0> regions;
};

struct CPSATFunctionSolution {
  struct Region {
    /// Per op: its start, its operator (an index into the model's `oprs`),
    /// when its operands reach the core in its cycle (ns), and its instance.
    SmallVector<unsigned> start, opr;
    SmallVector<float> inCycle;
    SmallVector<std::optional<int64_t>> instance;
    /// Its last cycle (a pipeline: its depth).
    int64_t length = 0;
    /// The II scheduled at (a pipeline's; 0 for a sequential region).
    unsigned ii = 0;
  };
  SmallVector<Region, 0> regions;
  /// The function's area (the objective's term), resources and cycles, and
  /// whether the solver proved the objective optimal (of the
  /// piecewise-linear logs).
  double area = 0.0;
  int64_t luts = 0, dsps = 0, ffs = 0;
  int64_t cycles = 0;
  bool optimal = false;
};

/// Allocate, schedule and bind all of \p model's regions at once, in one
/// solve. Fails, with \p why, when the solver finds no schedule.
LogicalResult scheduleCPSATFunction(const CPSATFunctionModel &model,
                                    const CPSATChainingConfig &config,
                                    CPSATFunctionSolution &solution,
                                    std::string &why);

/// The mux delay for \p users users in \p table (2, 3, ... users), past its
/// end extrapolated by its last step.
float cpsatMuxDelay(ArrayRef<float> table, unsigned users);

/// The least II the resources allow a pipeline: each limited resource's uses
/// over its limit, rounded up.
unsigned cpsatResourceMII(const CPSATFunctionModel::Region &region);

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_CPSATCHAINING_H
