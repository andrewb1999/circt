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
// shared operator instances), with the delay of the mux sharing puts on an
// instance's operands in the chaining.
//
// Time is a cycle and an offset in it, in picoseconds: an operation starts
// at `c * K + z`, `K` the cycle time, its operands arriving `z` into cycle
// `c`. A dependence's destination starts no earlier than its source's result
// arrives: `T_dst + distance * II * K >= R_src`, where a combinational
// operation's result arrives its outgoing delay (after the mux, if shared)
// past its start, a multi-cycle one's its outgoing delay into the cycle it
// is ready. An operation fits its cycle when `z` plus the mux plus its
// incoming delay is at most `K`. In cycles, each dependence also holds as in
// the plain problem (`c_dst + distance * II >= c_src + latency`).
//
// An operation's operator is one of its candidates. A shareable operator's
// operations each take an instance: one of the function's pool (instances
// earlier regions opened, which the operations of this region never run at
// the same time as: they are in other frames) or one this region opens. Two
// operations of this region on one instance issue in different cycles
// (modulo the II in a pipeline). The mux on an instance's operands grows
// with its users, by the given table, and every user's chain takes it: this
// region's in the model, an earlier region's through its recorded slack.
//
// The objective is lexicographic: the region's length (the last operation's
// start: a pipeline's depth at its II), then area (each opened instance, each
// unshared operation, the muxes), then the sum of start times. The length
// may exceed its minimum by `lengthSlack` cycles in the area phase.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_CPSATCHAINING_H
#define CIRCT_SCHEDULING_CPSATCHAINING_H

#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <optional>

namespace circt {
namespace scheduling {

/// What allocation and binding need to know of an operator type.
struct CPSATOperatorInfo {
  /// The area of one instance (a shareable operator) or of one operation's
  /// own logic (any other), in LUT equivalents.
  double area = 0.0;
  /// Instances may serve several operations.
  bool shareable = false;
  /// The area one more user adds to an instance's operand mux.
  double muxAreaPerUser = 0.0;
};

/// An instance of a shareable operator, from the regions scheduled before.
struct CPSATInstance {
  Problem::OperatorType opr;
  int64_t id = 0;
  unsigned users = 0;
  /// The least slack among its users before any mux, in ns: the most mux
  /// delay it can take.
  float rawSlack = 0.0f;
  /// A user is in a pipeline, where the mux's first-user NOR is a signal of
  /// its own (the slower table).
  bool pipelineUser = false;
  /// No other operation may take it (a user's carried result is read past
  /// its ready cycle).
  bool closed = false;
};

struct CPSATChainingConfig {
  /// The combinational budget per cycle, in ns.
  float cycleTime = 1.0f;
  /// The region is a pipeline at the problem's II.
  bool pipelined = false;
  /// Cycles the length may exceed its minimum by in the area phase; none
  /// for no bound.
  std::optional<unsigned> lengthSlack = 0;
  /// The area a cycle of length is worth in the area phase (a region taking
  /// a cycle more only where that saves more).
  double cycleCost = 0.0;
  /// The time limit of each solver phase, in seconds.
  double timeLimitS = 10.0;
  int numWorkers = 1;
  /// The mux delay for 2, 3, ... users, in ns; past the end extrapolated by
  /// the last step. `muxNorDelays` is the table for an instance with a user
  /// in a pipeline.
  ArrayRef<float> muxDelays;
  ArrayRef<float> muxNorDelays;
  /// Only find the length (the first phase), e.g. to test an II.
  bool feasibilityOnly = false;
};

struct CPSATChainingInput {
  /// The operator types each operation may take; one not in the map keeps
  /// its linked operator type. Each must be registered in the problem with
  /// its latency and delays.
  DenseMap<Operation *, SmallVector<Problem::OperatorType, 4>> candidates;
  DenseMap<Problem::OperatorType, CPSATOperatorInfo> operatorInfo;
  /// The function's pool.
  SmallVector<CPSATInstance> pool;
  /// Operations that keep a shareable operator's instance to themselves:
  /// on any choice, or (`aloneMultiCycle`) on a multi-cycle one (a pipeline's
  /// carried result also read after the loop, or passed along to another
  /// iter arg: a combinational one goes into the iter arg's register in its
  /// cycle, a multi-cycle one is read off the instance).
  DenseSet<Operation *> alone;
  DenseSet<Operation *> aloneMultiCycle;
  /// A pipeline's operations whose result it carries to its next iteration,
  /// with the operations that read it there. The next iteration reads the
  /// result from the instance's output, which holds it past its ready cycle
  /// only while no other user drives the instance: on an instance with
  /// other users, every read is in the ready cycle (`c_reader + II ==
  /// c_op + latency`).
  DenseMap<Operation *, SmallVector<Operation *, 2>> carriedReaders;
  /// A known schedule (the simplex's) to start the search from.
  DenseMap<Operation *, unsigned> hint;
  /// Per operator type, the operations of regions not scheduled yet that may
  /// take it. An instance may reserve mux capacity for some of them (its
  /// users' chains take the mux of its users and those reserved, so they can
  /// join it later), each credited the area it would save, less its mux's:
  /// the lookahead a region solved alone would otherwise lack.
  DenseMap<Problem::OperatorType, unsigned> futureUsers;
  /// Per operator type, what one of those saves by joining: the area of the
  /// instance it would open otherwise (the least, over them, of the cheapest
  /// operator each may take).
  DenseMap<Problem::OperatorType, double> futureCredit;
};

struct CPSATChainingResult {
  /// The instance each operation of a shareable operator takes, by id (one
  /// past the pool's for an instance this region opened).
  DenseMap<Operation *, int64_t> instance;
  /// Per operation, the slack before its mux, in ns, and its mux's delay.
  DenseMap<Operation *, float> rawSlack;
  DenseMap<Operation *, float> muxDelay;
  /// Operations whose instance no later operation may take (alone, or a
  /// carried result read past its ready cycle).
  DenseSet<Operation *> closes;
  /// The length and area the solver reached.
  int64_t length = 0;
  double area = 0.0;
  bool optimal = false;
};

/// Allocate, schedule and bind \p prob (a sequential region, its II unset)
/// or a pipeline (its II set). Sets each operation's linked operator type,
/// start time and start time in cycle. Fails when the solver finds no
/// solution in its time.
LogicalResult scheduleCPSATChaining(ChainingSharedOperatorsProblem &prob,
                                    Operation *lastOp,
                                    const CPSATChainingConfig &config,
                                    const CPSATChainingInput &input,
                                    CPSATChainingResult &result);
LogicalResult scheduleCPSATChaining(ChainingModuloProblem &prob,
                                    Operation *lastOp,
                                    const CPSATChainingConfig &config,
                                    const CPSATChainingInput &input,
                                    CPSATChainingResult &result);

/// The mux delay for \p users users in \p table (2, 3, ... users), past its
/// end extrapolated by its last step.
float cpsatMuxDelay(ArrayRef<float> table, unsigned users);

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_CPSATCHAINING_H
