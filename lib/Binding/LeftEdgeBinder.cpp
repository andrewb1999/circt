//===- LeftEdgeBinder.cpp - Left-Edge binding -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Left-Edge binding algorithm. Given a set of operations
// with fixed start times, latencies, and a linked resource type, it assigns
// each operation an instance id in [0, instanceLimit) such that no two ops
// on the same instance overlap in cycles.
//
// Left-Edge is optimal for interval graphs: the final instance count per
// pool equals the maximum pool utilization across cycles. Runtime is
// O(n log n) per pool (dominated by the initial sort).
//
// Three variants:
//  - BindingProblem: raw-cycle intervals [start, start+latency-1].
//  - ModuloBindingProblem: residue-class intervals mod II; handles ops
//    whose occupancy wraps around the II boundary.
//  - PortKindBindingProblem: partitions the physical instance id range by
//    access kind (Read / Write / ReadWrite) and restricts each op's
//    candidate set to its compatible sub-pool.
//
//===----------------------------------------------------------------------===//

#include "circt/Binding/Algorithms.h"
#include "circt/Binding/Problems.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt;
using namespace circt::binding;

namespace {

/// Stable op ordering: operations in the order they were registered.
DenseMap<Operation *, unsigned>
buildOpOrder(const BindingProblem::OperationSet &ops) {
  DenseMap<Operation *, unsigned> order;
  order.reserve(ops.size());
  for (auto [i, op] : llvm::enumerate(ops))
    order[op] = (unsigned)i;
  return order;
}

/// Group ops by their linked resource type. Ops without a linked resource
/// are silently skipped — they're not participating in binding.
DenseMap<BindingProblem::ResourceType, SmallVector<Operation *>>
groupByResource(BindingProblem &prob) {
  DenseMap<BindingProblem::ResourceType, SmallVector<Operation *>> byRsrc;
  for (auto *op : prob.getOperations()) {
    if (auto rsrc = prob.getLinkedResourceType(op))
      byRsrc[*rsrc].push_back(op);
  }
  return byRsrc;
}

} // namespace

//===----------------------------------------------------------------------===//
// BindingProblem — plain Left-Edge
//===----------------------------------------------------------------------===//

LogicalResult circt::binding::bindLeftEdge(BindingProblem &prob) {
  if (failed(prob.check()))
    return failure();

  auto opOrder = buildOpOrder(prob.getOperations());
  auto byRsrc = groupByResource(prob);

  for (auto &kv : byRsrc) {
    auto rsrc = kv.first;
    auto &ops = kv.second;
    unsigned limit = *prob.getInstanceLimit(rsrc);

    // Sort by start time, break ties stably by registration order.
    llvm::sort(ops, [&](Operation *a, Operation *b) {
      unsigned sa = *prob.getStartTime(a);
      unsigned sb = *prob.getStartTime(b);
      if (sa != sb)
        return sa < sb;
      return opOrder.lookup(a) < opOrder.lookup(b);
    });

    // For each instance, track the last end time (inclusive) assigned to it.
    // Empty slots have end = sentinel; we treat a slot as free when its last
    // end is strictly less than the candidate op's start.
    SmallVector<std::optional<unsigned>> lastEnd;

    for (auto *op : ops) {
      unsigned start = *prob.getStartTime(op);
      unsigned end = *prob.getEndTime(op);
      int picked = -1;
      for (unsigned k = 0; k < lastEnd.size(); ++k) {
        if (!lastEnd[k] || *lastEnd[k] < start) {
          picked = (int)k;
          break;
        }
      }
      if (picked < 0) {
        // No free existing instance — open a new one.
        if (lastEnd.size() >= limit)
          return prob.getContainingOp()->emitError()
                 << "Resource '" << rsrc.getValue()
                 << "' requires more than " << limit
                 << " instances to bind with non-overlapping intervals";
        picked = (int)lastEnd.size();
        lastEnd.emplace_back();
      }
      lastEnd[picked] = end;
      prob.setInstance(op, (unsigned)picked);
    }
  }

  return prob.verify();
}

//===----------------------------------------------------------------------===//
// ModuloBindingProblem — mod-II Left-Edge with residue-set conflict check
//===----------------------------------------------------------------------===//

LogicalResult circt::binding::bindLeftEdge(ModuloBindingProblem &prob) {
  if (failed(prob.check()))
    return failure();

  auto opOrder = buildOpOrder(prob.getOperations());
  unsigned ii = *prob.getInitiationInterval();

  // Group by resource (reuse base-class view).
  DenseMap<BindingProblem::ResourceType, SmallVector<Operation *>> byRsrc;
  for (auto *op : prob.getOperations())
    if (auto rsrc = prob.getLinkedResourceType(op))
      byRsrc[*rsrc].push_back(op);

  auto occupancy = [&](Operation *op) {
    llvm::SmallDenseSet<unsigned> res;
    unsigned start = *prob.getStartTime(op);
    unsigned lat = *prob.getLatency(op);
    for (unsigned i = 0; i < lat; ++i)
      res.insert((start + i) % ii);
    return res;
  };

  for (auto &kv : byRsrc) {
    auto rsrc = kv.first;
    auto &ops = kv.second;
    unsigned limit = *prob.getInstanceLimit(rsrc);

    // Stable order: sort by start time, then by registration index. In the
    // pipelined case this produces iteration-invariant bindings.
    llvm::sort(ops, [&](Operation *a, Operation *b) {
      unsigned sa = *prob.getStartTime(a);
      unsigned sb = *prob.getStartTime(b);
      if (sa != sb)
        return sa < sb;
      return opOrder.lookup(a) < opOrder.lookup(b);
    });

    // Per-instance residue footprint accumulator.
    SmallVector<llvm::SmallDenseSet<unsigned>> instResidues;

    for (auto *op : ops) {
      auto myRes = occupancy(op);
      int picked = -1;
      for (unsigned k = 0; k < instResidues.size(); ++k) {
        bool conflict = false;
        for (unsigned r : myRes) {
          if (instResidues[k].contains(r)) {
            conflict = true;
            break;
          }
        }
        if (!conflict) {
          picked = (int)k;
          break;
        }
      }
      if (picked < 0) {
        if (instResidues.size() >= limit)
          return prob.getContainingOp()->emitError()
                 << "Resource '" << rsrc.getValue()
                 << "' requires more than " << limit
                 << " instances for mod-" << ii << " binding";
        picked = (int)instResidues.size();
        instResidues.emplace_back();
      }
      for (unsigned r : myRes)
        instResidues[picked].insert(r);
      prob.setInstance(op, (unsigned)picked);
    }
  }

  return prob.verify();
}

//===----------------------------------------------------------------------===//
// PortKindBindingProblem — Left-Edge with access-kind-restricted candidate set
//===----------------------------------------------------------------------===//

LogicalResult circt::binding::bindLeftEdge(PortKindBindingProblem &prob) {
  if (failed(prob.check()))
    return failure();

  auto opOrder = buildOpOrder(prob.getOperations());

  // Group by resource.
  DenseMap<BindingProblem::ResourceType, SmallVector<Operation *>> byRsrc;
  for (auto *op : prob.getOperations())
    if (auto rsrc = prob.getLinkedResourceType(op))
      byRsrc[*rsrc].push_back(op);

  // Given an access kind, produce the legal instance id ranges for this
  // resource's layout [0,R) read, [R,R+W) write, [R+W,limit) rw.
  auto legalRange =
      [&](PortKindBindingProblem::ResourceType rsrc,
          PortKindBindingProblem::AccessKind kind,
          SmallVectorImpl<std::pair<unsigned, unsigned>> &ranges) {
    unsigned r = prob.getReadPorts(rsrc).value_or(0);
    unsigned w = prob.getWritePorts(rsrc).value_or(0);
    unsigned limit = *prob.getInstanceLimit(rsrc);
    switch (kind) {
    case PortKindBindingProblem::AccessKind::Read:
      if (r > 0)
        ranges.emplace_back(0, r);
      if (limit > r + w)
        ranges.emplace_back(r + w, limit);
      break;
    case PortKindBindingProblem::AccessKind::Write:
      if (w > 0)
        ranges.emplace_back(r, r + w);
      if (limit > r + w)
        ranges.emplace_back(r + w, limit);
      break;
    case PortKindBindingProblem::AccessKind::ReadWrite:
      if (limit > r + w)
        ranges.emplace_back(r + w, limit);
      break;
    }
  };

  for (auto &kv : byRsrc) {
    auto rsrc = kv.first;
    auto &ops = kv.second;
    unsigned limit = *prob.getInstanceLimit(rsrc);

    // Process most-constrained first: RW ops, then W, then R. Within a
    // kind, sort by start time, then registration index.
    auto kindRank = [](PortKindBindingProblem::AccessKind k) {
      switch (k) {
      case PortKindBindingProblem::AccessKind::ReadWrite:
        return 0;
      case PortKindBindingProblem::AccessKind::Write:
        return 1;
      case PortKindBindingProblem::AccessKind::Read:
        return 2;
      }
      return 3;
    };
    llvm::sort(ops, [&](Operation *a, Operation *b) {
      int ra = kindRank(*prob.getAccessKind(a));
      int rb = kindRank(*prob.getAccessKind(b));
      if (ra != rb)
        return ra < rb;
      unsigned sa = *prob.getStartTime(a);
      unsigned sb = *prob.getStartTime(b);
      if (sa != sb)
        return sa < sb;
      return opOrder.lookup(a) < opOrder.lookup(b);
    });

    // Per-physical-instance last-end tracking.
    SmallVector<std::optional<unsigned>> lastEnd(limit, std::nullopt);

    for (auto *op : ops) {
      SmallVector<std::pair<unsigned, unsigned>, 2> ranges;
      legalRange(rsrc, *prob.getAccessKind(op), ranges);
      unsigned start = *prob.getStartTime(op);
      unsigned end = *prob.getEndTime(op);
      int picked = -1;
      for (auto [lo, hi] : ranges) {
        for (unsigned k = lo; k < hi; ++k) {
          if (!lastEnd[k] || *lastEnd[k] < start) {
            picked = (int)k;
            break;
          }
        }
        if (picked >= 0)
          break;
      }
      if (picked < 0)
        return prob.getContainingOp()->emitError()
               << "Resource '" << rsrc.getValue()
               << "' has no free port of the required kind for operation at "
                  "cycle "
               << start;
      lastEnd[picked] = end;
      prob.setInstance(op, (unsigned)picked);
    }
  }

  return prob.verify();
}
