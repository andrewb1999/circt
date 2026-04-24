//===- GreedyBinder.cpp - Conflict-graph greedy binding -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the greedy conflict-graph binder. Per resource pool:
//   1. Collect the pool's ops.
//   2. Sort by `startTime`, then stable by registration index.
//   3. Precompute a conflict graph (edge iff `problem.conflicts(a, b)`).
//      Using an adjacency set is fine for the sizes typical of HLS frames.
//   4. Color the graph by assigning each op, in order, the lowest-numbered
//      instance not currently occupied by any already-assigned conflicting
//      neighbor.
//
// The ordering by start time is exact-optimal for pure interval graphs
// (degenerates to Left-Edge) and close-to-optimal for AP-mixed graphs. The
// algorithm correctly handles:
//   - sequential regions (pure intervals),
//   - pipelined regions (mod-II conflicts via AP-intersection in the
//     problem's `occupanciesOverlap`),
//   - concurrent inline scopes within a frame (same concurrency group),
//   - disjoint concurrency groups (never conflict, automatic sharing).
//
//===----------------------------------------------------------------------===//

#include "circt/Binding/Algorithms.h"
#include "circt/Binding/Problems.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
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
  for (auto *op : prob.getOperations())
    if (auto rsrc = prob.getLinkedResourceType(op))
      byRsrc[*rsrc].push_back(op);
  return byRsrc;
}

/// Core loop: given an ordered list of ops and a predicate for "which
/// instance ids are legal for this op", assign each its lowest legal
/// instance not conflicting with any already-assigned prior op.
template <typename LegalRangePredicate>
LogicalResult
assignInstances(BindingProblem &prob,
                 ArrayRef<Operation *> ops,
                 unsigned limit,
                 LegalRangePredicate isLegal,
                 BindingProblem::ResourceType rsrc) {
  // Walk ops in the caller-chosen order. Precompute pairwise conflicts on
  // the fly (we only need conflicts between the current op and already-
  // assigned ops, so O(n^2) work total, no persistent graph needed).
  for (unsigned i = 0; i < ops.size(); ++i) {
    Operation *op = ops[i];
    // Collect the set of instances currently "blocked" for this op by
    // conflicts with already-assigned predecessors.
    llvm::SmallDenseSet<unsigned, 8> blocked;
    for (unsigned j = 0; j < i; ++j) {
      Operation *other = ops[j];
      auto otherInst = prob.getInstance(other);
      if (!otherInst)
        continue; // (skipped earlier — shouldn't happen)
      if (prob.conflicts(op, other))
        blocked.insert(*otherInst);
    }
    // Pick the lowest-numbered instance that's both legal for this op
    // and not blocked.
    std::optional<unsigned> picked;
    for (unsigned k = 0; k < limit; ++k) {
      if (!isLegal(k))
        continue;
      if (blocked.contains(k))
        continue;
      picked = k;
      break;
    }
    if (!picked)
      return prob.getContainingOp()->emitError()
             << "Resource '" << rsrc.getValue()
             << "' has no free instance for operation (considered "
             << blocked.size() << " blocked instances, limit " << limit
             << ")";
    prob.setInstance(op, *picked);
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// BindingProblem — conflict-graph greedy
//===----------------------------------------------------------------------===//

LogicalResult circt::binding::bindGreedy(BindingProblem &prob) {
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
      unsigned sa = prob.getStartTime(a).value_or(0);
      unsigned sb = prob.getStartTime(b).value_or(0);
      if (sa != sb)
        return sa < sb;
      return opOrder.lookup(a) < opOrder.lookup(b);
    });

    auto anyInstance = [](unsigned) { return true; };
    if (failed(assignInstances(prob, ops, limit, anyInstance, rsrc)))
      return failure();
  }

  return prob.verify();
}

//===----------------------------------------------------------------------===//
// PortKindBindingProblem — greedy with access-kind-restricted candidate set
//===----------------------------------------------------------------------===//

LogicalResult circt::binding::bindGreedy(PortKindBindingProblem &prob) {
  if (failed(prob.check()))
    return failure();

  auto opOrder = buildOpOrder(prob.getOperations());
  auto byRsrc = groupByResource(prob);

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

  for (auto &kv : byRsrc) {
    auto rsrc = kv.first;
    auto &ops = kv.second;
    unsigned limit = *prob.getInstanceLimit(rsrc);
    unsigned readPorts = prob.getReadPorts(rsrc).value_or(0);
    unsigned writePorts = prob.getWritePorts(rsrc).value_or(0);

    llvm::sort(ops, [&](Operation *a, Operation *b) {
      int ra = kindRank(*prob.getAccessKind(a));
      int rb = kindRank(*prob.getAccessKind(b));
      if (ra != rb)
        return ra < rb;
      unsigned sa = prob.getStartTime(a).value_or(0);
      unsigned sb = prob.getStartTime(b).value_or(0);
      if (sa != sb)
        return sa < sb;
      return opOrder.lookup(a) < opOrder.lookup(b);
    });

    // Walk ops manually (same structure as assignInstances) so the legal
    // predicate can key off the *current* op's access kind.
    for (unsigned i = 0; i < ops.size(); ++i) {
      Operation *op = ops[i];
      auto kind = *prob.getAccessKind(op);

      auto isLegal = [&](unsigned k) {
        // Layout: [0, R) read, [R, R+W) write, [R+W, limit) rw.
        bool isRead = k < readPorts;
        bool isWrite = k >= readPorts && k < readPorts + writePorts;
        bool isRW = k >= readPorts + writePorts;
        switch (kind) {
        case PortKindBindingProblem::AccessKind::Read:
          return isRead || isRW;
        case PortKindBindingProblem::AccessKind::Write:
          return isWrite || isRW;
        case PortKindBindingProblem::AccessKind::ReadWrite:
          return isRW;
        }
        return false;
      };

      llvm::SmallDenseSet<unsigned, 8> blocked;
      for (unsigned j = 0; j < i; ++j) {
        Operation *other = ops[j];
        auto otherInst = prob.getInstance(other);
        if (!otherInst)
          continue;
        if (prob.conflicts(op, other))
          blocked.insert(*otherInst);
      }

      std::optional<unsigned> picked;
      for (unsigned k = 0; k < limit; ++k) {
        if (!isLegal(k))
          continue;
        if (blocked.contains(k))
          continue;
        picked = k;
        break;
      }
      if (!picked)
        return prob.getContainingOp()->emitError()
               << "Resource '" << rsrc.getValue()
               << "' has no free port of the required access kind for "
                  "operation";
      prob.setInstance(op, *picked);
    }
  }

  return prob.verify();
}
