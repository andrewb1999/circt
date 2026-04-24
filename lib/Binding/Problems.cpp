//===- Problems.cpp - Modeling of binding problems ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements base classes for binding problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Binding/Problems.h"

#include "mlir/IR/Operation.h"

#include <cstdint>
#include <numeric>

using namespace circt;
using namespace circt::binding;

//===----------------------------------------------------------------------===//
// BindingProblem — construction and property access
//===----------------------------------------------------------------------===//

BindingProblem::ResourceType
BindingProblem::getOrInsertResourceType(StringRef name) {
  auto rsrc = ResourceType::get(getContainingOp()->getContext(), name);
  resourceTypes.insert(rsrc);
  return rsrc;
}

BindingProblem::PropertyStringVector
BindingProblem::getProperties(Operation *op) {
  PropertyStringVector psv;
  if (auto startTime = getStartTime(op))
    psv.emplace_back("startTime", std::to_string(*startTime));
  if (auto lat = getLatency(op))
    psv.emplace_back("latency", std::to_string(*lat));
  if (auto p = getPeriod(op); p && *p > 0)
    psv.emplace_back("period", std::to_string(*p));
  if (auto c = getCount(op); c && *c != 1)
    psv.emplace_back("count", std::to_string(*c));
  if (auto rsrc = getLinkedResourceType(op))
    psv.emplace_back("rsrc", rsrc->getAttr().str());
  if (auto inst = getInstance(op))
    psv.emplace_back("instance", std::to_string(*inst));
  return psv;
}

BindingProblem::PropertyStringVector
BindingProblem::getProperties(ResourceType rsrc) {
  PropertyStringVector psv;
  if (auto limit = getInstanceLimit(rsrc))
    psv.emplace_back("instanceLimit", std::to_string(*limit));
  return psv;
}

BindingProblem::PropertyStringVector BindingProblem::getProperties() {
  return {};
}

//===----------------------------------------------------------------------===//
// BindingProblem — conflict predicate
//===----------------------------------------------------------------------===//

bool BindingProblem::areConcurrent(Operation *a, Operation *b) {
  auto ga = getConcurrencyGroup(a);
  auto gb = getConcurrencyGroup(b);
  // Unset groups both ways: fall back to "possibly concurrent" — gives the
  // caller back-compat behavior when concurrency info is not provided.
  if (!ga && !gb)
    return true;
  if (!ga || !gb)
    return true;
  return *ga == *gb;
}

/// Classic extended-Euclid GCD.
static uint64_t gcd64(uint64_t a, uint64_t b) {
  while (b != 0) {
    uint64_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/// Does AP-of-intervals {s + i*p : i ∈ [0, c)} × [0, L) intersect
/// {t + j*q : j ∈ [0, d)} × [0, M)?
/// Equivalently, exist i ∈ [0, c), j ∈ [0, d) with
///   |(s + i*p) - (t + j*q)| < max(L, M).
static bool apIntersects(uint64_t s, uint64_t L, uint64_t p, uint64_t c,
                          uint64_t t, uint64_t M, uint64_t q, uint64_t d) {
  // Normalize: treat period-0 as a one-shot (count must be 1).
  if (p == 0) {
    p = 1;
    c = 1;
  }
  if (q == 0) {
    q = 1;
    d = 1;
  }

  uint64_t lat = std::max<uint64_t>(L, 1);
  uint64_t mat = std::max<uint64_t>(M, 1);

  // Both are single activations — direct interval overlap.
  if (c == 1 && d == 1) {
    int64_t diff = (int64_t)s - (int64_t)t;
    if (diff >= 0)
      return (uint64_t)diff < mat;
    return (uint64_t)(-diff) < lat;
  }

  // Each activation of A holds cycles [s + i*p, s + i*p + L - 1].
  // Each activation of B holds cycles [t + j*q, t + j*q + M - 1].
  // They overlap iff there exist i ∈ [0,c), j ∈ [0,d),
  //   k ∈ [0,L), l ∈ [0,M) with
  //     s + i*p + k == t + j*q + l.
  // Rewriting: i*p - j*q == (t + l) - (s + k), i.e. the right-hand side
  // must be representable as i*p - j*q for bounded i,j. For small pool
  // sizes this would be a trivial O(c*d) scan, but many HLS pipelines have
  // large trip counts. Instead, use the classic two-AP overlap formula:
  //
  //   Let g = gcd(p, q). The set {i*p - j*q : i,j ∈ ℤ} = {k*g : k ∈ ℤ}.
  //   So an intersection exists (ignoring bounds) iff (t - s) is within
  //   [-(lat-1), mat-1] of some multiple of g.
  //
  // That covers the "unbounded" case. For bounded counts we additionally
  // need the witness `i, j` to lie in their respective ranges. We compute
  // the smallest non-negative `j` that works for each candidate offset in
  // the overlap window and check if both `i` and `j` are in-range.
  uint64_t g = gcd64(p, q);
  int64_t delta = (int64_t)t - (int64_t)s;

  // Scan every offset in the tolerance window around delta that is aligned
  // on g. The window size is (lat + mat - 1), which is small (~single
  // digits) even for multi-cycle latencies, so this loop is tight.
  int64_t tolLo = -(int64_t)(lat - 1);
  int64_t tolHi = (int64_t)(mat - 1);
  for (int64_t k = tolLo; k <= tolHi; ++k) {
    int64_t target = delta + k;
    // Need i*p - j*q == target for some i ∈ [0,c), j ∈ [0,d).
    // Iterate j: for each j, i = (target + j*q) / p must be an integer in
    // [0, c). The loop is bounded by d, which is the smaller of the two
    // trip counts if we pre-sort. For realistic HLS pipelines (c, d ≤ a
    // few hundred) this is still tight. Pick the smaller as the inner.
    uint64_t innerBound = std::min<uint64_t>(c, d);
    bool swap = c < d;
    uint64_t pa = swap ? q : p;
    uint64_t qb = swap ? p : q;
    int64_t tgt = swap ? -target : target;
    uint64_t outerBound = swap ? d : c;
    for (uint64_t j = 0; j < innerBound; ++j) {
      int64_t num = tgt + (int64_t)j * (int64_t)qb;
      if (num < 0)
        continue;
      if ((uint64_t)num % pa != 0)
        continue;
      uint64_t i = (uint64_t)num / pa;
      if (i < outerBound)
        return true;
    }
    (void)g;
  }
  return false;
}

bool BindingProblem::occupanciesOverlap(Operation *a, Operation *b) {
  unsigned sa = getStartTime(a).value_or(0);
  unsigned la = getLatency(a).value_or(1);
  unsigned pa = getPeriod(a).value_or(0);
  unsigned ca = getCount(a).value_or(1);
  unsigned sb = getStartTime(b).value_or(0);
  unsigned lb = getLatency(b).value_or(1);
  unsigned pb = getPeriod(b).value_or(0);
  unsigned cb = getCount(b).value_or(1);
  return apIntersects(sa, la, pa, ca, sb, lb, pb, cb);
}

bool BindingProblem::conflicts(Operation *a, Operation *b) {
  if (a == b)
    return false;
  if (!areConcurrent(a, b))
    return false;
  return occupanciesOverlap(a, b);
}

//===----------------------------------------------------------------------===//
// BindingProblem — validation
//===----------------------------------------------------------------------===//

LogicalResult BindingProblem::checkStartTime(Operation *op) {
  if (!getStartTime(op))
    return op->emitError("Operation has no start time");
  return success();
}

LogicalResult BindingProblem::checkLatency(Operation *op) {
  auto lat = getLatency(op);
  if (!lat)
    return op->emitError("Operation has no latency");
  if (*lat == 0)
    return op->emitError(
        "Operation has zero latency; binding requires a non-zero "
        "occupancy for every tracked op");
  return success();
}

LogicalResult BindingProblem::checkOccupancy(Operation *op) {
  auto period = getPeriod(op).value_or(0);
  auto count = getCount(op).value_or(1);
  if (period == 0 && count > 1)
    return op->emitError(
        "Operation has count > 1 but no period set; repeating ops must "
        "declare a non-zero period");
  if (period > 0 && count == 0)
    return op->emitError(
        "Operation has period > 0 but count == 0; repeating ops must "
        "declare at least one activation");
  return success();
}

LogicalResult BindingProblem::checkLinkedResourceType(Operation *op) {
  auto rsrc = getLinkedResourceType(op);
  if (!rsrc)
    return op->emitError("Operation is not linked to a resource type");
  if (!hasResourceType(*rsrc))
    return op->emitError("Operation uses an unregistered resource type '")
           << rsrc->getValue() << "'";
  return success();
}

LogicalResult BindingProblem::checkInstanceLimit(ResourceType rsrc) {
  auto limit = getInstanceLimit(rsrc);
  if (!limit)
    return getContainingOp()->emitError()
           << "Resource type '" << rsrc.getValue() << "' has no instance limit";
  if (*limit == 0)
    return getContainingOp()->emitError()
           << "Resource type '" << rsrc.getValue()
           << "' has zero instance limit";
  return success();
}

LogicalResult BindingProblem::verifyInstance(Operation *op) {
  auto inst = getInstance(op);
  if (!inst)
    return op->emitError("Operation has no assigned instance");
  auto rsrc = getLinkedResourceType(op);
  if (!rsrc)
    return op->emitError("Operation has an instance but no linked resource");
  auto limit = getInstanceLimit(*rsrc);
  if (!limit)
    return op->emitError("Operation's linked resource '")
           << rsrc->getValue() << "' has no instance limit";
  if (*inst >= *limit)
    return op->emitError("Operation's assigned instance ")
           << *inst << " is outside the resource's limit " << *limit;
  return success();
}

LogicalResult BindingProblem::verifyUtilization(ResourceType rsrc) {
  // Bucket ops by (resource, instance), then pairwise-check conflicts
  // within each bucket.
  llvm::SmallDenseMap<unsigned, SmallVector<Operation *>> perInstance;
  for (auto *op : getOperations()) {
    auto opRsrc = getLinkedResourceType(op);
    if (!opRsrc || *opRsrc != rsrc)
      continue;
    auto inst = getInstance(op);
    if (!inst)
      continue;
    perInstance[*inst].push_back(op);
  }
  for (auto &kv : perInstance) {
    auto &ops = kv.second;
    for (unsigned i = 0; i < ops.size(); ++i) {
      for (unsigned j = i + 1; j < ops.size(); ++j) {
        if (conflicts(ops[i], ops[j]))
          return getContainingOp()->emitError()
                 << "Resource type '" << rsrc.getValue()
                 << "' instance " << kv.first
                 << " is oversubscribed: two ops have conflicting occupancies";
      }
    }
  }
  return success();
}

LogicalResult BindingProblem::check() {
  for (auto *op : getOperations()) {
    if (failed(checkStartTime(op)))
      return failure();
    if (failed(checkLatency(op)))
      return failure();
    if (failed(checkOccupancy(op)))
      return failure();
    if (failed(checkLinkedResourceType(op)))
      return failure();
  }
  for (auto rsrc : getResourceTypes())
    if (failed(checkInstanceLimit(rsrc)))
      return failure();
  return success();
}

LogicalResult BindingProblem::verify() {
  for (auto *op : getOperations())
    if (failed(verifyInstance(op)))
      return failure();
  for (auto rsrc : getResourceTypes())
    if (failed(verifyUtilization(rsrc)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// PortKindBindingProblem
//===----------------------------------------------------------------------===//

LogicalResult PortKindBindingProblem::checkAccessKind(Operation *op) {
  if (!getAccessKind(op))
    return op->emitError("Operation has no access kind");
  return success();
}

LogicalResult PortKindBindingProblem::checkPortCounts(ResourceType rsrc) {
  auto limit = getInstanceLimit(rsrc);
  if (!limit)
    return failure(); // already diagnosed by base
  unsigned r = getReadPorts(rsrc).value_or(0);
  unsigned w = getWritePorts(rsrc).value_or(0);
  unsigned rw = getReadWritePorts(rsrc).value_or(0);
  if (r + w + rw != *limit)
    return getContainingOp()->emitError()
           << "Resource type '" << rsrc.getValue()
           << "' port-kind counts (" << r << "R + " << w << "W + " << rw
           << "RW = " << (r + w + rw) << ") do not match instance limit "
           << *limit;
  return success();
}

LogicalResult PortKindBindingProblem::verifyAccessKind(Operation *op) {
  auto kind = getAccessKind(op);
  auto rsrc = getLinkedResourceType(op);
  auto inst = getInstance(op);
  if (!kind || !rsrc || !inst)
    return success(); // already diagnosed
  unsigned r = getReadPorts(*rsrc).value_or(0);
  unsigned w = getWritePorts(*rsrc).value_or(0);
  // Layout: [0, r) = Read, [r, r+w) = Write, [r+w, limit) = ReadWrite.
  bool ok = false;
  switch (*kind) {
  case AccessKind::Read:
    ok = *inst < r || *inst >= r + w;
    break;
  case AccessKind::Write:
    ok = (*inst >= r && *inst < r + w) || *inst >= r + w;
    break;
  case AccessKind::ReadWrite:
    ok = *inst >= r + w;
    break;
  }
  if (!ok)
    return op->emitError("Operation's access kind is incompatible with its "
                          "assigned instance ")
           << *inst;
  return success();
}

LogicalResult PortKindBindingProblem::check() {
  if (failed(BindingProblem::check()))
    return failure();
  for (auto *op : getOperations())
    if (failed(checkAccessKind(op)))
      return failure();
  for (auto rsrc : getResourceTypes())
    if (failed(checkPortCounts(rsrc)))
      return failure();
  return success();
}

LogicalResult PortKindBindingProblem::verify() {
  if (failed(BindingProblem::verify()))
    return failure();
  for (auto *op : getOperations())
    if (failed(verifyAccessKind(op)))
      return failure();
  return success();
}
