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

using namespace circt;
using namespace circt::binding;

//===----------------------------------------------------------------------===//
// BindingProblem
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
  if (auto latency = getLatency(op))
    psv.emplace_back("latency", std::to_string(*latency));
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
  // For each instance in this pool, collect (start, end) intervals and
  // verify pairwise non-overlap.
  llvm::SmallDenseMap<unsigned, SmallVector<std::pair<unsigned, unsigned>>>
      perInstanceIntervals;
  for (auto *op : getOperations()) {
    auto opRsrc = getLinkedResourceType(op);
    if (!opRsrc || *opRsrc != rsrc)
      continue;
    auto inst = getInstance(op);
    if (!inst)
      continue; // diagnosed by verifyInstance
    unsigned start = *getStartTime(op);
    unsigned end = start + *getLatency(op) - 1;
    perInstanceIntervals[*inst].emplace_back(start, end);
  }

  for (auto &kv : perInstanceIntervals) {
    auto &intervals = kv.second;
    llvm::sort(intervals);
    for (unsigned i = 1; i < intervals.size(); ++i) {
      if (intervals[i].first <= intervals[i - 1].second)
        return getContainingOp()->emitError()
               << "Resource type '" << rsrc.getValue()
               << "' instance " << kv.first
               << " is oversubscribed: ops with overlapping intervals ["
               << intervals[i - 1].first << "," << intervals[i - 1].second
               << "] and [" << intervals[i].first << ","
               << intervals[i].second << "]";
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

std::optional<unsigned> BindingProblem::getEndTime(Operation *op) {
  auto start = getStartTime(op);
  auto lat = getLatency(op);
  if (!start || !lat || *lat == 0)
    return std::nullopt;
  return *start + *lat - 1;
}

//===----------------------------------------------------------------------===//
// ModuloBindingProblem
//===----------------------------------------------------------------------===//

LogicalResult ModuloBindingProblem::checkInitiationInterval() {
  if (!getInitiationInterval() || *getInitiationInterval() == 0)
    return getContainingOp()->emitError("Invalid initiation interval");
  return success();
}

LogicalResult ModuloBindingProblem::verifyUtilization(ResourceType rsrc) {
  unsigned ii = *getInitiationInterval();
  // Per instance, collect the set of occupied residues mod II. Any pair of
  // ops bound to the same instance whose occupancy residue sets overlap is a
  // conflict.
  llvm::SmallDenseMap<unsigned, llvm::SmallDenseSet<unsigned>>
      perInstanceResidues;
  for (auto *op : getOperations()) {
    auto opRsrc = getLinkedResourceType(op);
    if (!opRsrc || *opRsrc != rsrc)
      continue;
    auto inst = getInstance(op);
    if (!inst)
      continue;
    unsigned start = *getStartTime(op);
    unsigned lat = *getLatency(op);
    auto &residues = perInstanceResidues[*inst];
    for (unsigned i = 0; i < lat; ++i) {
      unsigned r = (start + i) % ii;
      auto [_, inserted] = residues.insert(r);
      if (!inserted)
        return getContainingOp()->emitError()
               << "Resource type '" << rsrc.getValue() << "' instance "
               << *inst << " is oversubscribed in residue class " << r
               << " mod " << ii;
    }
  }
  return success();
}

LogicalResult ModuloBindingProblem::check() {
  if (failed(BindingProblem::check()))
    return failure();
  return checkInitiationInterval();
}

LogicalResult ModuloBindingProblem::verify() {
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
