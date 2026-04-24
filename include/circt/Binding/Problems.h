//===- Problems.h - Modeling of binding problems ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binding (= hardware instance assignment) algorithms consume a completed
// schedule and produce, for each operation, an integer instance id within a
// shared resource pool. The classes in this file are the interface between
// clients and algorithm implementations, and intentionally mirror the
// abstractions in `circt/Scheduling/Problems.h` so that users familiar with
// the scheduling framework find the same shape here.
//
// The binding framework is deliberately decoupled from scheduling: a
// `BindingProblem` carries its own operations, resource types, per-op
// properties (start time, latency, linked resource type), and per-resource
// limits. Callers populate it explicitly from whatever schedule source they
// have — typically the output of a `scheduling::Problem` — without any
// header dependency between the two namespaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDING_PROBLEMS_H
#define CIRCT_BINDING_PROBLEMS_H

#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace circt {
namespace binding {

/// This class models the basic binding problem.
///
/// A problem instance is comprised of:
///
///  - *Operations*: The operations to be assigned hardware instances.
///  - *Resource types*: Named pools of fungible hardware instances, each pool
///    bounded by an *instance limit*. Every operation that participates in
///    binding is linked to exactly one resource type.
///
/// Per-operation inputs (populated by the client, typically from a completed
/// schedule):
///
///  - `startTime` — the cycle in which the operation begins using its
///    assigned instance.
///  - `latency` — the number of consecutive cycles for which the operation
///    occupies its assigned instance, starting at `startTime`. For
///    fully-pipelined hardware units a value of 1 is appropriate; for
///    non-pipelined multi-cycle units (e.g. a sequential divider), pass the
///    full occupancy in cycles.
///  - `linkedResourceType` — the resource pool this operation contends for.
///
/// Per-resource-type inputs:
///
///  - `instanceLimit` — the number of physical instances available in the
///    pool. An operation's assigned instance id must be strictly less than
///    this value.
///
/// Per-operation output (populated by a binder):
///
///  - `instance` — the instance id in `[0, instanceLimit)` that this
///    operation uses.
///
/// The `check...` methods perform validity checks on the inputs before
/// binding. The `verify...` methods check the correctness of the solution
/// determined by a concrete binding algorithm.
class BindingProblem {
public:
  static constexpr auto name = "BindingProblem";

  /// Construct an empty binding problem. \p containingOp is used for its
  /// MLIRContext and to emit diagnostics.
  explicit BindingProblem(Operation *containingOp)
      : containingOp(containingOp) {}
  virtual ~BindingProblem() = default;

protected:
  BindingProblem() = default;

  //===--------------------------------------------------------------------===//
  // Aliases for the problem components
  //===--------------------------------------------------------------------===//
public:
  /// Resource types are distinguished by name (chosen by the client).
  struct ResourceType {
    mlir::StringAttr attr;

    ResourceType() = default;
    ResourceType(mlir::StringAttr attr) : attr(attr) {}

    static ResourceType get(mlir::MLIRContext *ctx, llvm::StringRef name) {
      return ResourceType{mlir::StringAttr::get(ctx, name)};
    }

    mlir::StringAttr getAttr() const { return attr; }
    mlir::StringRef getValue() const { return attr.getValue(); }
    std::string str() const { return attr.str(); }

    friend bool operator==(const ResourceType &lhs, const ResourceType &rhs) {
      return lhs.attr == rhs.attr;
    }
    bool operator!=(const ResourceType &rhs) const { return !(*this == rhs); }
  };

  using OperationSet = llvm::SetVector<Operation *>;
  using ResourceTypeSet = llvm::SetVector<ResourceType>;

protected:
  template <typename T>
  using OperationProperty = llvm::DenseMap<Operation *, std::optional<T>>;
  template <typename T>
  using ResourceTypeProperty = llvm::DenseMap<ResourceType, std::optional<T>>;

  //===--------------------------------------------------------------------===//
  // Containers for problem components and properties
  //===--------------------------------------------------------------------===//
private:
  // Operation containing the ops for this binding problem. Used for its
  // MLIRContext and to emit diagnostics.
  Operation *containingOp = nullptr;

  // Problem components
  OperationSet operations;
  ResourceTypeSet resourceTypes;

  // Operation inputs (populated by the client).
  OperationProperty<unsigned> startTime;
  OperationProperty<unsigned> latency;
  OperationProperty<ResourceType> linkedResourceType;

  // Resource-type inputs (populated by the client).
  ResourceTypeProperty<unsigned> instanceLimit;

  // Operation output (populated by the binder).
  OperationProperty<unsigned> instance;

  //===--------------------------------------------------------------------===//
  // Problem construction
  //===--------------------------------------------------------------------===//
public:
  /// Include \p op in this binding problem.
  void insertOperation(Operation *op) { operations.insert(op); }

  /// Include \p rsrc in this binding problem.
  void insertResourceType(ResourceType rsrc) { resourceTypes.insert(rsrc); }

  /// Retrieves the resource type identified by the client-specific \p name.
  /// The resource type is automatically registered in the binding problem.
  ResourceType getOrInsertResourceType(StringRef name);

  //===--------------------------------------------------------------------===//
  // Access to problem components
  //===--------------------------------------------------------------------===//
public:
  /// Return the operation containing this problem, e.g. to emit diagnostics.
  Operation *getContainingOp() { return containingOp; }
  /// Set the operation containing this problem, e.g. to emit diagnostics.
  void setContainingOp(Operation *op) { containingOp = op; }

  /// Return true if \p op is part of this problem.
  bool hasOperation(Operation *op) { return operations.contains(op); }
  /// Return the set of operations.
  const OperationSet &getOperations() { return operations; }

  /// Return true if \p rsrc is part of this problem.
  bool hasResourceType(ResourceType rsrc) {
    return resourceTypes.contains(rsrc);
  }
  /// Return the set of resource types.
  const ResourceTypeSet &getResourceTypes() { return resourceTypes; }

  //===--------------------------------------------------------------------===//
  // Access to properties
  //===--------------------------------------------------------------------===//
public:
  /// The cycle in which \p op begins using its assigned instance. Input.
  std::optional<unsigned> getStartTime(Operation *op) {
    return startTime.lookup(op);
  }
  void setStartTime(Operation *op, unsigned val) { startTime[op] = val; }

  /// The number of consecutive cycles for which \p op occupies its assigned
  /// instance starting at its `startTime`. Input.
  std::optional<unsigned> getLatency(Operation *op) {
    return latency.lookup(op);
  }
  void setLatency(Operation *op, unsigned val) { latency[op] = val; }

  /// The resource pool \p op contends for. Input.
  std::optional<ResourceType> getLinkedResourceType(Operation *op) {
    return linkedResourceType.lookup(op);
  }
  void setLinkedResourceType(Operation *op, ResourceType rsrc) {
    linkedResourceType[op] = rsrc;
  }

  /// The number of physical instances available in \p rsrc's pool. Input.
  std::optional<unsigned> getInstanceLimit(ResourceType rsrc) {
    return instanceLimit.lookup(rsrc);
  }
  void setInstanceLimit(ResourceType rsrc, unsigned val) {
    instanceLimit[rsrc] = val;
  }

  /// The instance id assigned to \p op, in `[0, instanceLimit(rsrc))`. Output
  /// — populated by a binding algorithm.
  std::optional<unsigned> getInstance(Operation *op) {
    return instance.lookup(op);
  }
  void setInstance(Operation *op, unsigned val) { instance[op] = val; }

  //===--------------------------------------------------------------------===//
  // Access to derived properties
  //===--------------------------------------------------------------------===//
public:
  /// The last cycle (inclusive) in which \p op occupies its assigned instance.
  /// Equals `startTime + latency - 1`. Returns `std::nullopt` if either input
  /// is missing.
  std::optional<unsigned> getEndTime(Operation *op);

  //===--------------------------------------------------------------------===//
  // Properties as string key-value pairs (e.g. for DOT graphs)
  //===--------------------------------------------------------------------===//
public:
  using PropertyStringVector =
      llvm::SmallVector<std::pair<std::string, std::string>, 2>;

  virtual PropertyStringVector getProperties(Operation *op);
  virtual PropertyStringVector getProperties(ResourceType rsrc);
  virtual PropertyStringVector getProperties();

  //===--------------------------------------------------------------------===//
  // Property-specific validators
  //===--------------------------------------------------------------------===//
protected:
  /// \p op has a start time.
  virtual LogicalResult checkStartTime(Operation *op);
  /// \p op has a non-zero latency.
  virtual LogicalResult checkLatency(Operation *op);
  /// \p op is linked to a registered resource type.
  virtual LogicalResult checkLinkedResourceType(Operation *op);
  /// \p rsrc has a non-zero instance limit.
  virtual LogicalResult checkInstanceLimit(ResourceType rsrc);
  /// \p op has an assigned instance in `[0, limit)` for its linked resource.
  virtual LogicalResult verifyInstance(Operation *op);
  /// \p rsrc is not oversubscribed: for every pair of ops `a`, `b` in the
  /// pool with `getInstance(a) == getInstance(b)`, their occupancy intervals
  /// `[start, start+latency-1]` must not overlap.
  virtual LogicalResult verifyUtilization(ResourceType rsrc);

  //===--------------------------------------------------------------------===//
  // Client API for problem validation
  //===--------------------------------------------------------------------===//
public:
  /// Return success if the constructed binding problem is valid (inputs
  /// present and consistent, limits positive, etc.). Called by binding
  /// algorithms before they solve.
  virtual LogicalResult check();
  /// Return success if the computed solution is valid (every op has an
  /// instance, no pool is oversubscribed).
  virtual LogicalResult verify();
};

/// This class models a binding problem where operations belong to a
/// pipelined region with initiation interval `II`. Occupancy intervals are
/// evaluated modulo `II`: two operations conflict on the same instance if
/// and only if their `[start, start+latency-1]` intervals overlap when
/// projected into any residue class `mod II`.
///
/// Pipelined binding is the common case inside `loopschedule.pipeline` /
/// `loopschedule.func_pipeline`: the modulo scheduler has already guaranteed
/// that no residue class oversubscribes the pool, so binding becomes a
/// trivial in-class assignment. The verify contract is what differs from the
/// base problem.
class ModuloBindingProblem : public virtual BindingProblem {
public:
  static constexpr auto name = "ModuloBindingProblem";
  using BindingProblem::BindingProblem;

protected:
  ModuloBindingProblem() = default;

private:
  std::optional<unsigned> initiationInterval;

public:
  /// The initiation interval — the period, in cycles, at which new
  /// iterations enter the pipeline. Every op's occupancy interval is
  /// evaluated modulo this value.
  std::optional<unsigned> getInitiationInterval() { return initiationInterval; }
  void setInitiationInterval(unsigned val) { initiationInterval = val; }

protected:
  /// This problem has a non-zero II.
  virtual LogicalResult checkInitiationInterval();
  /// \p rsrc is not oversubscribed in any residue class `mod II`.
  virtual LogicalResult verifyUtilization(ResourceType rsrc) override;

public:
  virtual LogicalResult check() override;
  virtual LogicalResult verify() override;
};

/// This class models binding on a pool of physical instances partitioned by
/// access kind — e.g. a memory with some read-only ports, some write-only
/// ports, and some read/write ports. Every op declares its `accessKind`; a
/// read op can bind to `Read` or `ReadWrite` instances, a write op to
/// `Write` or `ReadWrite`. Instance ids are globally unique across the
/// resource (`[0, totalInstances)`), but the sub-pool partition determines
/// which ids are legal for each op.
///
/// This is the minimum machinery needed to express multi-port memories with
/// heterogeneous ports without introducing per-port-kind resource types
/// (which would explode the resource namespace).
class PortKindBindingProblem : public virtual BindingProblem {
public:
  static constexpr auto name = "PortKindBindingProblem";
  using BindingProblem::BindingProblem;

protected:
  PortKindBindingProblem() = default;

public:
  enum class AccessKind : uint8_t { Read, Write, ReadWrite };

private:
  OperationProperty<AccessKind> accessKind;
  // Per-resource counts of each port kind. Total instance count is the sum.
  ResourceTypeProperty<unsigned> readPorts;
  ResourceTypeProperty<unsigned> writePorts;
  ResourceTypeProperty<unsigned> readWritePorts;

public:
  /// The access kind for \p op — determines which instances are legal.
  std::optional<AccessKind> getAccessKind(Operation *op) {
    return accessKind.lookup(op);
  }
  void setAccessKind(Operation *op, AccessKind kind) {
    accessKind[op] = kind;
  }

  /// Per-kind port counts. The total instance limit for \p rsrc must equal
  /// `readPorts + writePorts + readWritePorts`.
  std::optional<unsigned> getReadPorts(ResourceType rsrc) {
    return readPorts.lookup(rsrc);
  }
  void setReadPorts(ResourceType rsrc, unsigned val) { readPorts[rsrc] = val; }

  std::optional<unsigned> getWritePorts(ResourceType rsrc) {
    return writePorts.lookup(rsrc);
  }
  void setWritePorts(ResourceType rsrc, unsigned val) {
    writePorts[rsrc] = val;
  }

  std::optional<unsigned> getReadWritePorts(ResourceType rsrc) {
    return readWritePorts.lookup(rsrc);
  }
  void setReadWritePorts(ResourceType rsrc, unsigned val) {
    readWritePorts[rsrc] = val;
  }

protected:
  /// \p op has an access kind.
  virtual LogicalResult checkAccessKind(Operation *op);
  /// Per-kind counts for \p rsrc sum to the instance limit.
  virtual LogicalResult checkPortCounts(ResourceType rsrc);
  /// \p op's assigned instance is compatible with its access kind.
  virtual LogicalResult verifyAccessKind(Operation *op);

public:
  virtual LogicalResult check() override;
  virtual LogicalResult verify() override;
};

} // namespace binding
} // namespace circt

namespace llvm {

template <>
struct DenseMapInfo<circt::binding::BindingProblem::ResourceType> {
  using ResourceType = circt::binding::BindingProblem::ResourceType;
  static ResourceType getEmptyKey() {
    return ResourceType{DenseMapInfo<mlir::StringAttr>::getEmptyKey()};
  }
  static ResourceType getTombstoneKey() {
    return ResourceType{DenseMapInfo<mlir::StringAttr>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ResourceType &r) {
    return DenseMapInfo<mlir::StringAttr>::getHashValue(r.attr);
  }
  static bool isEqual(const ResourceType &a, const ResourceType &b) {
    return a.attr == b.attr;
  }
};

} // namespace llvm

#endif // CIRCT_BINDING_PROBLEMS_H
