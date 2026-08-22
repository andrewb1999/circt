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
// Every operation's occupancy pattern is expressed as an *arithmetic
// progression of intervals*:
//
//   active at cycles  { startTime + i*period + k : i ∈ [0, count),
//                                                   k ∈ [0, latency) }
//
// With `period = 0` and `count = 1` this degenerates to a single interval
// `[startTime, startTime+latency-1]` — the common case for static ops in a
// sequential region. Pipelined ops set `period = II` and `count = tripCount`,
// which captures the full liveness without collapsing it to a contiguous
// range (important when mixing II > 1 pipelines with static work in the
// same region — interval collapsing would over-allocate instances).
//
// Two operations conflict on a resource instance iff:
//   (1) they share a *concurrency group* (an IR-structure-derived marker
//       identifying ops that can execute simultaneously), AND
//   (2) their AP-of-intervals occupancies intersect.
//
// Ops with disjoint concurrency groups never conflict — they are
// guaranteed non-concurrent by construction (e.g. distinct frames of a
// `loopschedule.func_sequential` serialize). This is how the same binder
// recovers cross-region instance sharing without any post-hoc renumbering.
//
// The binding framework is deliberately decoupled from scheduling: a
// `BindingProblem` carries its own operations, resource types, per-op
// properties (start time, latency, period, count, concurrency group,
// linked resource type), and per-resource limits. Callers populate it
// explicitly from whatever schedule source they have — typically the
// output of a `scheduling::Problem` plus a start-time analysis — without
// any header dependency between the two namespaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDING_PROBLEMS_H
#define CIRCT_BINDING_PROBLEMS_H

#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace circt {
namespace binding {

/// This class models the basic binding problem.
///
/// Per-operation inputs (populated by the client, typically from a completed
/// schedule plus an IR-structure analysis):
///
///  - `startTime` — the cycle at which the operation first activates,
///    measured in the timebase of its `concurrencyGroup`.
///  - `latency` — cycles the operation holds its assigned instance per
///    activation. For fully-pipelined hardware units a value of 1 is
///    appropriate; for non-pipelined multi-cycle units (e.g. a sequential
///    divider), pass the full occupancy in cycles. Must be non-zero.
///  - `period` — cycles between successive activations. `0` (default)
///    denotes a non-repeating op (a single activation).
///  - `count` — number of activations. Default is `1`. Ignored when
///    `period == 0`.
///  - `concurrencyGroup` — a pointer to the IR op whose timebase this op
///    lives in (typically the innermost inline frame or func_pipeline).
///    Ops in different concurrency groups are considered non-concurrent
///    and never conflict. An unset group is treated as a universal group
///    that may be concurrent with any other unset group — set this
///    explicitly if you want cross-region sharing.
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
///  - `instance` — the instance id in `[0, instanceLimit)` this operation
///    uses.
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
  OperationProperty<unsigned> period;
  OperationProperty<unsigned> count;
  OperationProperty<Operation *> concurrencyGroup;
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
  /// The cycle in which \p op first activates (in its concurrency group's
  /// timebase). Input.
  std::optional<unsigned> getStartTime(Operation *op) {
    return startTime.lookup(op);
  }
  void setStartTime(Operation *op, unsigned val) { startTime[op] = val; }

  /// Cycles \p op holds its assigned instance per activation. Input.
  std::optional<unsigned> getLatency(Operation *op) {
    return latency.lookup(op);
  }
  void setLatency(Operation *op, unsigned val) { latency[op] = val; }

  /// Cycles between successive activations of \p op. `0` (or unset) means
  /// \p op has a single activation (non-repeating). Input.
  std::optional<unsigned> getPeriod(Operation *op) {
    return period.lookup(op);
  }
  void setPeriod(Operation *op, unsigned val) { period[op] = val; }

  /// Number of activations of \p op. Default is 1 when unset. Ignored when
  /// `period == 0`. Input.
  std::optional<unsigned> getCount(Operation *op) { return count.lookup(op); }
  void setCount(Operation *op, unsigned val) { count[op] = val; }

  /// The IR op whose timebase \p op lives in (typically the innermost
  /// inline frame or func_pipeline). Ops sharing a concurrency group may
  /// execute simultaneously; ops with distinct groups are guaranteed
  /// non-concurrent and never conflict.
  std::optional<Operation *> getConcurrencyGroup(Operation *op) {
    return concurrencyGroup.lookup(op);
  }
  void setConcurrencyGroup(Operation *op, Operation *group) {
    concurrencyGroup[op] = group;
  }

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

  /// The instance id assigned to \p op, in `[0, instanceLimit(rsrc))`.
  /// Output — populated by a binding algorithm.
  std::optional<unsigned> getInstance(Operation *op) {
    return instance.lookup(op);
  }
  void setInstance(Operation *op, unsigned val) { instance[op] = val; }

  //===--------------------------------------------------------------------===//
  // Conflict predicate
  //===--------------------------------------------------------------------===//
public:
  /// True iff \p a and \p b can ever be live at the same cycle on the same
  /// resource instance. Decomposes into (1) a concurrency check — do they
  /// share a group — and (2) an AP-of-intervals intersection on their
  /// occupancies. Virtual so specializations (e.g. port-kind) can refine.
  virtual bool conflicts(Operation *a, Operation *b);

protected:
  /// True iff \p a and \p b may execute at overlapping wall-clock cycles.
  /// Default: same concurrency group (or both unset, treated as a shared
  /// universal group).
  virtual bool areConcurrent(Operation *a, Operation *b);

  /// True iff \p a and \p b's AP-of-intervals occupancies intersect,
  /// interpreted in a shared timebase. Caller must have already established
  /// concurrency (same group); this function does the arithmetic.
  bool occupanciesOverlap(Operation *a, Operation *b);

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
  /// If \p op sets `period > 0`, it also has a non-zero `count`.
  virtual LogicalResult checkOccupancy(Operation *op);
  /// \p op is linked to a registered resource type.
  virtual LogicalResult checkLinkedResourceType(Operation *op);
  /// \p rsrc has a non-zero instance limit.
  virtual LogicalResult checkInstanceLimit(ResourceType rsrc);
  /// \p op has an assigned instance in `[0, limit)` for its linked resource.
  virtual LogicalResult verifyInstance(Operation *op);
  /// \p rsrc is not oversubscribed: for every pair of ops `a`, `b` in the
  /// pool with `getInstance(a) == getInstance(b)`, `conflicts(a, b)` is
  /// false.
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

/// This class models binding on a pool of physical instances partitioned by
/// access kind — e.g. a memory with some read-only ports, some write-only
/// ports, and some read/write ports. Every op declares its `accessKind`; a
/// read op can bind to `Read` or `ReadWrite` instances, a write op to
/// `Write` or `ReadWrite`. Instance ids are globally unique across the
/// resource (`[0, totalInstances)`), but the sub-pool partition determines
/// which ids are legal for each op.
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

// DenseMapInfo no longer defines sentinel keys (getEmptyKey/getTombstoneKey);
// hash and equality are all a specialization provides.
template <>
struct DenseMapInfo<circt::binding::BindingProblem::ResourceType> {
  using ResourceType = circt::binding::BindingProblem::ResourceType;
  static unsigned getHashValue(const ResourceType &r) {
    return DenseMapInfo<mlir::StringAttr>::getHashValue(r.attr);
  }
  static bool isEqual(const ResourceType &a, const ResourceType &b) {
    return a.attr == b.attr;
  }
};

} // namespace llvm

#endif // CIRCT_BINDING_PROBLEMS_H
