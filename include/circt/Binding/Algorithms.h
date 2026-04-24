//===- Algorithms.h - Library of binding algorithms -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of binding algorithms. Each algorithm is a
// free function that takes a problem instance populated by the client and
// writes instance assignments back into it. A successful call guarantees
// `problem.verify()` passes; a failing call leaves the problem unchanged
// (or partially assigned but failed — callers should treat state as
// undefined on failure).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDING_ALGORITHMS_H
#define CIRCT_BINDING_ALGORITHMS_H

#include "circt/Binding/Problems.h"

namespace circt {
namespace binding {

/// Bind the operations in \p prob using the Left-Edge algorithm, run
/// independently per resource pool. Optimal for interval graphs: the final
/// instance count per pool equals the maximum pool utilization across
/// cycles. Fails if `prob.check()` fails, or if any pool requires more
/// instances than its `instanceLimit` allows (which indicates the upstream
/// scheduler did not enforce the resource constraint that binding expects).
LogicalResult bindLeftEdge(BindingProblem &prob);

/// Bind the operations in \p prob using Left-Edge within each residue class
/// `mod II`. For fully-pipelined modulo-scheduled regions, each residue
/// class independently reduces to a one-shot assignment: if the scheduler
/// respected the resource limit, the class size is at most `instanceLimit`
/// and any injective mapping suffices. Ops with `latency > 1` are handled
/// by treating their occupancy `[start, start+latency-1] mod II` as the
/// interval. Stable (iteration-invariant) output by tie-breaking on
/// `prob.getOperations()` order, so the same op binds to the same instance
/// across iterations.
LogicalResult bindLeftEdge(ModuloBindingProblem &prob);

/// Bind the operations in \p prob partitioning each resource pool by access
/// kind first. A `Read` op is assigned an id from the pool's read or
/// read/write instances; a `Write` op from write or read/write; etc.
/// Instance ids are globally unique within the resource (i.e. the read
/// sub-pool owns ids `[0, readPorts)`, write owns
/// `[readPorts, readPorts + writePorts)`, and read/write owns the rest).
/// Within each sub-pool, the same Left-Edge procedure runs as in the
/// unconstrained case.
LogicalResult bindLeftEdge(PortKindBindingProblem &prob);

} // namespace binding
} // namespace circt

#endif // CIRCT_BINDING_ALGORITHMS_H
