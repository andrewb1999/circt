//===- Algorithms.h - Library of binding algorithms -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Library of binding algorithms. Each algorithm is a free function that
// takes a problem instance populated by the client and writes instance
// assignments back into it. A successful call guarantees `problem.verify()`
// passes; a failing call leaves the problem in an undefined state.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDING_ALGORITHMS_H
#define CIRCT_BINDING_ALGORITHMS_H

#include "circt/Binding/Problems.h"

namespace circt {
namespace binding {

/// Bind the operations in \p prob via greedy conflict-graph coloring.
///
/// Builds a conflict graph pairwise (ops conflict iff `prob.conflicts(a,b)`
/// is true), then, processing ops in ascending `startTime` order with
/// stable tie-breaking by registration index, assigns each op the lowest-
/// numbered instance not currently occupied by a conflicting neighbor.
/// Fails if `prob.check()` fails or any pool requires more instances than
/// its `instanceLimit` allows.
///
/// This is the default binder for `BindingProblem`. It handles arbitrary
/// AP-of-intervals occupancies and arbitrary concurrency relations, so it
/// correctly covers:
///   - pure sequential regions (single intervals),
///   - pipelined regions (mod-II conflicts),
///   - frames hosting inline pipelines + concurrent static ops,
///   - multiple pipelines launched concurrently in one frame,
///   - cross-region sharing via disjoint concurrency groups.
///
/// Greedy coloring is not optimal on arbitrary conflict graphs (graph
/// coloring is NP-hard in general), but the start-time ordering is exact
/// on pure interval graphs (where it reduces to Left-Edge) and close to
/// optimal on the AP-mixed graphs typical of HLS output. If a tighter
/// bound is ever needed, a DSATUR or clique-partitioning variant can be
/// added as a parallel algorithm without disturbing this one.
LogicalResult bindGreedy(BindingProblem &prob);

/// Bind the operations in \p prob via greedy coloring, restricting each
/// op's candidate instance range by its declared `AccessKind`. Physical
/// ports are laid out as `[0, R) = Read`, `[R, R+W) = Write`, and
/// `[R+W, limit) = ReadWrite`. Ops are processed most-constrained first
/// (RW, then W, then R) so the narrow sub-pool is filled before ops that
/// have a wider legal range compete for it.
LogicalResult bindGreedy(PortKindBindingProblem &prob);

} // namespace binding
} // namespace circt

#endif // CIRCT_BINDING_ALGORITHMS_H
