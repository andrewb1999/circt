//===- SCFWhileTripCountAnalysis.h - scf.while trip count ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a constant trip count analysis for scf.while loops that
// are structurally for-loops: a single induction iter-arg with a constant
// initial value, a before-region condition of the form
// `arith.cmpi pred %iv, %const`, and an after-region update of the form
// `%iv + constStep`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_SCFWHILETRIPCOUNTANALYSIS_H
#define CIRCT_ANALYSIS_SCFWHILETRIPCOUNTANALYSIS_H

#include "llvm/ADT/APInt.h"
#include <optional>

namespace mlir {
namespace scf {
class WhileOp;
} // namespace scf
} // namespace mlir

namespace circt {
namespace analysis {

/// Attempt to infer a constant trip count for an scf.while loop matching the
/// canonical for-loop pattern:
///
///   * Exactly one iter-arg acts as the induction variable (IV).
///   * The IV's initial value (the matching scf.while operand) is a constant
///     integer.
///   * The before region's `scf.condition` uses an `arith.cmpi` with
///     predicate in {slt, sle, ult, ule, ne} whose lhs is the IV before-block
///     argument and rhs is a constant integer.
///   * The `scf.condition` forwards the IV before-block argument into the
///     after region at the same positional index.
///   * The after region's `scf.yield` at the IV index is
///     `arith.addi %ivAfter, %constStep` (either operand order) with a
///     strictly positive constant step.
///
/// Returns std::nullopt if any of the above fails or the arithmetic cannot be
/// evaluated.
std::optional<llvm::APInt>
getSCFWhileConstantTripCount(mlir::scf::WhileOp whileOp);

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_SCFWHILETRIPCOUNTANALYSIS_H
