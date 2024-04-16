//===- LoopScheduleDependenceAnalysis.h - memory dependence analyses ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving memory access dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_LOOPSCHEDULE_DEPENDENCE_ANALYSIS_H
#define CIRCT_ANALYSIS_LOOPSCHEDULE_DEPENDENCE_ANALYSIS_H

#include "circt/Analysis/NameAnalysis.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Pass/AnalysisManager.h"
#include <utility>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace circt {
namespace analysis {

/// MemoryDependence captures a dependence from one memory operation to another.
/// It represents the destination of the dependence edge, the type of the
/// dependence, and the components associated with each enclosing loop.
struct LoopScheduleDependence {
  LoopScheduleDependence(Operation *source, unsigned distance)
      : source(source), distance(distance) {}

  // The source Operation where this dependence originates.
  Operation *source;

  unsigned distance;
};

/// MemoryDependenceResult captures a set of memory dependencies. The map key is
/// the operation to which the dependencies exist, and the map value is zero or
/// more MemoryDependencies for that operation.
using LoopScheduleDependenceResult =
    std::map<Operation *, SmallVector<LoopScheduleDependence>>;

/// MemoryDependenceAnalysis traverses any AffineForOps in the FuncOp body and
/// checks for affine memory access dependencies. Non-affine memory dependencies
/// are currently not supported. Results are captured in a
/// MemoryDependenceResult, and an API is exposed to query dependencies of a
/// given Operation.
/// TODO(mikeurbach): consider upstreaming this to MLIR's AffineAnalysis.
struct LoopScheduleDependenceAnalysis {
  // Construct the analysis from a FuncOp.
  LoopScheduleDependenceAnalysis(Operation *op,
                                 mlir::AnalysisManager &analysisManager);

  // Returns the dependencies, if any, that the given Operation depends on.
  ArrayRef<LoopScheduleDependence> getDependencies(Operation *);

  // Replaces the dependencies, if any, from the oldOp to the newOp.
  void replaceOp(Operation *oldOp, Operation *newOp);

  // Contains op
  bool containsOp(Operation *op);

private:
  // Store dependence results.
  LoopScheduleDependenceResult results;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_LOOPSCHEDULE_DEPENDENCE_ANALYSIS_H
