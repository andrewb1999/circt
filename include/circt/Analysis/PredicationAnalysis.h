//===- SchedulingAnalysis.h - scheduling analyses -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_PREDICATION_ANALYSIS_H
#define CIRCT_ANALYSIS_PREDICATION_ANALYSIS_H

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
class AnalysisManager;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

using namespace mlir;
using namespace circt::scheduling;

namespace circt {
namespace analysis {

struct PredicationAnalysis {
  PredicationAnalysis(Operation *funcOp, AnalysisManager &am);

  SmallVector<Value> getPredicates(Operation *op);

private:
  void analyzeFuncOp(func::FuncOp funcOp);

  DenseMap<Operation *, SmallVector<Value>> predicates;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_PREDICATION_ANALYSIS_H
