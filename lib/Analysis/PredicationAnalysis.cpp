//===- SchedulingAnalysis.cpp - scheduling analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving scheduling.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/PredicationAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LogicalResult.h"
#include <limits>

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::scf;
using namespace circt::loopschedule;

/// CyclicSchedulingAnalysis constructs a CyclicProblem for each AffineForOp by
/// performing a memory dependence analysis and inserting dependences into the
/// problem. The client should retrieve the partially complete problem to add
/// and associate operator types.
circt::analysis::PredicationAnalysis::PredicationAnalysis(
    Operation *op, AnalysisManager &am) {
  auto funcOp = cast<func::FuncOp>(op);

  analyzeFuncOp(funcOp);
}

SmallVector<Value>
circt::analysis::PredicationAnalysis::getPredicates(Operation  *op) {
  auto preds = predicates.find(op);
  assert(preds != predicates.end() && "expected problem to exist");
  return preds->second;
}

void circt::analysis::PredicationAnalysis::analyzeFuncOp(
    func::FuncOp funcOp) {
  
}
