//===- Utils.h - LoopSchedule dialect utilities -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares a set of utilities for the LoopSchedule dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LOOPSCHEDULE_UTILS_H
#define MLIR_DIALECT_LOOPSCHEDULE_UTILS_H

#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace circt {

namespace loopschedule {

// Lowers affine structures for LoopSchedule while retaining memory dependence
// analysis.
mlir::LogicalResult ifOpHoisting(mlir::MLIRContext &context,
                                 mlir::Operation *op);

mlir::LogicalResult postLoweringOptimizations(mlir::MLIRContext &context,
                                              mlir::Operation *op);

Value getMemref(Operation *op);

scheduling::ModuloProblem
getModuloProblem(mlir::scf::ForOp forOp,
                 analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::SharedOperatorsProblem getSharedOperatorsProblem(
    mlir::scf::ForOp forOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::SharedOperatorsProblem getSharedOperatorsProblem(
    mlir::func::FuncOp funcOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

LogicalResult unrollSubLoops(mlir::affine::AffineForOp &forOp);

LogicalResult replaceMemoryAccesses(
    mlir::MLIRContext &context, mlir::Operation *op,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

LogicalResult bitwidthMinimization(
    mlir::MLIRContext &context, mlir::Operation *op,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

} // namespace loopschedule
} // namespace circt

#endif // MLIR_DIALECT_LOOPSCHEDULE_UTILS_H
