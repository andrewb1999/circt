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

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace circt {

namespace loopschedule {

// Lowers affine structures for LoopSchedule while retaining memory dependence
// analysis.
mlir::LogicalResult
lowerAffineStructures(mlir::MLIRContext &context, mlir::Operation *op,
                      analysis::MemoryDependenceAnalysis &memoryDependence);

mlir::LogicalResult postLoweringOptimizations(mlir::MLIRContext &context,
                                              mlir::Operation *op);

Value getMemref(Operation *op);

scheduling::ModuloProblem
getModuloProblem(mlir::affine::AffineForOp forOp,
                 analysis::MemoryDependenceAnalysis &dependenceAnalysis);

scheduling::SharedOperatorsProblem getSharedOperatorsProblem(
    mlir::affine::AffineForOp forOp,
    analysis::MemoryDependenceAnalysis &dependenceAnalysis);

scheduling::SharedOperatorsProblem getSharedOperatorsProblem(
    mlir::func::FuncOp funcOp,
    analysis::MemoryDependenceAnalysis &dependenceAnalysis);

LogicalResult unrollSubLoops(mlir::affine::AffineForOp &forOp);

LogicalResult
replaceMemoryAccesses(mlir::MLIRContext &context, mlir::Operation *op,
                      analysis::MemoryDependenceAnalysis &dependenceAnalysis);

LogicalResult
bitwidthMinimization(mlir::MLIRContext &context, mlir::Operation *op,
                     analysis::MemoryDependenceAnalysis &dependenceAnalysis);

} // namespace loopschedule
} // namespace circt

#endif // MLIR_DIALECT_LOOPSCHEDULE_UTILS_H
