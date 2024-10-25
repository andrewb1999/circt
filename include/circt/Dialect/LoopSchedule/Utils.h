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

using PredicateMap = llvm::DenseMap<Operation *, Value>;
using PredicateUse = llvm::DenseMap<Value, SmallVector<Operation *>>;
using ResourceMap = llvm::DenseMap<Operation *, SmallVector<std::string>>;
using ResourceLimits = llvm::StringMap<unsigned>;

Value getMemref(Operation *op);

scheduling::ModuloProblem
getModuloProblem(mlir::scf::ForOp forOp,
                 analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::ChainingModuloProblem getChainingModuloProblem(
    mlir::scf::ForOp forOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::SharedOperatorsProblem getSharedOperatorsProblem(
    mlir::scf::ForOp forOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::ChainingSharedOperatorsProblem getChainingSharedOperatorsProblem(
    mlir::scf::ForOp forOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::SharedOperatorsProblem getSharedOperatorsProblem(
    mlir::func::FuncOp funcOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

scheduling::ChainingSharedOperatorsProblem getChainingSharedOperatorsProblem(
    mlir::func::FuncOp funcOp,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis);

LogicalResult recordMemoryResources(Operation *op, Region &body,
                                    ResourceMap &resourceMap,
                                    ResourceLimits &resourceLimits);

LogicalResult addMemoryResources(Operation *op, Region &body,
                                 scheduling::SharedOperatorsProblem &problem,
                                 ResourceMap &resourceMap,
                                 ResourceLimits &resourceLimits);

LogicalResult ifOpConversion(Operation *op, Region &body,
                             PredicateMap &predicateMap);

void addPredicateDependencies(Operation *op, Region &body,
                              scheduling::SharedOperatorsProblem &problem,
                              const PredicateMap &predicateMap,
                              PredicateUse &predicateUse);

} // namespace loopschedule
} // namespace circt

#endif // MLIR_DIALECT_LOOPSCHEDULE_UTILS_H
