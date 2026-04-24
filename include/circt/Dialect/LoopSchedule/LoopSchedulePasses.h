//===- LoopSchedulePasses.h - LoopSchedule pass entry points ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEPASSES_H
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace loopschedule {

// Pull in TableGen-generated Options structs (e.g.
// LoopScheduleTestbenchGenerationOptions) so downstream callers can construct
// the options-aware pass factory.
#define GEN_PASS_DECL
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"

std::unique_ptr<mlir::Pass> createMarkMemoryAccessesPass();
std::unique_ptr<mlir::Pass> createConstructMemoryDependenciesPass();
std::unique_ptr<mlir::Pass> createUnrollSubLoopsPass();
std::unique_ptr<mlir::Pass> createBitwidthReductionForLoopSchedulePass();
std::unique_ptr<mlir::Pass> createUnrollForLoopSchedulePass();
std::unique_ptr<mlir::Pass> createPipelineForLoopSchedulePass();
std::unique_ptr<mlir::Pass> createUnrollMarkedLoopsPass();
std::unique_ptr<mlir::Pass> createLoopScheduleBindingPass();
std::unique_ptr<mlir::Pass> createLoopScheduleTestbenchGenerationPass();
std::unique_ptr<mlir::Pass> createLoopScheduleTestbenchGenerationPass(
    const LoopScheduleTestbenchGenerationOptions &options);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"

} // namespace loopschedule
} // namespace circt

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEPASSES_H
