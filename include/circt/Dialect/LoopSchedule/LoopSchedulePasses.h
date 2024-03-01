//===- Passes.h - HW pass entry points --------------------------*- C++ -*-===//
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

std::unique_ptr<mlir::Pass> createBitwidthMinimization();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LoopSchedule/Passes.h.inc"

} // namespace loopschedule
} // namespace circt

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEPASSES_H