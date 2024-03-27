//===- PassDetails.h - LoopSchedule pass class details ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different LoopSchedule passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_LOOPSCHEDULE_TRANSFORMS_PASSDETAILS_H
#define DIALECT_LOOPSCHEDULE_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace loopschedule {

#define GEN_PASS_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h.inc"

} // namespace loopschedule
} // namespace circt

#endif // DIALECT_LOOPSCHEDULE_TRANSFORMS_PASSDETAILS_H
