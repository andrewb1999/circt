//===- LoopScheduleTypes.h - LoopSchedule type definitions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopSchedule dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULETYPES_H
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULETYPES_H

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/LoopSchedule/LoopScheduleTypes.h.inc"

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULETYPES_H
