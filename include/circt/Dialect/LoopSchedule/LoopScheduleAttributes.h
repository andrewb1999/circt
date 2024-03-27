//===- LoopScheduleAttributes.h - LoopSchedule attribute definitions
//--------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LoopSchedule dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEATTRIBUTES_H
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEATTRIBUTES_H

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/LoopSchedule/LoopScheduleAttributes.h.inc"

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEATTRIBUTES_H
