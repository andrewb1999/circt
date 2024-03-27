//===- LoopScheduleOps.h - LoopSchdule Op Definitions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEOPS_H
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEOPS_H

#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#include "circt/Dialect/LoopSchedule/LoopScheduleAttributes.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"

namespace circt {
namespace loopschedule {

LogicalResult verifyLoop(Operation *op);

} // namespace loopschedule
} // namespace circt

#include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedule.h.inc"

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEOPS_H
