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
#include "circt/Dialect/LoopSchedule/LoopScheduleTypes.h"

namespace circt {
namespace loopschedule {

LogicalResult verifyLoop(Operation *op);

} // namespace loopschedule
} // namespace circt

#include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.h.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedule.h.inc"

namespace circt {
namespace loopschedule {

/// Return the `iter_arg_update` op in `loop`'s body that writes
/// `iterArgBlockArg`, or a null op if none exists.
LoopScheduleIterArgUpdateOp getIterArgUpdate(LoopInterface loop,
                                             BlockArgument iterArgBlockArg);

/// Return the new-value SSA (the `value` operand of the matching
/// `iter_arg_update`) for `iterArgBlockArg`, or a null Value if no update is
/// present.
Value getIterArgNewValue(LoopInterface loop, BlockArgument iterArgBlockArg);

/// Return the `iter_arg_update` ops for `loop`, keyed positionally by iter-arg
/// block-argument index (matching `loop.getInits()` / `loop.getBodyArgs()`).
/// Entries are null where no matching update op exists.
SmallVector<LoopScheduleIterArgUpdateOp>
getIterArgUpdatesInOrder(LoopInterface loop);

/// Given an `iter_arg_update` op, return the SSA value outside its enclosing
/// phase (step / pipeline.stage) that carries the same registered value
/// — i.e., the phase result produced by registering `u.getValue()`. This is
/// the "feedback" value that consumers (FSM / Calyx) wire into iter-arg
/// registers. Returns the `value` operand unchanged if the phase does not
/// register it (should not occur for well-formed IR).
Value getIterArgPhaseResult(LoopScheduleIterArgUpdateOp u);

} // namespace loopschedule
} // namespace circt

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEOPS_H
