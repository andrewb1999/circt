//===- LoopScheduleTypes.td - Types for LoopSchedule -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines LoopSchedule dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULETYPES_TD
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULETYPES_TD

include "mlir/IR/AttrTypeBase.td"

class LoopScheduleType<string name, string typeMnemonic>
    : TypeDef<LoopSchedule_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def HandleType : LoopScheduleType<"Handle", "handle"> {
  let summary = "SSA completion token for a launched variable-latency child";
  let description = [{
    A `!loopschedule.handle` is an opaque SSA value produced by a
    `loopschedule.launch` op. It represents the completion of the child loop
    started by that launch. Consumers use `loopschedule.await` on a handle
    (inside a frame's await region) to block until the launched child is done
    and, if the child yields values, to recover those values.
  }];
}

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULETYPES_TD
