//===- LoopScheduleToFSM.h - LoopSchedule to FSM pass entry point ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the LoopScheduleToFSM pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LOOPSCHEDULETOFSM_H
#define CIRCT_CONVERSION_LOOPSCHEDULETOFSM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_LOOPSCHEDULETOFSM
#include "circt/Conversion/Passes.h.inc"

/// Create a LoopSchedule to FSM + HW conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createLoopScheduleToFSMPass();

} // namespace circt

#endif // CIRCT_CONVERSION_LOOPSCHEDULETOFSM_H
