//===- SCFToLoopSchedule.h ------------------------------------------------===//
//-------------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SCFTOLOOPSCHEDULE_H_
#define CIRCT_CONVERSION_SCFTOLOOPSCHEDULE_H_

#include "circt/Dialect/OpLib/OpLibDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>

namespace circt {
#define GEN_PASS_DECL_SCFTOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createSCFToLoopSchedulePass(const SCFToLoopScheduleOptions &options = {});

} // namespace circt

#endif // CIRCT_CONVERSION_SCFTOLOOPSCHEDULE_H_