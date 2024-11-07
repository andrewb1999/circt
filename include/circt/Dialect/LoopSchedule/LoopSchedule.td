//===- LoopSchedule.td - LoopSchedule Definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LOOPSCHEDULE_DIALECT_TD
#define CIRCT_LOOPSCHEDULE_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def LoopSchedule_Dialect : Dialect {
  let name = "loopschedule";
  let cppNamespace = "::circt::loopschedule";
  let summary = "Representation of scheduled loops";

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  // let usePropertiesForAttributes = 1;

  let extraClassDeclaration = [{
    /// Register all LoopSchedule attributes.
    void registerAttributes();
  }];

  let useDefaultAttributePrinterParser = 1;
}

include "circt/Dialect/LoopSchedule/LoopScheduleAttributes.td"
include "circt/Dialect/LoopSchedule/LoopScheduleOps.td"

#endif // CIRCT_LOOPSCHEDULE_DIALECT_TD 
