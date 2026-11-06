//===- LoopScheduleDialect.td - LoopSchedule dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEDIALECT_TD
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEDIALECT_TD

include "mlir/IR/DialectBase.td"

def LoopSchedule_Dialect : Dialect {
  let name = "loopschedule";
  let cppNamespace = "::circt::loopschedule";
  let summary = "Representation of scheduled loops";
  // TODO: Migrate custom assembly formats to prop-dict.
  let useStrictPropertiesInAssemblyFormat = 0;

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  // let usePropertiesForAttributes = 1;

  let extraClassDeclaration = [{
    /// Register all LoopSchedule attributes.
    void registerAttributes();
    /// Register all LoopSchedule types.
    void registerTypes();
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEDIALECT_TD
