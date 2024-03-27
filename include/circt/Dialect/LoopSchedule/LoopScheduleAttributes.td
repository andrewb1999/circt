//===- LoopScheduleAttributes.td - Attributes for LoopSchedule ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Handshake dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEATTRIBUTES_TD
#define CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Attributes 
//===----------------------------------------------------------------------===//

def AccessNameAttr : AttrDef<LoopSchedule_Dialect, "AccessName"> {
  let summary = "Unique name given to a memory access operation";
  let description = [{
  }];

  let parameters = (ins "mlir::StringAttr":$name);

  let mnemonic = "access_name";
  let assemblyFormat = [{
    `<` $name `>`
  }];
}

def DistanceAttr : AttrDef<LoopSchedule_Dialect, "Distance"> {
  let summary = "Distance between dependence memory accesses";
  let description = [{
  }];

  let parameters = (ins "unsigned":$dist);

  let mnemonic = "dist";
  let assemblyFormat = [{ `<` $dist `>` }];
}

#endif // CIRCT_DIALECT_LOOPSCHEDULE_LOOPSCHEDULEATTRIBUTES_TD
