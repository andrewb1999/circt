//===-- Passes.td - HW pass definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_PASSES_TD
#define CIRCT_DIALECT_LOOPSCHEDULE_PASSES_TD

include "mlir/Pass/PassBase.td"

/* def BitwidthMinimization : Pass<"bitwidth-minimization", "mlir::func::FuncOp"> { */
/*   let summary = "Minimize the bitwidth of loopschedule registers."; */
/*   let constructor =  "circt::loopschedule::createBitwidthMinimization()"; */
/* } */

def MarkMemoryAccesses : Pass<"mark-memory-accesses", "mlir::func::FuncOp"> {
  let summary = "Mark memory accesses with a unique name.";
  let constructor =  "circt::loopschedule::createMarkMemoryAccessesPass()";
  let dependentDialects = ["circt::loopschedule::LoopScheduleDialect"];
}

def ConstructMemoryDependencies : Pass<"construct-memory-dependencies", "mlir::func::FuncOp"> {
  let summary = "Construct memory dependencies operation.";
  let constructor =  "circt::loopschedule::createConstructMemoryDependenciesPass()";
  let dependentDialects = ["circt::loopschedule::LoopScheduleDialect"];
}

#endif // CIRCT_DIALECT_LOOPSCHEDULE_PASSES_TD
