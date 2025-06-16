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

def UnrollSubLoops : Pass<"unroll-sub-loops", "::mlir::func::FuncOp"> {
  let summary = "Unroll loops within other pipelined loops to enable scheduling";
  let description = [{Unroll loops within other pipelined loops to enable scheduling}];
  let constructor = "circt::loopschedule::createUnrollSubLoopsPass()";
  let dependentDialects = ["mlir::scf::SCFDialect", "mlir::func::FuncDialect"];
}

def UnrollMarkedLoops : Pass<"unroll-marked-loops", "mlir::func::FuncOp"> {
  let summary = "Unrolls loops with an `hls.unroll` attribute by a given factor.";
  let description = [{
  }];
  let dependentDialects = ["mlir::scf::SCFDialect,mlir::affine::AffineDialect"];
  let constructor = "circt::loopschedule::createUnrollMarkedLoopsPass()";
}

def BitwidthReductionForLoopSchedule : Pass<"bitwidth-reduction-for-loopschedule", "::mlir::func::FuncOp"> {
  let summary = "Reduce bitwidth of ops before lowering to loopschedule";
  let description = [{Reduce bitwidth of ops before lowering to loopschedule}];
  let constructor = "circt::loopschedule::createBitwidthReductionForLoopSchedulePass()";
  let dependentDialects = ["mlir::scf::SCFDialect", "circt::loopschedule::LoopScheduleDialect",
                           "mlir::func::FuncDialect", "mlir::arith::ArithDialect"];
}

def UnrollForLoopSchedule : Pass<"unroll-for-loopschedule", "::mlir::func::FuncOp"> {
  let summary = "Unroll loops to exploit data-level parallelism";
  let description = [{Unroll loops to exploit data-level parallelism}];
  let constructor = "circt::loopschedule::createUnrollForLoopSchedulePass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::func::FuncDialect"];
}

def PipelineForLoopSchedule : Pass<"pipeline-for-loopschedule", "::mlir::func::FuncOp"> {
  let summary = "Unroll and pipeline loops inner loops";
  let description = [{Unroll and pipeline loops inner loops}];
  let constructor = "circt::loopschedule::createPipelineForLoopSchedulePass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::func::FuncDialect"];
}

#endif // CIRCT_DIALECT_LOOPSCHEDULE_PASSES_TD
