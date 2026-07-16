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
  let summary = "Unrolls loops marked `loopschedule.parallel` in place by "
                "the requested factor.";
  let description = [{
    Consumes the `loopschedule.parallel = N` in-place unroll directive on
    `affine.for` loops (Allo's `s.unroll(axis, factor)` is the producer).

    A factor N > 1 unrolls in place by N — a remainder loop is cloned for
    non-divisible trip counts and keeps the original's attributes (e.g.
    `hls.pipeline`). A factor of 0 means "fully unroll" (Allo's encoding
    for a factor-less `s.unroll(axis)`) and dissolves the loop entirely. A
    factor of 1 requests no unrolling and just erases the directive. Any
    directive that cannot be honored is a loud error, never a silent drop.
    Must run before memory-dependence construction and before AMC
    allocation (see the pass source for why both orderings are
    load-bearing).
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

def LoopScheduleBinding : Pass<"loopschedule-binding", "mlir::ModuleOp"> {
  let summary = "Assign hardware instance ids to scheduled loopschedule ops.";
  let description = [{
    Walks each `loopschedule.func_sequential` / `loopschedule.func_pipeline`
    and runs the binding framework on its scheduled ops, stamping a
    `loopschedule.binding = K` attr identifying the hardware instance
    within each resource's pool.

    Ops whose `oplib.operator` declares a `limit` bind against that cap
    (memory accesses arrive this way). Ops whose operator is backed by an
    extern hw.instance but declares no limit bind in SHARE mode: the pool
    is sized to the op count, so the greedy binder assigns the minimal
    number of physical instances consistent with the schedule — never
    fewer than the concurrent demand, hence zero schedule impact. The FSM
    lowering coalesces same-(operator, binding) ops in one hw.module onto
    a single extern instance.
  }];
  let constructor = "circt::loopschedule::createLoopScheduleBindingPass()";
  let dependentDialects = [
    "circt::loopschedule::LoopScheduleDialect"
  ];
  let options = [
    Option<"bindMemories", "bind-memories", "bool", "true",
           "Bind memory accesses against their declared port limits. "
           "Pipeline integrations that assign memory ports elsewhere "
           "(e.g. AMC port-group binding) should disable this.">,
    Option<"shareOperators", "share-operators", "bool", "true",
           "Bind extern-instance compute operators in share mode "
           "(minimal instances, zero schedule impact).">
  ];
}

def LoopScheduleTestbenchGeneration : Pass<"loopschedule-testbench-generation", "mlir::ModuleOp"> {
  let summary = "Generate a simulation testbench for LoopScheduleToFSM output";
  let constructor = "circt::loopschedule::createLoopScheduleTestbenchGenerationPass()";
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::sv::SVDialect",
    "circt::comb::CombDialect"
  ];
  let options = [
    Option<"dataDir", "data-dir", "std::string", "",
           "Directory containing hex data files for testbench inputs/outputs">
  ];
}

#endif // CIRCT_DIALECT_LOOPSCHEDULE_PASSES_TD
