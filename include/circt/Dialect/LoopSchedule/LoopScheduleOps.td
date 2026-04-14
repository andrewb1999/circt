//===- LoopScheduleOps.td - LoopSchedule Op Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LoopSchedule Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCIT_LOOP_SCHEDULE_OPS_TD
#define CIRCIT_LOOP_SCHEDULE_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "circt/Dialect/LoopSchedule/LoopSchedule.td"
include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.td"

class LoopScheduleOp<string mnemonic, list<Trait> traits = []> :
  Op<LoopSchedule_Dialect, mnemonic, traits>;

def LoopSchedulePipelineOp : LoopScheduleOp<"pipeline", [LoopOpInterface]> {
  let summary = "LoopSchedule dialect pipeline-loop.";
  let description = [{
    The `loopschedule.pipeline` operation represents a statically scheduled
    pipeline stucture that executes while a condition is true. For more details,
    see: https://llvm.discourse.group/t/rfc-representing-pipelined-loops/4171.

    A pipeline captures the result of scheduling, and is not generally safe to
    transform, besides lowering to hardware dialects. For more discussion about
    relaxing this, see: https://github.com/llvm/circt/issues/2204.

    This is the top-level operation representing a high-level pipeline. It is
    not isolated from above, but could be if this is helpful. A pipeline
    contains a single `stages` region.

    The pipeline may accept an optional `inits`, similar to the SCF dialect,
    for representing loop-carried values like induction variables or reductions.
    When the pipeline starts execution, the registers indicated as `inits`
    by `loopschedule.terminator` should be initialized to the initial
    values specified in the `inits` section here. The `inits` relate to
    the initiation interval of the loop. The maximum distance in stages between
    where an `init` is used and where that `init` is registered must be
    less than the loop's initiation interval. For example, with II=1, each
    `init` must be used and registered in the same stage.

    The pipeline initiates new iterations while the condition operand of
    `loopschedule.terminator` is `true : i1`. The condition must be produced
    by the first `loopschedule.pipeline.stage`.

    The single-block `stages` region wraps `loopschedule.pipeline.stage`
    operations. It has a `loopschedule.terminator` terminator, which can
    both return results from the pipeline and register `inits`. Stages may
    access SSA values dominating the pipeline, as well as `inits`, which are
    block arguments.
  }];

  let arguments = (ins
    I64Attr:$II,
    OptionalAttr<I64Attr>:$tripCount,
    Variadic<AnyType>:$inits
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$stages
  );

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$II,
                   "std::optional<IntegerAttr>": $tripCount,
                   "mlir::ValueRange":$inits)>
  ];

  let extraClassDeclaration = [{
    Block &getStagesBlock() { return getStages().front(); }

    Block::BlockArgListType getBodyArgs() {
      return getStagesBlock().getArguments();
    }
    Block *getBodyBlock() {
      return &getStagesBlock();
    }
    Value getConditionValue();
    Operation::operand_range getTerminatorIterArgs();
    std::optional<uint64_t> getBound() {
      return getTripCount();
    }
    bool isPipelined() {
      return true;
    }

    uint64_t getBodyLatency();
    bool canStall();
  }];

  let extraClassDefinition = [{
    Operation::operand_range $cppClass::getTerminatorIterArgs() {
      return cast<LoopScheduleTerminatorOp>(getBodyBlock()->getTerminator()).getIterArgs();
    }
  }];
}

def LoopSchedulePipelineStageOp : LoopScheduleOp<"pipeline.stage",
    [HasParent<"LoopSchedulePipelineOp">,
     PhaseOpInterface,
     SingleBlockImplicitTerminator<"LoopScheduleRegisterOp">]> {
  let summary = "LoopSchedule dialect pipeline stage.";
  let description = [{
    This operation has a single-block region which dictates the operations that
    may occur concurrently.

    It has a `start` attribute, which indicates the start cycle for this stage.

    It may have an optional `when` predicate, which supports conditional
    execution for each stage. This is in addition to the `condition` region that
    controls the execution of the whole pipeline. A stage with a `when`
    predicate should only execute when the predicate is `true : i1`, and push a
    bubble through the pipeline otherwise.

    It has a `register` terminator, which passes the concurrently
    computed values forward to the next stage.

    Any stage may access `inits`. If a stage accesses an `init` after
    the stage in which it is defined, it is up to lowering passes to preserve
    this value until the last stage that needs it.

    Other than `inits`, stages may only access SSA values dominating the
    pipeline or SSA values computed by any previous stage. This ensures the
    stages capture the coarse-grained schedule of the pipeline and how values
    feed forward and backward.
  }];

  let arguments = (ins
    I64Attr:$start,
    I64Attr:$end,
    Optional<I1>:$when
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    `start` `=` $start `end` `=` $end (`when` $when^)? $body (`:` qualified(type($results))^)? attr-dict
  }];

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$start,
               "mlir::IntegerAttr":$end)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    unsigned getStageNumber();

    std::string getRegisterNamePrefix() {
      return "stage_" + std::to_string(getStageNumber());
    }
    bool isStatic() {
      return true;
    }
    std::optional<uint64_t> getStartTime() {
      return getStart();
    }
    LoopScheduleRegisterOp getRegisterOp();
  }];
}

def LoopScheduleSequentialOp : LoopScheduleOp<"sequential", [LoopOpInterface]> {
  let summary = "LoopSchedule dialect sequential while-loop.";
  let description = [{
    The `loopschedule.sequential` operation represents a statically scheduled
    sequential loop that executes while a condition is true.

    A sequential loop captures the result of scheduling, and is not generally 
    safe to transform, besides lowering to hardware dialects. For more 
    discussion about relaxing this, see:
    https://github.com/llvm/circt/issues/2204.

    This is the top-level operation representing a sequential loop. It is
    not isolated from above, but could be if this is helpful. A sequential
    loop contains a single `schedule` region.

    The sequential loop may accept an optional `inits`, similar to the SCF
    dialect, for representing loop-carried values like induction variables or
    reductions. When the sequential loop starts execution, the registers
    indicated as `inits` by `loopschedule.terminator` should be initialized
    to the initial values specified in the `inits` section here.

    The loop initiates new iterations while the condition operand of
    `loopschedule.terminator` is `true : i1`. The condition must be produced
    by the first `loopschedule.step`.

    The single-block `schedule` region wraps `loopschedule.step`
    operations. It has a `loopschedule.terminator` terminator, which can
    both return results from the loop and register `inits`. Steps may
    access SSA values dominating the loop, as well as `inits`, which are
    block arguments.

    In contrast to the `loopschedule.pipeline` op, sequential loops guarantee
    that only one iteration is in-flight at a time. `loopschedule.sequential`
    ops are also allowed to nest other sequential or pipeline loops within
    to represent nested loops.
  }];

  let arguments = (ins
    OptionalAttr<I64Attr>:$tripCount,
    Variadic<AnyType>:$inits
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$schedule
  );

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes,
                   "std::optional<IntegerAttr>": $tripCount,
                   "mlir::ValueRange":$inits)>
  ];

  let extraClassDeclaration = [{
    Block &getScheduleBlock() { return getSchedule().front(); }
    Block::BlockArgListType getBodyArgs() {
      return getScheduleBlock().getArguments();
    }
    Block *getBodyBlock() {
      return &getScheduleBlock();
    }
    Value getConditionValue();
    Operation::operand_range getTerminatorIterArgs();
    std::optional<uint64_t> getBound() {
      return getTripCount();
    }
    bool isPipelined() {
      return false;
    }
    bool canStall();
  }];

  let extraClassDefinition = [{
    Operation::operand_range $cppClass::getTerminatorIterArgs() {
      return cast<LoopScheduleTerminatorOp>(getBodyBlock()->getTerminator()).getIterArgs();
    }
  }];
}

def LoopScheduleStepOp : LoopScheduleOp<"step",
    [ParentOneOf<["LoopScheduleSequentialOp", "mlir::func::FuncOp"]>,
     SingleBlockImplicitTerminator<"LoopScheduleRegisterOp">,
     PhaseOpInterface]> {
  let summary = "LoopSchedule dialect schedule step.";
  let description = [{
    This operation has a single-block region which dictates the operations that
    must start concurrently.

    It may have an optional `when` predicate, which supports conditional
    execution for each step. A step with a `when` predicate should only 
    execute when the predicate is `true : i1`, and continue onto the next
    step otherwise.

    It has a `register` terminator, which passes the concurrently
    computed values forward to a future step.
  }];

  let arguments = (ins
    Optional<I1>:$when
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    (`when` $when^)? $body (`:` qualified(type($results))^)? attr-dict
  }];

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    unsigned getStepNumber();

    std::string getRegisterNamePrefix() {
      return "step_" + std::to_string(getStepNumber());
    }
    bool isStatic() {
      return false;
    }
    std::optional<uint64_t> getStartTime() {
      return std::nullopt;
    }
    LoopScheduleRegisterOp getRegisterOp();
  }];
}

def LoopScheduleDelayOp : LoopScheduleOp<"delay",
    [ParentOneOf<["LoopScheduleStepOp", "LoopScheduleDelayOp"]>,
     SingleBlockImplicitTerminator<"LoopScheduleRegisterOp">]> {
  let summary = "Delays the start of its body by `latency` cycles within "
                "the enclosing step.";
  let description = [{
    The `loopschedule.delay` operation models a sub-step start-time offset.
    Operations in its single-block body are scheduled to begin executing
    `latency` cycles after the start of the enclosing `loopschedule.step`.
    Values referenced by ops outside the delay region are passed out via
    its terminating `loopschedule.register`.

    A delay op contributes `latency + max(child end cycles)` cycles to its
    enclosing step's total latency. Multiple delay ops in the same step
    body run concurrently with each other and with the step's offset-0 ops
    (the enclosing step is conceptually a static-par).
  }];

  let arguments = (ins I64Attr:$latency);

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    $latency $body (`->` qualified(type($results))^)? attr-dict
  }];

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "uint64_t":$latency, "mlir::TypeRange":$resultTypes)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    LoopScheduleRegisterOp getRegisterOp();
  }];
}

def LoopScheduleRegisterOp : LoopScheduleOp<"register",
    [ParentOneOf<["LoopSchedulePipelineStageOp", "LoopScheduleStepOp",
                  "LoopScheduleDelayOp"]>, Terminator]> {
  let summary = "LoopSchedule dialect loop register.";
  let description = [{
    The `loopschedule.register` terminates a pipeline stage and
    "registers" the values specified as operands. These values become the
    results of the stage.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = [{
    $operands (`:` qualified(type($operands))^)? attr-dict
  }];

  let builders = [OpBuilder<(ins), [{ return; }]>];

  let hasVerifier = 1;
}

def LoopScheduleTerminatorOp : LoopScheduleOp<"terminator",
    [ParentOneOf<["LoopSchedulePipelineOp", "LoopScheduleSequentialOp"]>, Terminator, AttrSizedOperandSegments]> {
  let summary = "LoopSchedule dialect terminator.";
  let description = [{
    The `loopschedule.terminator` operation represents the terminator of
    a `loopschedule.pipeline` or `loopschedule.sequential` op.

    The `condition` operand is an `i1` value that controls whether the loop
    initiates a new iteration. It must be produced by the first phase
    (`loopschedule.pipeline.stage` or `loopschedule.step`) of the body.

    The `results` section accepts a variadic list of values which become the
    loop's return values. These must be results of a phase, and their types
    must match the loop's return types. The results need not be defined in
    the final phase, and it is up to lowering passes to preserve these values
    until the final phase is complete.

    The `iter_args` section accepts a variadic list of values which become the
    next iteration's `iter_args`. These may be the results of any phase, and
    their types must match the loop's `iter_args` types.

    The optional `await` section accepts a variadic list of
    `!loopschedule.handle` values. Before the terminator fires its transition,
    every listed handle must have completed. This is a shorthand for an
    iteration-boundary barrier on launched children whose completion value is
    not otherwise consumed in the body.
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$iter_args,
    Variadic<AnyType>:$results,
    Variadic<HandleType>:$await
  );

  let assemblyFormat = [{
    `condition` `(` $condition `)` `,`
    `iter_args` `(` $iter_args `)` `,`
    (`await` `(` $await^ `)` `,`)?
    `results` `(` $results `)` `:`
    functional-type($iter_args, $results) attr-dict
  }];

  let hasVerifier = 1;
}

def LoopScheduleLoadOp : LoopScheduleOp<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "::llvm::cast<MemRefType>($_self).getElementType()">
      ]> {
  let summary = "load operation";
  let description = [{
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<AnyTypeOf<[AnyInteger, Index]>>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);
  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return ::llvm::cast<MemRefType>(getMemRef().getType());
    }
  }];

  /* let hasFolder = 1; */
  let hasVerifier = 1;

  let assemblyFormat = "$memref `[` $indices `:` type($indices) `]` attr-dict `:` type($memref)";
}

def LoopScheduleStoreOp : LoopScheduleOp<"store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "::llvm::cast<MemRefType>($_self).getElementType()">
      ]> {
  let summary = "store operation";
  let description = [{
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<AnyTypeOf<[AnyInteger, Index]>>:$indices,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return ::llvm::cast<MemRefType>(getMemRef().getType());
      }
  }];

  /* let hasFolder = 1; */
  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `:` type($indices) `]` attr-dict `:` type($memref)
  }];
}

def LoopScheduleDependenciesOp : LoopScheduleOp<"dependencies",
     [NoRegionArguments, SingleBlock, NoTerminator, Symbol]> {
  let summary = "Memory Dependencies operation";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let assemblyFormat = "$sym_name $body attr-dict";
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    // OpAsmOpInterface
    /* static ::llvm::StringRef getDefaultDialect() { return "loopschedule"; } */

    // SymbolOpInterface
    static bool isOptionalSymbol() { return false; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];

  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name), [{
      ::mlir::Region* region = $_state.addRegion();
      region->push_back(new ::mlir::Block());
      $_state.getOrAddProperties<Properties>().sym_name = sym_name;
    }]>
  ];
  /* let hasVerifier = 1; */
}

def LoopScheduleDependsOnOp : LoopScheduleOp<"depends_on",
    [
    HasParent<"LoopScheduleDependenciesOp">,
    NoTerminator
    ]> {
  let summary = "Defines the accesses that an op depends on";
  let description = [{
  }];

  let arguments = (ins StrAttr:$accessName);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$accessName $body attr-dict";

  let extraClassDeclaration = [{
    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];
  let skipDefaultBuilders = true;
  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$accessName), [{
      ::mlir::Region* region = $_state.addRegion();
      region->push_back(new ::mlir::Block());
      $_state.getOrAddProperties<Properties>().accessName = accessName;
    }]>
  ];
  /* let hasVerifier = 1; */
}

def LoopScheduleAccessOp : LoopScheduleOp<"access",
    [
      HasParent<"LoopScheduleDependsOnOp">
    ]> {
  let summary = "Defines an access that another op depends on";
  let description = [{
  }];

  let arguments = (ins StrAttr:$accessName,
                       OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$dist);
  let hasCustomAssemblyFormat = 1;
}

def LoopScheduleYieldOp : LoopScheduleOp<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["LoopScheduleIfOp", "LoopScheduleFrameOp",
                 "LoopScheduleAtOp", "LoopScheduleLaunchOp"]>]> {
  let summary = "Terminator yielding values to the enclosing region-bearing op";
  let description = [{
    Shared terminator op. Used as the terminator of:
      - `loopschedule.if`'s body region (yields the op's results),
      - `loopschedule.frame`'s await region (yields values that feed the
        body region's entry-block args),
      - `loopschedule.frame`'s body region (yields the frame op's results),
      - `loopschedule.at`'s body region (yields the at op's results),
      - `loopschedule.launch`'s body region (yields child loop results,
        though the launch op's only surface result is a handle).

    The verifier dispatches on the parent op + enclosing region to enforce
    the right type-matching rule.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];

  let hasVerifier = 1;
}

def LoopScheduleIfOp : LoopScheduleOp<"if",
       [SingleBlockImplicitTerminator<"LoopScheduleYieldOp">,
        NoRegionArguments]> {
  let summary = "Conditional if operator";

  let arguments = (ins
    I1:$cond
  );

  let results = (outs
    Variadic<AnyType>:$output
  );

  let regions = (region
    SizedRegion<1>:$body
  );
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond)>
  ];

  let hasCustomAssemblyFormat = 1;
}

def LoopScheduleBufferOp : LoopScheduleOp<"buffer", [Pure, SameOperandsAndResultType]> {
  let summary = "Explicitly ensures the schedule delays a value by 1 clock cycle";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  let assemblyFormat =
      [{  $input attr-dict `:` type($output) }];
}

//===----------------------------------------------------------------------===//
// Frames + handles (new sequential-loop surface).
//
// See `LOOPSCHEDULE_FRAMES_HANDLES.md` in the amc-dialect repo root for the
// full design. This slice adds the ops/type so they round-trip; no emitter
// or consumer uses them yet.
//===----------------------------------------------------------------------===//

def LoopScheduleFrameOp : LoopScheduleOp<"frame", [PhaseOpInterface]> {
  let summary = "Statically-timed window inside a sequential loop.";
  let description = [{
    A `loopschedule.frame` op bounds a statically-timed region of work within
    a sequential loop body. It has two regions:

      - **await region** (first, always present, may be empty-bodied):
        contains `loopschedule.await` ops that block the frame's start until
        their handle operands complete. Terminated by `loopschedule.yield`
        whose operands become the body region's entry-block args.
      - **body region** (second, after the `do` keyword): statically timed.
        Contains `loopschedule.at` and `loopschedule.launch` ops. The body's
        latency is the max of (offset + op latency) across its children, and
        is compile-time known. Terminated by `loopschedule.yield` whose
        operands become the frame op's SSA results.

    The assembly syntax mirrors `scf.while`:

        %a, %b = loopschedule.frame -> (i32, i32) {
          // await region
          loopschedule.yield
        } do {
        ^body:
          // body region (statically timed)
          loopschedule.yield %x, %y : i32, i32
        }
  }];

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region
    SizedRegion<1>:$awaitRegion,
    SizedRegion<1>:$bodyRegion
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getAwaitBlock() { return getAwaitRegion().front(); }
    Block &getBodyBlock() { return getBodyRegion().front(); }
    LoopScheduleYieldOp getAwaitYield();
    LoopScheduleYieldOp getBodyYield();

    std::string getRegisterNamePrefix() { return "frame"; }
    bool isStatic() { return true; }
    std::optional<uint64_t> getStartTime() { return std::nullopt; }
  }];
}

def LoopScheduleAtOp : LoopScheduleOp<"at",
    [HasParent<"LoopScheduleFrameOp">,
     SingleBlockImplicitTerminator<"LoopScheduleYieldOp">,
     NoRegionArguments]> {
  let summary = "Offset wrapper inside a frame body.";
  let description = [{
    Marks that the ops in its single-block body execute `offset` cycles
    after the enclosing frame body's start. Values yielded by the region's
    terminator become the `at` op's results, visible to the rest of the
    frame body.

    `loopschedule.at` may appear only inside a frame's body region. The
    verifier enforces this (the body region is always the frame's second
    region).
  }];

  let arguments = (ins I64Attr:$offset);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    LoopScheduleYieldOp getYieldOp();
  }];
}

def LoopScheduleLaunchOp : LoopScheduleOp<"launch",
    [HasParent<"LoopScheduleFrameOp">,
     SingleBlockImplicitTerminator<"LoopScheduleYieldOp">,
     NoRegionArguments]> {
  let summary = "Asynchronously start a variable-latency child loop.";
  let description = [{
    Fires the child loop's `start` at `offset` cycles into the enclosing
    frame body. Returns immediately with a `!loopschedule.handle` SSA value
    representing completion; a later frame's await region picks up that
    handle to block until the child is done.

    The body region contains exactly one child loop op
    (`loopschedule.sequential` or `loopschedule.pipeline`), plus a
    terminating `loopschedule.yield` forwarding the child's produced values.

    `loopschedule.launch` may appear only inside a frame's body region.
  }];

  let arguments = (ins I64Attr:$offset);
  let results = (outs HandleType:$handle);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    LoopScheduleYieldOp getYieldOp();
  }];
}

def LoopScheduleAwaitOp : LoopScheduleOp<"await",
    [HasParent<"LoopScheduleFrameOp">]> {
  let summary = "Barrier on one or more child-completion handles.";
  let description = [{
    Consumes one or more `!loopschedule.handle` operands and (optionally)
    forwards values produced by the launched children. The enclosing frame's
    await region as a whole is a barrier: the frame body does not start
    until every await in the region has completed.

    `loopschedule.await` may appear only inside a frame's await region
    (the first region of a `loopschedule.frame`). The verifier enforces this.
  }];

  let arguments = (ins Variadic<HandleType>:$handles);
  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LoopScheduleIterArgUpdateOp : LoopScheduleOp<"iter_arg_update",
    [ParentOneOf<["LoopScheduleAtOp", "LoopSchedulePipelineStageOp"]>,
     TypesMatchWith<"iter-arg type matches new-value type",
                    "value", "iterArg",
                    "$_self">]> {
  let summary = "Phi-style update of a sequential loop iter-arg.";
  let description = [{
    Writes `value` into the iter-arg `iterArg` at the offset of the enclosing
    `loopschedule.at` region. In the final shape, `iterArg` is expected to be
    an iter-arg block argument of the surrounding `loopschedule.sequential`
    op, and this update must appear in the final frame's body (at most once
    per iter-arg per iteration). This slice only enforces the immediate
    parent-is-an-`at` constraint; tighter checks come with the emitter.
  }];

  let arguments = (ins AnyType:$iterArg, AnyType:$value);

  let assemblyFormat = [{
    $iterArg `=` $value attr-dict `:` type($value)
  }];

  let hasVerifier = 1;
}

#endif // CIRCIT_LOOP_SCHEDULE_OPS_TD
