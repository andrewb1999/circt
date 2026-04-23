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
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "circt/Dialect/LoopSchedule/LoopSchedule.td"
include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.td"
include "circt/Dialect/LoopSchedule/LoopScheduleLoweringInterfaces.td"

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
    std::optional<uint64_t> getBound() {
      return getTripCount();
    }
    bool isPipelined() {
      return true;
    }

    bool canStall();
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
  let hasCanonicalizer = 1;

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
    std::optional<uint64_t> getBound() {
      return getTripCount();
    }
    bool isPipelined() {
      return false;
    }
    bool canStall();
  }];
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

    The optional `await` section accepts a variadic list of
    `!loopschedule.handle` values. Before the terminator fires its transition,
    every listed handle must have completed. This is a shorthand for an
    iteration-boundary barrier on launched children whose completion value is
    not otherwise consumed in the body.
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$results,
    Variadic<HandleType>:$await
  );

  let assemblyFormat = [{
    `condition` `(` $condition `)` `,`
    (`await` `(` $await^ `)` `,`)?
    `results` `(` $results `)`
    (`:` type($results)^)? attr-dict
  }];

  let hasVerifier = 1;
}

def LoopScheduleLoadOp : LoopScheduleOp<"load",
     [TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "::llvm::cast<MemRefType>($_self).getElementType()">,
      DeclareOpInterfaceMethods<HWLoadLoweringInterface,
          ["getMemoryValue", "getAddrWidths", "getReadLatency",
           "requiresReadEnable"]>
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
                     "::llvm::cast<MemRefType>($_self).getElementType()">,
      DeclareOpInterfaceMethods<HWStoreLoweringInterface,
          ["getMemoryValue", "getValueToStore", "getAddrWidths",
           "getWriteLatency"]>
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
    [ParentOneOf<["LoopScheduleFrameOp", "LoopSchedulePipelineOp",
                  "LoopScheduleFuncPipelineOp"]>,
     PhaseOpInterface,
     SingleBlockImplicitTerminator<"LoopScheduleYieldOp">,
     NoRegionArguments]> {
  let summary = "Offset wrapper for a phase of work in a frame or pipeline.";
  let description = [{
    Marks that the ops in its single-block body execute `offset` cycles
    after the enclosing parent's start. Values yielded by the region's
    terminator become the `at` op's results, visible to later phases.

    `loopschedule.at` may appear:
      - inside a `loopschedule.frame`'s body region (statically-timed work
        within a sequential loop), or
      - directly inside a `loopschedule.pipeline`'s stages region (a
        pipeline phase, where `offset` is the absolute cycle within the
        pipeline body).
  }];

  let arguments = (ins I64Attr:$offset);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$offset)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    LoopScheduleYieldOp getYieldOp();

    // PhaseOpInterface
    bool isStatic() { return true; }
    std::optional<uint64_t> getStartTime() { return getOffset(); }
    std::string getRegisterNamePrefix();
    unsigned getStageNumber();
  }];
}

def LoopScheduleLaunchOp : LoopScheduleOp<"launch",
    [HasParent<"LoopScheduleAtOp">,
     SingleBlockImplicitTerminator<"LoopScheduleYieldOp">,
     NoRegionArguments]> {
  let summary = "Issue a variable-latency op or child loop.";
  let description = [{
    Fires when the enclosing `loopschedule.at` is entered (at the at's
    offset relative to the surrounding frame or pipeline). Returns
    immediately with a `!loopschedule.handle` SSA value representing
    completion; a later frame's `loopschedule.await`, or — for in-pipeline
    dynamic-latency ops — a later-stage `loopschedule.expect`, picks up
    that handle to block until the launched work is done.

    Two forms of launch body:
      - Frame-level: the body contains exactly one child loop op
        (`loopschedule.sequential` or `loopschedule.pipeline`), plus a
        terminating `loopschedule.yield` forwarding the child's values.
        The handle is consumed by `loopschedule.await` in a later frame.
      - Pipeline-level: the body contains a dynamic-latency op (e.g. an
        `amc.load` on a dynamic port), plus a terminating
        `loopschedule.yield` forwarding that op's results. The handle is
        consumed by `loopschedule.expect` in a later `loopschedule.at`
        stage of the same pipeline — the stage scheduled exactly
        `latency(op)` cycles after the launch's at-stage.

    `loopschedule.launch` may appear only inside a `loopschedule.at` body.
    Multiple launches may share an `at`; they all start at the at's offset.
  }];

  let results = (outs HandleType:$handle);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    LoopScheduleYieldOp getYieldOp();
  }];
}

def LoopScheduleExpectOp : LoopScheduleOp<"expect",
    [HasParent<"LoopScheduleAtOp">]> {
  let summary = "Consume a pipeline-scoped launch handle at the cycle its "
                "result is expected to be ready.";
  let description = [{
    Marks the cycle (the enclosing `loopschedule.at`'s offset) at which the
    pipeline expects the handle's underlying dynamic-latency op to have
    completed. The op takes a `!loopschedule.handle` produced by a
    `loopschedule.launch` in an earlier stage of the *same* pipeline, and
    yields SSA values matching the launch's body-level
    `loopschedule.yield` types.

    Consumers in the expect's stage use the expect's results directly;
    later stages pick them up through the pipeline's normal stage-register
    chain. Downstream FSM lowering uses the expect's stage and handle to
    gate pipeline stage-enables on the launch's completion signal — if the
    signal is not asserted by the expect's cycle, the whole pipeline
    stalls until it is.

    The expect's stage must sit exactly `latency(op)` cycles after the
    launch's stage, where `latency(op)` is the nominal latency the
    scheduler assigned to the wrapped op. (Enforced structurally by the
    scheduler; not checked here.)
  }];

  let arguments = (ins HandleType:$handle);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $handle attr-dict (`:` type($results)^)?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// The `loopschedule.launch` that produced the consumed handle, or
    /// null if the handle is not produced by a launch in the same
    /// pipeline.
    LoopScheduleLaunchOp getLaunchOp();
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
    [HasParent<"LoopScheduleAtOp">,
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

//===----------------------------------------------------------------------===//
// Func-level container ops
//
// `loopschedule.func_sequential` and `loopschedule.func_pipeline` replace
// `func.func` as the top-level container after SCFToLoopSchedule. Each
// holds a single body region whose contents are constrained to be a valid
// schedule (allocations + init ops first, then frames or pipeline stages,
// then a `loopschedule.return`).
//
// Both ops carry FunctionOpInterface so external passes that walk function-
// like ops (symbol resolution, call resolution, walk-the-module utilities)
// continue to work. They are SymbolOp + Callable so `func.call @foo` can
// still target them.
//===----------------------------------------------------------------------===//

class LoopScheduleFuncOpBase<string mnemonic, list<Trait> traits = []> :
    LoopScheduleOp<mnemonic, !listconcat(traits, [
      AutomaticAllocationScope,
      FunctionOpInterface,
      IsolatedFromAbove,
      Symbol,
      CallableOpInterface,
      OpAsmOpInterface
    ])> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface
    //===------------------------------------------------------------------===//

    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    ::llvm::ArrayRef<::mlir::Type> getCallableResults() {
      return getFunctionType().getResults();
    }
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }
  }];
}

def LoopScheduleFuncSequentialOp
    : LoopScheduleFuncOpBase<"func_sequential"> {
  let summary = "LoopSchedule sequential function-level container.";
  let description = [{
    A function whose body is a sequential schedule: allocations + init ops,
    followed by zero or more `loopschedule.frame` ops in monotonic phase
    order, terminated by `loopschedule.return`. Replaces `func.func` as the
    top-level container after SCFToLoopSchedule for the non-pipelined case.

    No iter-args, no II — execution semantics match the existing
    func-level lowering: frames execute in order and the function returns
    once the last frame completes.
  }];

  let builders = [
    OpBuilder<(ins
      "::mlir::StringRef":$name,
      "::mlir::FunctionType":$type,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
      CArg<"::llvm::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)>
  ];
}

def LoopScheduleFuncPipelineOp
    : LoopScheduleFuncOpBase<"func_pipeline"> {
  let summary = "LoopSchedule pipelined function-level container.";
  let description = [{
    A function whose body is a pipelined schedule: allocations + init ops,
    followed by `loopschedule.at` stages at increasing offsets, terminated
    by `loopschedule.return`. New invocations may be issued every `II`
    cycles. Replaces `func.func` as the top-level container after
    SCFToLoopSchedule for the pipelined case.

    No iter-args (function arguments are fresh per transaction). The
    scheduling problem is a `ChainingModuloProblem` so resource sharing
    across overlapping in-flight transactions is bounded by II, even when
    the dependence graph itself has no cyclic edges.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    I64Attr:$II,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let builders = [
    OpBuilder<(ins
      "::mlir::StringRef":$name,
      "::mlir::FunctionType":$type,
      "::mlir::IntegerAttr":$II,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
      CArg<"::llvm::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)>
  ];
}

def LoopScheduleReturnOp : LoopScheduleOp<"return",
    [Pure, ReturnLike, Terminator,
     ParentOneOf<["LoopScheduleFuncSequentialOp",
                  "LoopScheduleFuncPipelineOp"]>]> {
  let summary = "Return from a loopschedule func-level container.";
  let description = [{
    Terminator for `loopschedule.func_sequential` and
    `loopschedule.func_pipeline` bodies. Operand types must match the
    parent op's result types.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];

  let hasVerifier = 1;
}

#endif // CIRCIT_LOOP_SCHEDULE_OPS_TD
