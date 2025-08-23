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
    contains two regions: `condition` and `stages`.

    The pipeline may accept an optional `inits`, similar to the SCF dialect,
    for representing loop-carried values like induction variables or reductions.
    When the pipeline starts execution, the registers indicated as `inits`
    by `loopschedule.terminator` should be initialized to the initial
    values specified in the `inits` section here. The `inits` relate to
    the initiation interval of the loop. The maximum distance in stages between
    where an `init` is used and where that `init` is registered must be
    less than the loop's initiation interval. For example, with II=1, each
    `init` must be used and registered in the same stage.

    The single-block `condition` region dictates the condition under which the
    pipeline should execute. It has a `register` terminator, and the
    pipeline initiates new iterations while the registered value is `true : i1`.
    It may access SSA values dominating the pipeline, as well as `inits`,
    which are block arguments. The body of the block may only contain
    "combinational" operations, which are currently defined to be simple
    arithmetic, comparisons, and selects from the `Standard` dialect.

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
    SizedRegion<1>:$condition,
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
    Block &getCondBlock() { return getCondition().front(); }
    Block &getStagesBlock() { return getStages().front(); }

    Block::BlockArgListType getBodyArgs() {
      return getStagesBlock().getArguments();
    }
    Block *getBodyBlock() {
      return &getStagesBlock();
    }
    Block *getConditionBlock() {
      return &getCondBlock();
    }
    Value getConditionValue() {
      return getCondBlock().getTerminator()->getOperand(0);
    }
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
    loop contains two regions: `condition` and `stages`.

    The sequential loop may accept an optional `inits`, similar to the SCF 
    dialect, for representing loop-carried values like induction variables or 
    reductions. When the sequential loop starts execution, the registers 
    indicated as `inits` by `loopschedule.terminator` should be initialized 
    to the initial values specified in the `inits` section here.

    The single-block `condition` region dictates the condition under which the
    sequential loop should execute. It has a `register` terminator, and the
    loop initiates new iterations while the registered value is `true : i1`.
    It may access SSA values dominating the loop, as well as `inits`,
    which are block arguments.

    The single-block `steps` region wraps `loopschedule.step`
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
    SizedRegion<1>:$condition,
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
    Block &getCondBlock() { return getCondition().front(); }
    Block &getScheduleBlock() { return getSchedule().front(); }
    Block::BlockArgListType getBodyArgs() {
      return getScheduleBlock().getArguments();
    }
    Block *getBodyBlock() {
      return &getScheduleBlock();
    }
    Block *getConditionBlock() {
      return &getCondBlock();
    }
    Value getConditionValue() {
      return getCondBlock().getTerminator()->getOperand(0);
    }
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

def LoopScheduleRegisterOp : LoopScheduleOp<"register",
    [ParentOneOf<["LoopSchedulePipelineOp", "LoopSchedulePipelineStageOp", "LoopScheduleSequentialOp", "LoopScheduleStepOp"]>, Terminator]> {
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
    a `loopschedule.pipeline`.

    The `results` section accepts a variadic list of values which become the
    pipeline’s return values. These must be results of a stage, and their types
    must match the pipeline's return types. The results need not be defined in
    the final stage, and it is up to lowering passes to preserve these values
    until the final stage is complete.

    The `iter_args` section accepts a variadic list of values which become the
    next iteration’s `iter_args`. These may be the results of any stage, and
    their types must match the pipeline's `iter_args` types.
  }];

  let arguments = (ins
    Variadic<AnyType>:$iter_args,
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    `iter_args` `(` $iter_args `)` `,`
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
    HasParent<"LoopScheduleIfOp">]> {
  let summary = "loop schedule if op yield";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
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

#endif // CIRCIT_LOOP_SCHEDULE_OPS_TD
