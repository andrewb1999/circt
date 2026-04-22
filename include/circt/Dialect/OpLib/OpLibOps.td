//===- OpLibOps.td - OpLib Op Definitions ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for OpLib.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OPLIB_OPS_TD
#define CIRCT_DIALECT_OPLIB_OPS_TD

include "circt/Dialect/OpLib/OpLib.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class OpLibOp<string mnemonic, list<Trait> traits = []> :
  Op<OpLib_Dialect, mnemonic, traits>;

def LibraryOp : OpLibOp<"library", [IsolatedFromAbove, SingleBlock,
                                    Symbol, SymbolTable, NoTerminator,
                                    NoRegionArguments,
                                    HasParent<"mlir::ModuleOp">]> {
  let summary = "OpLib dialect Library operation";
  let description = [{
    The `oplib.library` operation represents an operator library to be
    used by passes that lower to one of a few hardware design dialects. 
    A library is made up of a single block and contains only
    `oplib.operator` ops.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name), [{
      Region* region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addAttribute($_builder.getStringAttr("sym_name"), sym_name);
    }]>
  ];

  let extraClassDeclaration = [{
    static bool isOptionalSymbol() { return false; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];

  let hasVerifier = 1;
}

def OperatorOp : OpLibOp<"operator", [IsolatedFromAbove,
                                         SingleBlock, NoTerminator,
                                         NoRegionArguments,
                                         Symbol, SymbolTable,
                                         HasParent<"LibraryOp">]> {
  let summary = "OpLib dialect Operator operation";
  let description = [{
    The `oplib.operator` operation represents an operator in the
    operator library.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, 
                       I32Attr:$latency,
                       OptionalAttr<F64Attr>:$incDelay,
                       OptionalAttr<F64Attr>:$outDelay);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $sym_name `latency` `<` $latency `>` (`,` `incDelay` `<` $incDelay^ `>`)? 
    (`,` `outDelay` `<` $outDelay^ `>`)? attr-dict-with-keyword $body
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name,
                   "mlir::IntegerAttr":$latency), [{
      Region* region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addAttribute($_builder.getStringAttr("sym_name"), sym_name);
      $_state.addAttribute($_builder.getStringAttr("latency"), latency);
    }]>,
    OpBuilder<(ins "mlir::StringAttr":$sym_name,
                   "mlir::IntegerAttr":$latency,
                   "mlir::FloatAttr":$incDelay,
                   "mlir::FloatAttr":$outDelay), [{
      Region* region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addAttribute($_builder.getStringAttr("sym_name"), sym_name);
      $_state.addAttribute($_builder.getStringAttr("latency"), latency);
      $_state.addAttribute($_builder.getStringAttr("incDelay"), incDelay);
      $_state.addAttribute($_builder.getStringAttr("outDelay"), outDelay);
    }]>
  ];

  let extraClassDeclaration = [{
    static bool isOptionalSymbol() { return false; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];

  let hasVerifier = 1;
}

def TargetOp : OpLibOp<"target", [Symbol, 
                                  FunctionOpInterface,
                                  SingleBlockImplicitTerminator<"OutputOp">,
                                  HasParent<"OperatorOp">]> {
  let summary = "OpLib dialect Target operation";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$symName,
                   "mlir::FunctionType":$functionType,
                   CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs,
                   CArg<"ArrayRef<DictionaryAttr>", "{}">:$resAttrs)>
  ];

  let extraClassDeclaration = [{
    static bool isOptionalSymbol() { return false; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() {
      return &getBodyRegion();
    }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!llvm::isa<FunctionType>(type))
        return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                           "' attribute of function type");
      return success();
    }
  }];
  // let assemblyFormat = [{
  //   $sym_name custom<FunctionType>($function_type) attr-dict-with-keyword $body
  // }];
  let hasVerifier = 1;
}

def OperationOp : OpLibOp<"operation", [HasParent<"TargetOp">]> {
  let summary = "OpLib dialect Operation operation";
  let description = [{
    Placeholder for the logical operator this target describes. `opName`
    is the fully-qualified MLIR op name (e.g. `"arith.muli"`), so it
    round-trips as a single quoted string.
  }];
  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$opName,
                       OptionalAttr<DictionaryAttr>:$opDict);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $opName (`with` $opDict^)? `(` ($operands^ `:` qualified(type($operands)))? `)`
    attr-dict (`:` qualified(type($results))^)?
  }];

  let extraClassDeclaration = [{
    /// Split `opName` on the first `.` separator. Returns the pair
    /// (dialect, op) — e.g. `"arith.muli"` -> (`"arith"`, `"muli"`).
    std::pair<::llvm::StringRef, ::llvm::StringRef> getDialectAndOp() {
      return getOpName().split('.');
    }
  }];
}

def CalyxMatchOp : OpLibOp<"calyx_match",
    [SymbolTable,
     HasParent<"OperatorOp">,
     NoRegionArguments,
     SingleBlock,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "OpLib dialect Target operation";
  let description = [{
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
                       TypeAttrOf<FunctionType>:$targetType);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::FlatSymbolRefAttr":$target,
                   "mlir::TypeAttr":$targetType), [{
      Region* region = $_state.addRegion();
      region->push_back(new Block());
      $_state.addAttribute($_builder.getStringAttr("target"), target);
      $_state.addAttribute($_builder.getStringAttr("targetType"), targetType);
    }]>
  ];

  let extraClassDeclaration = [{
    static bool isOptionalSymbol() { return false; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];

  let assemblyFormat = [{
    `(` $target `:` $targetType `)` `produce` attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def HwMatchOp : OpLibOp<"hw_match",
    [HasParent<"OperatorOp">,
     SingleBlock,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "OpLib dialect HW match operation";
  let description = [{
    The `oplib.hw_match` op describes how a logical operator lowers to
    an HW-dialect primitive (typically an `hw.instance` of an
    `hw.module.extern`). The body block takes named arguments for every
    extern input port (clk, reset, ce, then each operand in declaration
    order) so the body can instantiate the primitive using real SSA
    values rather than placeholder constants. The terminator is
    `oplib.hw_return`, which yields the output values in target-function
    result order.

    Each body block argument carries a role from `argRoles`: `"clk"`,
    `"reset"`, `"ce"`, or `"in"`. The custom printer uses these to label
    each argument (e.g. `clk %arg0: i1, in %arg3: i32`).
  }];

  let arguments = (ins FlatSymbolRefAttr:$target,
                       TypeAttrOf<FunctionType>:$targetType,
                       StrArrayAttr:$argRoles);

  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::FlatSymbolRefAttr":$target,
                   "mlir::TypeAttr":$targetType,
                   "mlir::TypeRange":$blockArgTypes,
                   "mlir::ArrayAttr":$argRoles), [{
      Region* region = $_state.addRegion();
      Block *body = new Block();
      region->push_back(body);
      ::llvm::SmallVector<::mlir::Location, 4> locs(
          blockArgTypes.size(), $_builder.getUnknownLoc());
      body->addArguments(blockArgTypes, locs);
      $_state.addAttribute($_builder.getStringAttr("target"), target);
      $_state.addAttribute($_builder.getStringAttr("targetType"), targetType);
      $_state.addAttribute($_builder.getStringAttr("argRoles"), argRoles);
    }]>
  ];

  let extraClassDeclaration = [{
    static bool isOptionalSymbol() { return false; }

    // Convenience
    ::mlir::Block *getBodyBlock() {
      return &getBody().getBlocks().front();
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def HwInstanceOp : OpLibOp<"hw_instance",
    [ParentOneOf<["HwMatchOp"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Instantiate an `hw.module.extern` inside a hw_match body";
  let description = [{
    Same shape as `hw.instance` — operand inputs + typed results,
    argNames / resultNames metadata, symbol reference to an
    `hw.module.extern`. Its operands are real SSA values (the enclosing
    `oplib.hw_match` body's block arguments), not placeholder constants.

    The symbol verifier resolves the extern from the enclosing `ModuleOp`
    (like `calyx.primitive` does), so the op can live inside
    `oplib.operator`'s `SymbolTable` even though a real `hw.instance`
    cannot.
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs,
                       StrArrayAttr:$argNames,
                       StrArrayAttr:$resultNames);
  let results = (outs Variadic<AnyType>:$results);

  let hasCustomAssemblyFormat = 1;
}

def HwReturnOp : OpLibOp<"hw_return", [Terminator, ReturnLike,
                                          ParentOneOf<["HwMatchOp"]>]> {
  let summary = "Terminator for hw_match bodies";
  let description = [{
    Yields the hw_match's output values. The operand types must match
    the target function's result types.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
}

def OutputOp : OpLibOp<"output", [Terminator,
                                ReturnLike, ParentOneOf<["TargetOp"]>]> {
  let summary = "OpLib dialect Output operation";
  let description = [{
    The `oplib.output` operation.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    attr-dict ($outputs^ `:` qualified(type($outputs)))?
  }];
}

def YieldOp : OpLibOp<"yield", [Terminator,
                                   ReturnLike,
                                   AttrSizedOperandSegments,
                                   ParentOneOf<["CalyxMatchOp"]>]> {
  let summary = "OpLib dialect Yield operation";
  let description = [{
    The `oplib.output` operation.
  }];
  let arguments = (ins Optional<AnyType>:$clock,
                       Optional<AnyType>:$clockEnable,
                       Optional<AnyType>:$reset,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs, "ValueRange":$outputs), [{
      build($_builder, $_state, Value(), Value(), Value(), inputs, outputs);
    }]>
  ];
  let assemblyFormat = [{
    attr-dict (`clk` `(` $clock^ `:` qualified(type($clock)) `)` `,` )? 
              (`ce` `(` $clockEnable^ `:` qualified(type($clockEnable)) `)` `,` )?
              (`reset` `(` $reset^ `:` qualified(type($reset)) `)` `,` )?
              (`ins` `(` $inputs^ `:` qualified(type($inputs)) `)` `,` )?
              (`outs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
  }];
}

#endif // CIRCT_DIALECT_OPLIB_OPS_TD
