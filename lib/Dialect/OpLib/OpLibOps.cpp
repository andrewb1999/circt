//===- OpLibOps.cpp - OpLib Dialect Operations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the OpLib ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OpLib/OpLibOps.h"
#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/OpLib/OpLibAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/Support/LogicalResult.h"

#include "circt/Dialect/OpLib/OpLibInterfaces.cpp.inc"

using namespace mlir;
using namespace circt;
using namespace circt::oplib;

//===----------------------------------------------------------------------===//
// LibraryOp
//===----------------------------------------------------------------------===//

LogicalResult LibraryOp::verify() {
  for (auto &op : this->getBodyRegion().getOps()) {
    if (!isa<LibraryEntryOpInterface>(op))
      return emitOpError("can only contain library entries "
                         "(LibraryEntryOpInterface), not ")
             << op.getName();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// OperatorOp
//===----------------------------------------------------------------------===//

unsigned OperatorOp::getEntryLatency() { return getLatency(); }

std::optional<double> OperatorOp::getIncomingDelay() {
  if (auto delay = getIncDelay())
    return delay->convertToDouble();
  return std::nullopt;
}

std::optional<double> OperatorOp::getOutgoingDelay() {
  if (auto delay = getOutDelay())
    return delay->convertToDouble();
  return std::nullopt;
}

std::optional<unsigned> OperatorOp::getEntryLimit() {
  if (auto limit = getLimit())
    return static_cast<unsigned>(*limit);
  return std::nullopt;
}

/// The `oplib.operation` an operator's target matches ops against, if it
/// has one.
static OperationOp patternOf(OperatorOp op) {
  auto targets = op.getBodyBlock()->getOps<TargetOp>();
  if (targets.empty() || (*targets.begin()).getBodyBlock()->empty())
    return {};
  return dyn_cast<OperationOp>((*targets.begin()).getBodyBlock()->front());
}

std::optional<StringRef> OperatorOp::getMatchedOperationName() {
  if (auto pattern = patternOf(*this))
    return pattern.getOpName();
  return std::nullopt;
}

/// An op matches an operator's pattern when it is the same operation, with
/// the pattern's attributes and its operand and result types. An operator
/// without a pattern matches whatever is assigned to it.
bool OperatorOp::matches(Operation *op) {
  OperationOp pattern = patternOf(*this);
  if (!pattern)
    return true;
  if (op->getName().getStringRef() != pattern.getOpName())
    return false;
  if (auto dict = pattern.getOpDict())
    for (NamedAttribute attr : *dict)
      if (op->getAttr(attr.getName()) != attr.getValue())
        return false;
  return TypeRange(op->getOperandTypes()) ==
             TypeRange(pattern->getOperandTypes()) &&
         TypeRange(op->getResultTypes()) ==
             TypeRange(pattern->getResultTypes());
}

Operation *OperatorOp::getHwMatch() {
  auto matches = getBodyBlock()->getOps<HwMatchOp>();
  return matches.empty() ? nullptr : (*matches.begin()).getOperation();
}

LogicalResult OperatorOp::verify() {
  unsigned numMatches = 0;
  unsigned numTargets = 0;
  for (auto &op : getBodyBlock()->getOperations()) {
    if (isa<HwMatchOp>(op)) {
      ++numMatches;
      continue;
    }
    if (isa<TargetOp>(op)) {
      ++numTargets;
      continue;
    }
    // Allow hoisted constants. The OperatorOp body is IsolatedFromAbove,
    // so constant-folding canonicalization patterns may legitimately
    // hoist constant placeholders out of the match-op bodies; rejecting
    // them here breaks any downstream pass pipeline that includes
    // canonicalize after operator-allocation.
    if (op.hasTrait<OpTrait::ConstantLike>())
      continue;
    return op.emitOpError(
        "operator body may only contain target ops, match ops, and hoisted "
        "constants");
  }
  // An operator that declares a `TargetOp` is meant to drive codegen and
  // must supply at least one matcher. An operator with no targets serves
  // purely as a container for properties (e.g. `limit`) consumed by
  // downstream analyses — memory operators emitted by `OperatorAllocation`
  // are the canonical example — and does not need a match op.
  if (numTargets > 0 && numMatches == 0)
    return emitOpError("must contain at least one match op");

  if (getIncDelay().has_value() != getOutDelay().has_value()) {
    return emitOpError(
        "must have either both incDelay and outDelay or neither");
  }

  if (getLatency() == 0 &&
      (getIncDelay().has_value() || getOutDelay().has_value())) {
    if (getIncDelay() != getOutDelay()) {
      return emitError(
          "incDelay and outDelay of combinational operators must be the same");
    }
  }

  return success();
}

ParseResult OperatorOp::parse(OpAsmParser &parser, OperationState &result) {
  // `@sym_name latency<N>[, incDelay<F>][, outDelay<F>][, limit<N>]
  //  [attributes {...}] { body }`
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr,
                              mlir::SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return failure();

  // Required `latency<N>`.
  IntegerAttr latencyAttr;
  if (parser.parseKeyword("latency") || parser.parseLess() ||
      parser.parseAttribute(latencyAttr, parser.getBuilder().getIntegerType(32),
                             "latency", result.attributes) ||
      parser.parseGreater())
    return failure();

  // Optional `, keyword<value>` terms, in any order. We scan them in a
  // loop so the user isn't bound to a particular emission order.
  while (succeeded(parser.parseOptionalComma())) {
    StringRef kw;
    if (parser.parseKeyword(&kw))
      return failure();
    if (parser.parseLess())
      return failure();
    if (kw == "incDelay") {
      FloatAttr fa;
      if (parser.parseAttribute(fa, parser.getBuilder().getF64Type(),
                                 "incDelay", result.attributes) ||
          parser.parseGreater())
        return failure();
    } else if (kw == "outDelay") {
      FloatAttr fa;
      if (parser.parseAttribute(fa, parser.getBuilder().getF64Type(),
                                 "outDelay", result.attributes) ||
          parser.parseGreater())
        return failure();
    } else if (kw == "limit") {
      IntegerAttr ia;
      if (parser.parseAttribute(ia, parser.getBuilder().getIntegerType(64),
                                 "limit", result.attributes) ||
          parser.parseGreater())
        return failure();
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "expected one of 'incDelay', 'outDelay', 'limit'; got '" << kw
             << "'";
    }
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();
  // SizedRegion<1> requires exactly one block. An empty `{}` parses as a
  // zero-block region; materialize an empty block so the verifier passes
  // (memory-only operators emitted by OperatorAllocation need this).
  if (body->empty())
    body->emplaceBlock();
  return success();
}

void OperatorOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << " latency<" << getLatency() << ">";
  if (auto v = getIncDelay())
    p << ", incDelay<" << *v << ">";
  if (auto v = getOutDelay())
    p << ", outDelay<" << *v << ">";
  if (auto v = getLimit())
    p << ", limit<" << *v << ">";
  SmallVector<StringRef> elided{
      mlir::SymbolTable::getSymbolAttrName(),
      getLatencyAttrName().getValue(),
      getIncDelayAttrName().getValue(),
      getOutDelayAttrName().getValue(),
      getLimitAttrName().getValue(),
  };
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elided);
  p << ' ';
  p.printRegion(getBody());
}

//===----------------------------------------------------------------------===//
// TargetOp
//===----------------------------------------------------------------------===//

LogicalResult TargetOp::verify() {
  if (!getBodyBlock()->mightHaveTerminator()) {
    return emitOpError("body block does not have terminator");
  }

  auto *term = getBodyRegion().front().getTerminator();
  if (!isa<oplib::OutputOp>(term)) {
    return emitOpError("region terminator must be OutputOp");
  }

  auto outputOp = cast<oplib::OutputOp>(term);

  if (outputOp.getOutputs().size() != getNumResults()) {
    return emitError("OutputOp must have the same number of arguments as "
                     "results returned by TargetOp");
  }

  for (auto [t1, t2] :
       llvm::zip(outputOp.getOperandTypes(), getResultTypes())) {
    if (t1 != t2) {
      return emitError(
          "OutputOp return type does not match TargetOp result type");
    }
  }

  return success();
}

ParseResult TargetOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void TargetOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

void TargetOp::build(OpBuilder &builder, OperationState &state,
                     StringAttr symName, FunctionType functionType,
                     ArrayRef<DictionaryAttr> argAttrs,
                     ArrayRef<DictionaryAttr> resAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), symName);
  state.addAttribute(TargetOp::getFunctionTypeAttrName(state.name),
                     TypeAttr::get(functionType));

  Region *region = state.addRegion();
  Block *body = new Block();
  region->push_back(body);
  body->addArguments(functionType.getInputs(),
                     SmallVector<Location, 4>(functionType.getNumInputs(),
                                              builder.getUnknownLoc()));

  if (argAttrs.empty() && resAttrs.empty())
    return;
  assert(functionType.getNumInputs() == argAttrs.size());
  assert(functionType.getNumResults() == resAttrs.size());
  call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, resAttrs,
      TargetOp::getArgAttrsAttrName(state.name),
      TargetOp::getResAttrsAttrName(state.name));
}

//===----------------------------------------------------------------------===//
// HwMatchOp helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyMatchSymbolUse(Operation *op, StringRef targetName,
                                          FunctionType type,
                                          SymbolTableCollection &symbolTable) {
  auto fn = symbolTable.lookupSymbolIn<TargetOp>(
      op->getParentOfType<OperatorOp>(),
      StringAttr::get(op->getContext(), targetName));
  if (!fn)
    return op->emitOpError()
           << "reference to undefined target '" << targetName << "'";

  if (fn.getFunctionType() != type)
    return op->emitOpError("reference to target with mismatched type");

  return success();
}

//===----------------------------------------------------------------------===//
// HwMatchOp
//===----------------------------------------------------------------------===//

LogicalResult
HwMatchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMatchSymbolUse(*this, getTarget(), getTargetType(), symbolTable);
}

// Format:
//   `(` $target `:` $targetType `)` `produce`
//   `(` ( role `%name` `:` type (`,` ...)* )? `)`
//   attr-dict-with-keyword $body
//
// Valid roles: `clk`, `reset`, `ce`, `in`, `attr`. They're stored in
// `argRoles` as a StrArrayAttr in block-arg order, and labelled block args
// are printed in the same order.
static bool isHwMatchArgRole(StringRef role) {
  return role == "clk" || role == "reset" || role == "ce" || role == "in" ||
         role == "attr";
}

ParseResult HwMatchOp::parse(OpAsmParser &parser, OperationState &result) {
  FlatSymbolRefAttr targetAttr;
  TypeAttr targetTypeAttr;
  if (parser.parseLParen() ||
      parser.parseAttribute(targetAttr, "target", result.attributes) ||
      parser.parseColon() ||
      parser.parseAttribute(targetTypeAttr, "targetType", result.attributes) ||
      parser.parseRParen() || parser.parseKeyword("produce"))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Attribute> roles;
  auto parseArg = [&]() -> ParseResult {
    StringRef role;
    if (parser.parseKeyword(&role))
      return failure();
    if (!isHwMatchArgRole(role))
      return parser.emitError(parser.getCurrentLocation())
             << "expected role keyword 'clk', 'reset', 'ce', 'in', or 'attr'";
    roles.push_back(
        StringAttr::get(parser.getContext(), role));
    OpAsmParser::Argument arg;
    if (parser.parseArgument(arg, /*allowType=*/false, /*allowAttrs=*/false) ||
        parser.parseColon() || parser.parseType(arg.type))
      return failure();
    args.push_back(arg);
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseArg))
    return failure();
  result.addAttribute("argRoles",
                      ArrayAttr::get(parser.getContext(), roles));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, /*enableNameShadowing=*/false))
    return failure();
  if (body->empty())
    body->emplaceBlock();
  return success();
}

void HwMatchOp::print(OpAsmPrinter &p) {
  p << '(' << getTargetAttr() << " : " << getTargetType() << ") produce (";
  Block *body = getBodyBlock();
  auto roles = getArgRoles();
  llvm::interleaveComma(
      llvm::zip(body->getArguments(), roles), p,
      [&](auto pair) {
        auto [arg, roleAttr] = pair;
        p << cast<StringAttr>(roleAttr).getValue() << ' ';
        p.printOperand(arg);
        p << " : " << arg.getType();
      });
  p << ')';
  p.printOptionalAttrDictWithKeyword(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{"target", "targetType", "argRoles"});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult HwMatchOp::verify() {
  Block *body = getBodyBlock();
  auto roles = getArgRoles();
  if (roles.size() != body->getNumArguments())
    return emitOpError("argRoles size (")
           << roles.size() << ") must match block-arg count ("
           << body->getNumArguments() << ")";
  for (auto roleAttr : roles) {
    auto s = dyn_cast<StringAttr>(roleAttr);
    if (!s)
      return emitOpError("argRoles entries must be strings");
    auto v = s.getValue();
    if (!isHwMatchArgRole(v))
      return emitOpError("argRoles entry '")
             << v << "' must be 'clk', 'reset', 'ce', 'in', or 'attr'";
  }

  // The `in`-role block args, in declaration order, must match the target
  // function's input types one-for-one (bitwidth comparison, consistent
  // with how hw_return outputs are checked against target results).
  // Clock / reset / enable args are ignored — they carry HW-level
  // plumbing that sits outside the logical operator signature — and so
  // are `attr` args, constants taken from one of the op's attributes.
  auto targetType = getTargetType();
  SmallVector<Type, 4> inArgTypes;
  for (auto [roleAttr, arg] :
       llvm::zip(roles, body->getArguments())) {
    if (cast<StringAttr>(roleAttr).getValue() == "in")
      inArgTypes.push_back(arg.getType());
  }
  if (inArgTypes.size() != targetType.getNumInputs())
    return emitOpError("body has ")
           << inArgTypes.size() << " `in`-role block arg(s), target expects "
           << targetType.getNumInputs();
  for (auto [i, t] : llvm::enumerate(inArgTypes)) {
    if (t.getIntOrFloatBitWidth() !=
        targetType.getInput(i).getIntOrFloatBitWidth())
      return emitOpError("body `in`-role block arg #")
             << i << " bitwidth does not match target input type";
  }

  if (!body->mightHaveTerminator())
    return emitOpError("must be terminated by an `oplib.hw_return`");
  auto retOp = dyn_cast<oplib::HwReturnOp>(body->getTerminator());
  if (!retOp)
    return emitOpError("must be terminated by an `oplib.hw_return`");

  if (retOp.getOutputs().size() != targetType.getNumResults())
    return emitOpError("hw_return yields ")
           << retOp.getOutputs().size() << " value(s), target expects "
           << targetType.getNumResults();
  for (auto [i, t] : llvm::enumerate(retOp.getOutputs().getTypes())) {
    if (t.getIntOrFloatBitWidth() !=
        targetType.getResult(i).getIntOrFloatBitWidth())
      return emitOpError("hw_return output #")
             << i << " bitwidth does not match target result type";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// HwInstanceOp
//===----------------------------------------------------------------------===//

// Same assembly form as `hw.instance`:
//   "name" @module(arg: %val: type, ...) -> (res: type, ...)
ParseResult HwInstanceOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr instanceNameAttr;
  FlatSymbolRefAttr moduleNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<Type, 1> inputsTypes, allResultTypes;
  ArrayAttr argNames, resultNames;
  auto noneType = parser.getBuilder().getType<NoneType>();

  if (parser.parseAttribute(instanceNameAttr, noneType, "instanceName",
                            result.attributes) ||
      parser.parseAttribute(moduleNameAttr, noneType, "moduleName",
                            result.attributes))
    return failure();

  llvm::SMLoc inputsLoc = parser.getCurrentLocation();
  if (circt::parseInputPortList(parser, inputsOperands, inputsTypes, argNames) ||
      parser.resolveOperands(inputsOperands, inputsTypes, inputsLoc,
                             result.operands) ||
      parser.parseArrow() ||
      circt::parseOutputPortList(parser, allResultTypes, resultNames) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addAttribute("argNames", argNames);
  result.addAttribute("resultNames", resultNames);
  result.addTypes(allResultTypes);
  return success();
}

void HwInstanceOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printAttributeWithoutType(getInstanceNameAttr());
  p << ' ';
  p.printAttributeWithoutType(getModuleNameAttr());
  circt::printInputPortList(p, *this, getInputs(), getInputs().getTypes(),
                         getArgNames());
  p << " -> ";
  circt::printOutputPortList(p, *this, getResultTypes(), getResultNames());
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{"instanceName", "moduleName", "argNames",
                       "resultNames"});
}

LogicalResult
HwInstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Resolve the extern from the enclosing ModuleOp, bypassing the
  // oplib.operator / oplib.library SymbolTable boundaries.
  auto moduleOp = (*this)->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return emitOpError("must be nested in a ModuleOp");
  auto *referenced =
      symbolTable.lookupSymbolIn(moduleOp, getModuleNameAttr());
  if (!referenced)
    return emitOpError("references unknown extern module '")
           << getModuleName() << "'";
  // Usually an extern; a module once the operator's implementation has been
  // built (e.g. a wrapper around a vendor IP core).
  auto externOp = dyn_cast<hw::HWModuleLike>(referenced);
  if (!externOp)
    return emitOpError("referenced symbol '")
           << getModuleName() << "' is not an hw.module or hw.module.extern";

  // Shape-check operands and results against the referenced extern:
  // counts, per-port types, AND per-port names must all match exactly.
  SmallVector<Type> inputPortTypes, outputPortTypes;
  SmallVector<StringAttr> inputPortNames, outputPortNames;
  for (auto port : externOp.getPortList()) {
    if (port.dir == hw::ModulePort::Direction::Input) {
      inputPortTypes.push_back(port.type);
      inputPortNames.push_back(port.name);
    } else {
      outputPortTypes.push_back(port.type);
      outputPortNames.push_back(port.name);
    }
  }
  if (getInputs().size() != inputPortTypes.size())
    return emitOpError("has ")
           << getInputs().size() << " input(s) but extern has "
           << inputPortTypes.size();
  if (getResults().size() != outputPortTypes.size())
    return emitOpError("has ")
           << getResults().size() << " result(s) but extern has "
           << outputPortTypes.size();
  if (getArgNames().size() != inputPortTypes.size())
    return emitOpError("argNames size must equal extern input-port count");
  if (getResultNames().size() != outputPortTypes.size())
    return emitOpError("resultNames size must equal extern output-port count");
  for (auto [i, t] : llvm::enumerate(inputPortTypes)) {
    if (getInputs()[i].getType() != t)
      return emitOpError("input #")
             << i << " type mismatch with extern input port";
    auto name = dyn_cast<StringAttr>(getArgNames()[i]);
    if (!name || name != inputPortNames[i])
      return emitOpError("input #")
             << i << " name '"
             << (name ? name.getValue() : StringRef("<non-string>"))
             << "' does not match extern port name '"
             << inputPortNames[i].getValue() << "'";
  }
  for (auto [i, t] : llvm::enumerate(outputPortTypes)) {
    if (getResults()[i].getType() != t)
      return emitOpError("result #")
             << i << " type mismatch with extern output port";
    auto name = dyn_cast<StringAttr>(getResultNames()[i]);
    if (!name || name != outputPortNames[i])
      return emitOpError("result #")
             << i << " name '"
             << (name ? name.getValue() : StringRef("<non-string>"))
             << "' does not match extern port name '"
             << outputPortNames[i].getValue() << "'";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Wrappers for the `custom<Properties>` ODS directive.
//===----------------------------------------------------------------------===//

static ParseResult parseOpLibProperties(OpAsmParser &parser, ArrayAttr &attr) {
  auto result = parseOptionalPropertyArray(attr, parser);
  if (!result.has_value() || succeeded(*result))
    return success();
  return failure();
}

static void printOpLibProperties(OpAsmPrinter &p, Operation *op,
                                 ArrayAttr attr) {
  if (!attr)
    return;
  printPropertyArray(attr, p);
}

static ParseResult parseTargetTypes(OpAsmParser &parser, TypeAttr &attr) {
  auto &builder = parser.getBuilder();
  SmallVector<OpAsmParser::Argument> arguments;

  auto result = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        // Parse argument name if present.
        OpAsmParser::Argument argument;
        auto argPresent = parser.parseOptionalArgument(
            argument, /*allowType=*/true, /*allowAttrs=*/true);
        if (argPresent.has_value()) {
          if (failed(argPresent.value()))
            return failure(); // Present but malformed.

          // Reject this if the preceding argument was missing a name.
          if (!arguments.empty() && arguments.back().ssaName.name.empty())
            return parser.emitError(argument.ssaName.location,
                                    "expected type instead of SSA identifier");

        } else {
          return failure();
        }
        arguments.push_back(argument);
        return success();
      });
  if (failed(result))
    return failure();

  // Parse the function signature.
  SMLoc signatureLocation = parser.getCurrentLocation();
  std::string errorMessage;
  SmallVector<Type> argTypes;
  SmallVector<Type> resultTypes;
  argTypes.reserve(arguments.size());
  for (auto &arg : arguments)
    argTypes.push_back(arg.type);
  Type type = builder.getFunctionType(argTypes, resultTypes);
  if (!type) {
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;
  }
  attr = TypeAttr::get(type);

  return success();
}

static void printTargetTypes(OpAsmPrinter &p, Operation *op, TypeAttr attr) {
  if (!attr)
    return;

  Region &body = op->getRegion(1);

  auto funcType = llvm::dyn_cast_or_null<FunctionType>(attr.getValue());

  if (funcType == nullptr)
    return;

  ArrayRef<Type> argTypes = funcType.getInputs();
  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    ArrayRef<NamedAttribute> attrs;
    p.printRegionArgument(body.getArgument(i), attrs);
  }

  p << ')';
}

//===----------------------------------------------------------------------===//
// TableGen'ed code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/OpLib/OpLib.cpp.inc"
