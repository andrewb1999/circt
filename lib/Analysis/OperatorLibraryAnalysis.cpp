//===- OperatorLibraryAnalysis.cpp - operator library analyses -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis on an operator library.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/OpLib/OpLibOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;

OperatorLibraryAnalysis::OperatorLibraryAnalysis(Operation *op) {
  auto *context = op->getContext();
  // Accept any function-like op carrying an `oplib.library` attribute.
  // SCFToLoopSchedule wraps the input func in a
  // `loopschedule.func_sequential` (still a FunctionOpInterface), so this
  // analysis must not be locked to `func::FuncOp`.
  if (!isa<mlir::FunctionOpInterface>(op)) {
    op->emitOpError(
        "must be a function-like op for OperatorLibraryAnalysis");
    return;
  }
  auto moduleOp = op->getParentOfType<ModuleOp>();

  if (!op->hasAttrOfType<SymbolRefAttr>("oplib.library")) {
    return;
  }

  auto libraryName = op->getAttrOfType<SymbolRefAttr>("oplib.library");
  auto libraryOp = cast<oplib::LibraryOp>(moduleOp.lookupSymbol(libraryName));

  // Every entry is tracked by symbol, so clients can query its properties
  // (latency, delays, `limit`) by name; entries that match ops by pattern
  // are also indexed by the operation they match, for
  // `getPotentialOperators`. Entries ops are assigned to explicitly (a
  // memref's or a memory port's) are found only through the reference.
  for (auto entry :
       libraryOp.getBodyBlock()->getOps<oplib::LibraryEntryOpInterface>()) {
    Operator operatorStruct(entry.getEntryLatency());
    if (auto delay = entry.getIncomingDelay())
      operatorStruct.incDelay = llvm::APFloat(*delay);
    if (auto delay = entry.getOutgoingDelay())
      operatorStruct.outDelay = llvm::APFloat(*delay);
    operatorStruct.limit = entry.getEntryLimit();
    operatorStruct.entry = entry;
    operatorStruct.hwMatchOp =
        dyn_cast_or_null<oplib::HwMatchOp>(entry.getHwMatch());

    StringRef operatorName = entry.getEntryName();
    if (auto matched = entry.getMatchedOperationName())
      potentialOperatorsMap[OperationName(*matched, context).getStringRef()]
          .push_back(operatorName);
    operatorMap.insert(std::pair(operatorName, operatorStruct));
  }
}

circt::oplib::HwMatchOp
OperatorLibraryAnalysis::getHwMatchOp(StringRef operatorName) {
  return operatorMap.at(operatorName).hwMatchOp;
}

SmallVector<StringRef>
OperatorLibraryAnalysis::getPotentialOperators(Operation *op) {
  SmallVector<StringRef> potentialOperators;
  for (auto po : potentialOperatorsMap[op->getName().getStringRef()])
    if (operatorMap.at(po).entry.matches(op))
      potentialOperators.push_back(po);
  return potentialOperators;
}

bool OperatorLibraryAnalysis::operatorMatches(StringRef operatorName,
                                              Operation *op) {
  auto it = operatorMap.find(operatorName);
  return it != operatorMap.end() && it->second.entry.matches(op);
}

StringRef
OperatorLibraryAnalysis::getOperatorBySymbol(mlir::SymbolRefAttr symbol) {
  auto name = symbol.getLeafReference();

  return name.getValue();
}

bool OperatorLibraryAnalysis::hasOperator(StringRef operatorName) {
  return operatorMap.find(operatorName) != operatorMap.end();
}

unsigned OperatorLibraryAnalysis::getOperatorLatency(StringRef operatorName) {
  auto operatorStruct = operatorMap.at(operatorName);

  return operatorStruct.latency;
}

std::optional<float>
OperatorLibraryAnalysis::getOperatorIncomingDelay(StringRef operatorName) {
  auto operatorStruct = operatorMap.at(operatorName);

  if (!operatorStruct.incDelay.has_value())
    return std::nullopt;

  return (float)operatorStruct.incDelay.value().convertToDouble();
}

std::optional<float>
OperatorLibraryAnalysis::getOperatorOutgoingDelay(StringRef operatorName) {
  auto operatorStruct = operatorMap.at(operatorName);

  if (!operatorStruct.outDelay.has_value())
    return std::nullopt;

  return (float)operatorStruct.outDelay.value().convertToDouble();
}

std::optional<unsigned>
OperatorLibraryAnalysis::getOperatorLimit(StringRef operatorName) {
  auto it = operatorMap.find(operatorName);
  if (it == operatorMap.end())
    return std::nullopt;
  return it->second.limit;
}
