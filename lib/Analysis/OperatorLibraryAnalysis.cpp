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
#include "circt/Dialect/Calyx/CalyxDialect.h"
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

  auto operatorOps = libraryOp.getBodyBlock()->getOps<oplib::OperatorOp>();

  for (auto operatorOp : operatorOps) {
    // The signature / matching info lives on the OperatorOp's TargetOp +
    // OperationOp; the calyx_match body is only needed for downstream
    // Calyx lowering. Operators without a calyx_match are still tracked
    // here so their latency/signature drives scheduling — LoopScheduleToCalyx
    // is responsible for falling back to its TypeSwitch path when the
    // operator has no calyx_match template.
    auto targetOps = operatorOp.getBodyBlock()->getOps<oplib::TargetOp>();
    if (targetOps.empty())
      continue;
    auto targetOp = *targetOps.begin();
    if (targetOp.getBodyBlock()->empty())
      continue;
    auto operationOp =
        dyn_cast<oplib::OperationOp>(targetOp.getBodyBlock()->front());
    if (!operationOp)
      continue;

    std::string name = operationOp.getOpName().str();
    auto operationName = OperationName(name, context);

    Operator operatorStruct(operatorOp.getLatency());

    operatorStruct.incDelay = operatorOp.getIncDelay();
    operatorStruct.outDelay = operatorOp.getOutDelay();
    operatorStruct.templateOp = nullptr; // populated only when calyx_match exists
    operatorStruct.opToMatch = operationOp;

    if (operationOp.getOpDict().has_value()) {
      auto dict = operationOp.getOpDict().value();

      for (auto attr : dict) {
        operatorStruct.attrs.push_back(attr);
      }
    }

    auto hwMatches = operatorOp.getBodyBlock()->getOps<oplib::HwMatchOp>();
    if (!hwMatches.empty())
      operatorStruct.hwMatchOp = *hwMatches.begin();

    auto calyxMatches = operatorOp.getBodyBlock()->getOps<oplib::CalyxMatchOp>();
    if (!calyxMatches.empty()) {
      auto matchOp = *calyxMatches.begin();
      operatorStruct.templateOp = &matchOp.getBodyBlock()->front();

      auto yieldOp =
          cast<oplib::YieldOp>(matchOp.getBodyBlock()->getTerminator());

      if (yieldOp.getClock() != nullptr) {
        auto clockResultNum =
            cast<OpResult>(yieldOp.getClock()).getResultNumber();
        operatorStruct.clock = clockResultNum;
      }

      if (yieldOp.getReset() != nullptr) {
        auto resetResultNum =
            cast<OpResult>(yieldOp.getReset()).getResultNumber();
        operatorStruct.reset = resetResultNum;
      }

      if (yieldOp.getClockEnable() != nullptr) {
        auto ceResultNum =
            cast<OpResult>(yieldOp.getClockEnable()).getResultNumber();
        operatorStruct.ce = ceResultNum;
      }

      for (auto iv : llvm::enumerate(yieldOp.getInputs())) {
        auto i = iv.index();
        auto input = iv.value();
        assert(input.getDefiningOp() == &matchOp.getBodyBlock()->front() &&
               "matchOp can only contain one calyx cell");
        auto opResultNum = cast<OpResult>(input).getResultNumber();
        operatorStruct.operandToCellResultMapping[i] = opResultNum;
      }

      for (auto iv : llvm::enumerate(yieldOp.getOutputs())) {
        auto i = iv.index();
        auto output = iv.value();
        assert(output.getDefiningOp() == &matchOp.getBodyBlock()->front() &&
               "matchOp can only contain one calyx cell");
        auto opResultNum = cast<OpResult>(output).getResultNumber();
        operatorStruct.resultToCellResultMapping[i] = opResultNum;
      }
    }

    auto operatorName = operatorOp.getSymName();
    potentialOperatorsMap[operationName.getStringRef()].push_back(operatorName);

    operatorMap.insert(std::pair(operatorName, operatorStruct));
  }
}

Operation *
OperatorLibraryAnalysis::getOperatorTemplateOp(StringRef operatorName) {
  return operatorMap.at(operatorName).templateOp;
}

circt::oplib::HwMatchOp
OperatorLibraryAnalysis::getHwMatchOp(StringRef operatorName) {
  return operatorMap.at(operatorName).hwMatchOp;
}

static bool allAttributesMatch(Operation *op, Operator &operatorStruct) {
  // All attributes match
  for (auto namedAttr : operatorStruct.attrs) {
    auto otherAttr = op->getAttr(namedAttr.getName());
    if (namedAttr.getValue() != otherAttr) {
      return false;
    }
  }
  return true;
}

static bool allOperandTypesMatch(Operation *op, Operator &operatorStruct) {
  auto *opToMatch = operatorStruct.opToMatch;
  if (op->getOperands().size() != opToMatch->getOperands().size()) {
    return false;
  }

  return llvm::all_of(
      llvm::zip(op->getOperandTypes(), opToMatch->getOperandTypes()),
      [](const auto &p) { return std::get<0>(p) == std::get<1>(p); });
}

static bool allResultTypesMatch(Operation *op, Operator &operatorStruct) {
  auto *opToMatch = operatorStruct.opToMatch;
  if (op->getResults().size() != opToMatch->getResults().size()) {
    return false;
  }

  return llvm::all_of(
      llvm::zip(op->getResultTypes(), opToMatch->getResultTypes()),
      [](const auto &p) { return std::get<0>(p) == std::get<1>(p); });
}

SmallVector<StringRef>
OperatorLibraryAnalysis::getPotentialOperators(Operation *op) {
  ArrayRef<StringRef> vec = potentialOperatorsMap[op->getName().getStringRef()];
  SmallVector<StringRef> potentialOperators;

  for (auto po : vec) {
    auto operatorStruct = operatorMap.at(po);

    if (!allAttributesMatch(op, operatorStruct)) {
      continue;
    }

    if (!allOperandTypesMatch(op, operatorStruct)) {
      continue;
    }

    if (!allResultTypesMatch(op, operatorStruct)) {
      continue;
    }
    potentialOperators.push_back(po);
  }

  return potentialOperators;
}

StringRef
OperatorLibraryAnalysis::getOperatorBySymbol(mlir::SymbolRefAttr symbol) {
  auto name = symbol.getLeafReference();

  return name.getValue();
}

SmallVector<StringRef> OperatorLibraryAnalysis::getAllSupportedTargets() {
  SmallVector<StringRef> targets;

  for (auto &[key, _] : potentialOperatorsMap) {
    targets.push_back(key);
  }

  return targets;
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
OperatorLibraryAnalysis::getCEResultNum(StringRef operatorName) {
  auto operatorStruct = operatorMap.at(operatorName);
  return operatorStruct.ce;
}

unsigned
OperatorLibraryAnalysis::getCellResultForOperandNum(StringRef operatorName,
                                                    unsigned int operandNum) {
  auto operatorStruct = operatorMap.at(operatorName);
  auto num = operatorStruct.operandToCellResultMapping.at(operandNum);
  return num;
}

unsigned
OperatorLibraryAnalysis::getCellResultForResultNum(StringRef operatorName,
                                                   unsigned int resultNum) {
  auto operatorStruct = operatorMap.at(operatorName);
  auto num = operatorStruct.resultToCellResultMapping.at(resultNum);
  return num;
}
