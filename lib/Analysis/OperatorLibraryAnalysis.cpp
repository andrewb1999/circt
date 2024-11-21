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
#include <cassert>

using namespace mlir;
using namespace mlir::affine;
using namespace circt::analysis;

OperatorLibraryAnalysis::OperatorLibraryAnalysis(Operation *op) {
  auto *context = op->getContext();
  if (!isa<func::FuncOp>(op)) {
    op->emitOpError("must be a FuncOp for OperatorLibraryAnalysis");
    return;
  }
  auto funcOp = cast<func::FuncOp>(op);
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();

  if (!funcOp->hasAttrOfType<SymbolRefAttr>("oplib.library")) {
    return;
  }

  auto libraryName = funcOp->getAttrOfType<SymbolRefAttr>("oplib.library");
  auto libraryOp = cast<oplib::LibraryOp>(moduleOp.lookupSymbol(libraryName));

  auto operatorOps = libraryOp.getBodyBlock()->getOps<oplib::OperatorOp>();

  for (auto operatorOp : operatorOps) {
    auto matchOp =
        cast<oplib::CalyxMatchOp>(operatorOp.getBodyBlock()->getTerminator());

    auto targetOp =
        cast<oplib::TargetOp>(operatorOp.lookupSymbol(matchOp.getTarget()));

    auto operationOp =
        cast<oplib::OperationOp>(targetOp.getBodyBlock()->front());

    std::string name = operationOp.getDialectName().str() + "." +
                       operationOp.getOpName().str();
    auto operationName = OperationName(name, context);

    Operator operatorStruct(operatorOp.getLatency());

    operatorStruct.incDelay = operatorOp.getIncDelay();
    operatorStruct.outDelay = operatorOp.getOutDelay();
    operatorStruct.templateOp = &matchOp.getBodyBlock()->front();

    auto yieldOp =
        cast<oplib::YieldOp>(matchOp.getBodyBlock()->getTerminator());

    auto clockResultNum = cast<OpResult>(yieldOp.getClock()).getResultNumber();
    operatorStruct.clock = clockResultNum;

    auto resetResultNum =
        cast<OpResult>(yieldOp.getReset()).getResultNumber();
    operatorStruct.reset = resetResultNum;

    auto ceResultNum =
        cast<OpResult>(yieldOp.getClockEnable()).getResultNumber();
    operatorStruct.ce = ceResultNum;

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

    auto operatorName = operatorOp.getSymName();
    potentialOperatorsMap[operationName.getStringRef()].push_back(operatorName);

    operatorMap.insert(std::pair(operatorName, operatorStruct));
  }
}

Operation *
OperatorLibraryAnalysis::getOperatorTemplateOp(StringRef operatorName) {
  return operatorMap.at(operatorName).templateOp;
}

ArrayRef<StringRef>
OperatorLibraryAnalysis::getPotentialOperators(Operation *op) {
  return potentialOperatorsMap[op->getName().getStringRef()];
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

  return operatorStruct.incDelay.value().convertToFloat();
}

std::optional<float>
OperatorLibraryAnalysis::getOperatorOutgoingDelay(StringRef operatorName) {
  auto operatorStruct = operatorMap.at(operatorName);

  if (!operatorStruct.outDelay.has_value())
    return std::nullopt;

  return operatorStruct.outDelay.value().convertToFloat();
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
