//===- AccessNameAnalysis.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/AccessNameAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace circt::loopschedule;
using namespace mlir;
using namespace affine;

AccessNameAnalysis::AccessNameAnalysis(Operation *op) {
  op->walk([&](Operation *op) {
    if (isa<AffineLoadOp, AffineStoreOp, memref::LoadOp, memref::StoreOp,
            LoopScheduleLoadOp, LoopScheduleStoreOp, LoadInterface,
            StoreInterface>(op)) {
      auto nameAttr = op->getAttrOfType<StringAttr>("loopschedule.access_name");
      assert(nameAttr != nullptr &&
             "memory access must have `loopschedule.access_name` attribute");
      auto name = nameAttr.getValue();
      nameToOperationMap.insert(std::pair(name, op));
      operationToNameMap.insert(std::pair(op, name));
    }
  });
}

Operation *AccessNameAnalysis::getOperationFromName(StringRef name) {
  return nameToOperationMap.lookup(name);
}

StringRef AccessNameAnalysis::getOperationName(Operation *op) {
  return operationToNameMap.lookup(op);
}

void AccessNameAnalysis::replaceOp(Operation *oldOp, Operation *newOp) {
  assert(operationToNameMap.contains(oldOp));
  auto name = getOperationName(oldOp);
  operationToNameMap.erase(oldOp);
  operationToNameMap.insert(std::pair(newOp, name));
  nameToOperationMap.insert(std::pair(name, newOp));
}
