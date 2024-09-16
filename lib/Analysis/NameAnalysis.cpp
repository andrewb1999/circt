//===- NameAnalysis.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace circt::loopschedule;
using namespace mlir;
using namespace affine;

NameAnalysis::NameAnalysis(Operation *op) {
  op->walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp, LoopScheduleLoadOp,
            LoopScheduleStoreOp, LoadInterface, StoreInterface,
            AffineReadOpInterface, AffineWriteOpInterface>(op)) {
      auto nameAttr =
          op->getAttrOfType<StringAttr>(NameAnalysis::getAttributeName());
      assert(nameAttr != nullptr &&
             ("memory access must have `" + NameAnalysis::getAttributeName() +
              "` attribute")
                 .c_str());
      auto name = nameAttr.getValue();
      nameToOperationMap.insert(std::pair(name, op));
      operationToNameMap.insert(std::pair(op, name));
    }
  });
}

Operation *NameAnalysis::getOperationFromName(StringRef name) {
  auto *op = nameToOperationMap.lookup(name);
  assert(op != nullptr);
  return op;
}

StringRef NameAnalysis::getOperationName(Operation *op) {
  assert(operationToNameMap.contains(op));
  auto name = operationToNameMap.lookup(op);
  assert(!name.empty());
  return name;
}

void NameAnalysis::replaceOp(Operation *oldOp, Operation *newOp) {
  assert(operationToNameMap.contains(oldOp));
  auto name = getOperationName(oldOp);
  operationToNameMap.erase(oldOp);
  operationToNameMap.insert(std::pair(newOp, name));
  nameToOperationMap.insert(std::pair(name, newOp));
}
