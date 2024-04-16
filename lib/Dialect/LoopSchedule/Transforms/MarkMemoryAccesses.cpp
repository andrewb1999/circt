//===- Schedule.cpp - Schedule pass -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Schedule pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleAttributes.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "circt/Scheduling/Algorithms.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/StringExtras.h"
#include <set>

using namespace circt;
using namespace loopschedule;
using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct MarkMemoryAccessesPass
    : public MarkMemoryAccessesBase<MarkMemoryAccessesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

std::string getUniqueName(const std::string &nameBase,
                          llvm::StringMap<Operation *> &usedNames,
                          std::map<std::string, int> &counters) {
  std::string candidate;
  do {
    candidate = nameBase + std::to_string(counters[nameBase]++);
  } while (usedNames.contains(candidate));

  return candidate;
}

std::string getBaseName(Operation *op) {
  if (isa<AffineReadOpInterface, memref::LoadOp, LoopScheduleLoadOp,
          LoadInterface>(op)) {
    return "load";
  }

  return "store";
}

void MarkMemoryAccessesPass::runOnOperation() {
  auto funcOp = getOperation();
  llvm::StringMap<Operation *> usedNames;
  std::map<std::string, int> counters;

  // First walk through and record names that have already been assigned
  funcOp->walk([&](Operation *op) {
    if (isa<AffineWriteOpInterface, AffineReadOpInterface, memref::LoadOp,
            memref::StoreOp, LoopScheduleLoadOp, LoopScheduleStoreOp,
            LoadInterface, StoreInterface>(op)) {
      if (op->hasAttrOfType<StringAttr>(NameAnalysis::getAttributeName())) {
        auto nameAttr =
            op->getAttrOfType<StringAttr>(NameAnalysis::getAttributeName());
        auto name = nameAttr.getValue();
        if (usedNames.contains(name)) {
          emitError(op->getLoc()) << "name '" + name.str() + "' is not unique";
          signalPassFailure();
        }
        usedNames.insert(std::pair(name, op));
      }
    }
    return WalkResult::advance();
  });

  // Assign unique names to accesses that do not have names yet
  funcOp->walk([&](Operation *op) {
    if (isa<AffineWriteOpInterface, AffineReadOpInterface, memref::LoadOp,
            memref::StoreOp, LoopScheduleLoadOp, LoopScheduleStoreOp,
            LoadInterface, StoreInterface>(op)) {
      if (!op->hasAttrOfType<StringAttr>(NameAnalysis::getAttributeName())) {
        auto name = getUniqueName(getBaseName(op), usedNames, counters);
        StringAttr nameAttr = StringAttr::get(funcOp.getContext(), name);
        op->setAttr(NameAnalysis::getAttributeName(), nameAttr);
        usedNames.insert(std::pair(StringRef(name), op));
      }
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<mlir::Pass>
circt::loopschedule::createMarkMemoryAccessesPass() {
  return std::make_unique<MarkMemoryAccessesPass>();
}
