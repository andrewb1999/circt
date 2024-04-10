//===- LoopScheudleDependenceAnalysis.cpp - memory dependence analyses ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that perform analysis involving memory access
// dependences.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Analysis/AccessNameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include <cassert>

using namespace mlir;
using namespace mlir::affine;
// using namespace circt;
using namespace circt::analysis;
using namespace circt::loopschedule;

/// Returns the closest surrounding block common to `opA` and `opB`. `opA` and
/// `opB` should be in the same affine scope. Returns nullptr if such a block
/// does not exist (when the two ops are in different blocks of an op starting
/// an `AffineScope`).
static Block *getCommonBlockInAffineScope(Operation *opA, Operation *opB) {
  // Get the chain of ancestor blocks for the given `MemRefAccess` instance. The
  // chain extends up to and includnig an op that starts an affine scope.
  auto getChainOfAncestorBlocks =
      [&](Operation *op, SmallVectorImpl<Block *> &ancestorBlocks) {
        Block *currBlock = op->getBlock();
        // Loop terminates when the currBlock is nullptr or its parent operation
        // holds an affine scope.
        while (currBlock &&
               !currBlock->getParentOp()->hasTrait<OpTrait::AffineScope>()) {
          ancestorBlocks.push_back(currBlock);
          currBlock = currBlock->getParentOp()->getBlock();
        }
        assert(currBlock &&
               "parent op starting an affine scope is always expected");
        ancestorBlocks.push_back(currBlock);
      };

  // Find the closest common block.
  SmallVector<Block *, 4> srcAncestorBlocks, dstAncestorBlocks;
  getChainOfAncestorBlocks(opA, srcAncestorBlocks);
  getChainOfAncestorBlocks(opB, dstAncestorBlocks);

  Block *commonBlock = nullptr;
  for (int i = srcAncestorBlocks.size() - 1, j = dstAncestorBlocks.size() - 1;
       i >= 0 && j >= 0 && srcAncestorBlocks[i] == dstAncestorBlocks[j];
       i--, j--)
    commonBlock = srcAncestorBlocks[i];

  return commonBlock;
}

/// MemoryDependenceAnalysis traverses any AffineForOps in the FuncOp body and
/// checks for memory access dependences. Results are captured in a
/// MemoryDependenceResult, which can by queried by Operation.
circt::analysis::LoopScheduleDependenceAnalysis::LoopScheduleDependenceAnalysis(
    Operation *op, AnalysisManager &analysisManager) {
  auto funcOp = cast<func::FuncOp>(op);
  auto accessNameAnalysis = analysisManager.getAnalysis<AccessNameAnalysis>();

  if (!funcOp->hasAttrOfType<SymbolRefAttr>("loopschedule.dependencies"))
    return;

  auto deps = funcOp->getAttrOfType<SymbolRefAttr>("loopschedule.dependencies");
  auto *symbolOp =
      SymbolTable::lookupNearestSymbolFrom(funcOp.getOperation(), deps);
  auto depsOp = cast<LoopScheduleDependenciesOp>(symbolOp);
  for (auto dependsOn : depsOp.getBody().getOps<LoopScheduleDependsOnOp>()) {
    auto destinationName = dependsOn.getAccessName();
    auto *destination =
        accessNameAnalysis.getOperationFromName(destinationName);
    assert(destination != nullptr);
    for (auto access : dependsOn.getBody().getOps<LoopScheduleAccessOp>()) {
      auto sourceName = access.getAccessName();
      auto *source = accessNameAnalysis.getOperationFromName(sourceName);
      assert(source != nullptr);
      auto dist = access.getDist().value_or(0);
      if (auto *commonBlock =
              getCommonBlockInAffineScope(source, destination)) {
        Operation *srcOrAncestor = commonBlock->findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock->findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          return;
        // Check if the dst or its ancestor is before the src or its ancestor.
        // We want to dst to be before the src to insert iter-iteration deps.
        if ((dist != 0 && dstOrAncestor->isBeforeInBlock(srcOrAncestor)) ||
            (dist == 0 && srcOrAncestor->isBeforeInBlock(dstOrAncestor))) {
          results[dstOrAncestor].emplace_back(srcOrAncestor, dist);
        }
      }
    }
  }
}

/// Returns the dependencies, if any, that the given Operation depends on.
ArrayRef<LoopScheduleDependence>
circt::analysis::LoopScheduleDependenceAnalysis::getDependencies(
    Operation *op) {
  return results[op];
}

/// Replaces the dependences, if any, from the oldOp to the newOp.
void circt::analysis::LoopScheduleDependenceAnalysis::replaceOp(
    Operation *oldOp, Operation *newOp) {
  // If oldOp had any dependences.
  auto deps = results[oldOp];
  // Move the dependences to newOp.
  results[newOp] = deps;
  results.erase(oldOp);

  // Find any dependences originating from oldOp and make newOp the source.
  // TODO(mikeurbach): consider adding an inverted index to avoid this scan.
  for (auto &it : results)
    for (auto &dep : it.second) {
      if (OperationEquivalence::isEquivalentTo(
              dep.source, oldOp, OperationEquivalence::IgnoreLocations)) {
        // if (dep.source == oldOp) {
        // llvm::errs() << "replace dest\n";
        // it.first->dump();
        // llvm::errs() << "replace src\n";
        // dep.source->dump();
        dep.source = newOp;
      }
    }

  // dumpMap(results);
}

bool circt::analysis::LoopScheduleDependenceAnalysis::containsOp(
    Operation *op) {
  if (results.count(op) > 0 && !results[op].empty()) {
    return true;
  }

  for (auto &it : results)
    for (auto &dep : it.second)
      if (dep.source == op) {
        return true;
      }

  return false;
}
