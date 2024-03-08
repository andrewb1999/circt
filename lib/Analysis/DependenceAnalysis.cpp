//===- DependenceAnalysis.cpp - memory dependence analyses ----------------===//
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

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

static void checkMemrefDependence(SmallVectorImpl<Operation *> &memoryOps,
                                  unsigned depth,
                                  MemoryDependenceResult &results) {
  if (memoryOps.empty())
    return;

  auto funcOp = memoryOps.front()->getParentOfType<func::FuncOp>();

  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      // Initialize the dependence list for this destination.
      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      // Look for inter-iteration dependences on the same memory location.
      MemRefAccess src(source);
      MemRefAccess dst(destination);
      FlatAffineValueConstraints dependenceConstraints;
      SmallVector<DependenceComponent, 2> depComps;

      // Requested depth might not be a valid comparison if they do not belong
      // to the same loop nest
      if (depth > getInnermostCommonLoopDepth({source, destination}))
        continue;

      DependenceResult result = checkMemrefAccessDependence(
          src, dst, depth, &dependenceConstraints, &depComps, true);

      results[destination].emplace_back(source, result.value, depComps);

      // Also consider intra-iteration dependences on the same memory location.
      // This currently does not consider aliasing.
      if (src != dst)
        continue;

      // Collect surrounding loops to use in dependence components. Only proceed
      // if we are in the innermost loop.
      SmallVector<AffineForOp> enclosingLoops;
      getAffineForIVs(*destination, &enclosingLoops);
      if (enclosingLoops.size() != depth)
        continue;

      // Look for the common parent that src and dst share. If there is none,
      // there is nothing more to do.
      SmallVector<Operation *> srcParents;
      getEnclosingAffineOps(*source, &srcParents);
      SmallVector<Operation *> dstParents;
      getEnclosingAffineOps(*destination, &dstParents);

      Operation *commonParent = nullptr;
      for (auto *srcParent : llvm::reverse(srcParents)) {
        for (auto *dstParent : llvm::reverse(dstParents)) {
          if (srcParent == dstParent)
            commonParent = srcParent;
          if (commonParent != nullptr)
            break;
        }
        if (commonParent != nullptr)
          break;
      }

      if (commonParent == nullptr)
        commonParent = funcOp;

      // Check the common parent's regions.
      for (auto &commonRegion : commonParent->getRegions()) {
        if (commonRegion.empty())
          continue;

        // Only support structured constructs with single-block regions for now.
        assert(commonRegion.hasOneBlock() &&
               "only single-block regions are supported");

        Block &commonBlock = commonRegion.front();

        // Find the src and dst ancestor in the common block, if any.
        Operation *srcOrAncestor = commonBlock.findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock.findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          continue;

        // Check if the src or its ancestor is before the dst or its ancestor.
        if (srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
          SmallVector<DependenceComponent> intraDeps;

          // Func
          DependenceComponent depComp;
          depComp.op = funcOp;
          depComp.lb = std::nullopt;
          depComp.ub = std::nullopt;
          intraDeps.push_back(depComp);

          // Build dependence components for each loop depth.
          for (size_t i = 0; i < depth; ++i) {
            DependenceComponent depComp;
            depComp.op = enclosingLoops[i];
            depComp.lb = 0;
            depComp.ub = 0;
            intraDeps.push_back(depComp);
          }

          results[dstOrAncestor].emplace_back(
              srcOrAncestor, DependenceResult::HasDependence, intraDeps);
        }
      }
    }
  }
}

static Value getMemref(Operation *op) {
  Value memref = isa<AffineStoreOp>(*op)  ? cast<AffineStoreOp>(*op).getMemRef()
                 : isa<AffineLoadOp>(*op) ? cast<AffineLoadOp>(*op).getMemRef()
                 : isa<memref::StoreOp>(*op)
                     ? cast<memref::StoreOp>(*op).getMemRef()
                     : cast<memref::LoadOp>(*op).getMemRef();
  return memref;
}

/// Helper to iterate through memory operation pairs and check for dependencies
/// at a given loop nesting depth.
static void
checkSchedInterfaceDependence(SmallVectorImpl<Operation *> &memoryOps,
                              MemoryDependenceResult &results) {

  auto funcOp = memoryOps.front()->getParentOfType<func::FuncOp>();

  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      assert(isa<SchedulableAffineInterface>(source));
      assert(isa<SchedulableAffineInterface>(destination));

      // Initialize the dependence list for this destination.
      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      // Insert inter-iteration dependencies for SchedulableAffineInterface
      auto src = dyn_cast<SchedulableAffineInterface>(source);
      auto dst = dyn_cast<SchedulableAffineInterface>(destination);

      auto depth = getNumCommonSurroundingLoops(*src, *dst);

      // Check if the src or its ancestor is before the dst or its ancestor.
      if (depth > 0) {
        if (auto *commonBlock =
                getCommonBlockInAffineScope(source, destination)) {

          Operation *srcOrAncestor =
              commonBlock->findAncestorOpInBlock(*source);
          Operation *dstOrAncestor =
              commonBlock->findAncestorOpInBlock(*destination);
          if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
            continue;

          // Check if the dst or its ancestor is before the src or its ancestor.
          // We want to dst to be before the src to insert iter-iteration deps.
          // This is a short-term, conservative hack to avoid having to do real
          // affine memory access analysis.
          if (dstOrAncestor->isBeforeInBlock(srcOrAncestor)) {
            SmallVector<AffineForOp> enclosingLoops;
            getAffineForIVs(*destination, &enclosingLoops);

            if (dst.hasDependence(src)) {
              SmallVector<DependenceComponent> depComps;
              for (unsigned i = 0; i < depth; ++i) {
                DependenceComponent comp;
                comp.lb = 1;
                comp.ub = 1;
                comp.op = enclosingLoops[i];
                depComps.push_back(comp);
              }
              results[dstOrAncestor].emplace_back(
                  srcOrAncestor, DependenceResult::HasDependence, depComps);
            }
          }
        }
      }

      // Some AMC operations (dynamic ports) do not have intra-iteration deps
      if (!src.hasIntraIterationDeps() || !dst.hasIntraIterationDeps())
        continue;

      // Collect surrounding loops to use in dependence components. Only proceed
      // if we are in the innermost loop.
      SmallVector<AffineForOp> enclosingLoops;
      getAffineForIVs(*destination, &enclosingLoops);

      // Look for the common parent that src and dst share. If there is none,
      // there is nothing more to do.
      SmallVector<Operation *> srcParents;
      getEnclosingAffineOps(*source, &srcParents);
      SmallVector<Operation *> dstParents;
      getEnclosingAffineOps(*destination, &dstParents);

      Operation *commonParent = nullptr;
      for (auto *srcParent : llvm::reverse(srcParents)) {
        for (auto *dstParent : llvm::reverse(dstParents)) {
          if (srcParent == dstParent)
            commonParent = srcParent;
          if (commonParent != nullptr)
            break;
        }
        if (commonParent != nullptr)
          break;
      }

      if (commonParent == nullptr)
        continue;
      // commonParent = funcOp;

      // Check the common parent's regions.
      for (auto &commonRegion : commonParent->getRegions()) {
        if (commonRegion.empty())
          continue;

        // Only support structured constructs with single-block regions for now.
        assert(commonRegion.hasOneBlock() &&
               "only single-block regions are supported");

        Block &commonBlock = commonRegion.front();

        // Find the src and dst ancestor in the common block, if any.
        Operation *srcOrAncestor = commonBlock.findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock.findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          continue;

        // Check if the src or its ancestor is before the dst or its ancestor.
        if (srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
          // Build dependence components for each loop depth.
          SmallVector<DependenceComponent> intraDeps;

          // Func
          DependenceComponent depComp;
          depComp.op = funcOp;
          depComp.lb = std::nullopt;
          depComp.ub = std::nullopt;
          intraDeps.push_back(depComp);

          for (size_t i = 0; i < depth; ++i) {
            DependenceComponent depComp;
            depComp.op = enclosingLoops[i];
            depComp.lb = 0;
            depComp.ub = 0;
            intraDeps.push_back(depComp);
          }

          results[dstOrAncestor].emplace_back(
              srcOrAncestor, DependenceResult::HasDependence, intraDeps);
        }
      }
    }
  }
}

/// Helper to iterate through memory operation pairs and check for dependencies
/// at a given loop nesting depth.
static void checkNonAffineDependence(SmallVectorImpl<Operation *> &memoryOps,
                                     MemoryDependenceResult &results) {

  auto funcOp = memoryOps.front()->getParentOfType<func::FuncOp>();

  for (auto *source : memoryOps) {
    for (auto *destination : memoryOps) {
      if (source == destination)
        continue;

      assert((isa<LoadInterface, StoreInterface, memref::LoadOp,
                  memref::StoreOp, SchedulableAffineInterface>(source)));
      assert((isa<LoadInterface, StoreInterface, memref::LoadOp,
                  memref::StoreOp, SchedulableAffineInterface>(destination)));

      // Initialize the dependence list for this destination.
      if (results.count(destination) == 0)
        results[destination] = SmallVector<MemoryDependence>();

      auto depth = getNumCommonSurroundingLoops(*source, *destination);

      // Check if the src or its ancestor is before the dst or its ancestor.
      if (depth > 0) {
        if (auto *commonBlock =
                getCommonBlockInAffineScope(source, destination)) {

          Operation *srcOrAncestor =
              commonBlock->findAncestorOpInBlock(*source);
          Operation *dstOrAncestor =
              commonBlock->findAncestorOpInBlock(*destination);
          if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
            continue;

          // Check if the dst or its ancestor is before the src or its ancestor.
          // We want to dst to be before the src to insert iter-iteration deps.
          // This is a short-term, conservative hack to avoid having to do real
          // affine memory access analysis.
          if (dstOrAncestor->isBeforeInBlock(srcOrAncestor)) {
            SmallVector<AffineForOp> enclosingLoops;
            getAffineForIVs(*destination, &enclosingLoops);

            bool hasDep = false;
            if (auto dst = dyn_cast<LoadInterface>(destination)) {
              hasDep = dst.hasDependence(source);
            } else if (auto dst = dyn_cast<StoreInterface>(destination)) {
              hasDep = dst.hasDependence(source);
            } else if (auto dst =
                           dyn_cast<SchedulableAffineInterface>(destination)) {
              hasDep = dst.hasDependence(source);
            } else if (isa<memref::LoadOp, memref::StoreOp, AffineLoadOp,
                           AffineStoreOp>(source)) {
              hasDep = getMemref(destination) == getMemref(source);
            }

            if (hasDep) {
              SmallVector<DependenceComponent> depComps;
              for (unsigned i = 0; i < depth; ++i) {
                DependenceComponent comp;
                comp.lb = 1;
                comp.ub = 1;
                comp.op = enclosingLoops[i];
                depComps.push_back(comp);
              }
              results[dstOrAncestor].emplace_back(
                  srcOrAncestor, DependenceResult::HasDependence, depComps);
            }
          }
        }
      }

      // Look for the common parent that src and dst share. If there is none,
      // there is nothing more to do.
      SmallVector<Operation *> srcParents;
      getEnclosingAffineOps(*source, &srcParents);
      SmallVector<Operation *> dstParents;
      getEnclosingAffineOps(*destination, &dstParents);

      Operation *commonParent = nullptr;
      for (auto *srcParent : llvm::reverse(srcParents)) {
        for (auto *dstParent : llvm::reverse(dstParents)) {
          if (srcParent == dstParent)
            commonParent = srcParent;
          if (commonParent != nullptr)
            break;
        }
        if (commonParent != nullptr)
          break;
      }

      if (commonParent == nullptr)
        commonParent = funcOp;

      // Check the common parent's regions.
      for (auto &commonRegion : commonParent->getRegions()) {
        if (commonRegion.empty())
          continue;

        // Only support structured constructs with single-block regions for now.
        assert(commonRegion.hasOneBlock() &&
               "only single-block regions are supported");

        Block &commonBlock = commonRegion.front();

        // Find the src and dst ancestor in the common block, if any.
        Operation *srcOrAncestor = commonBlock.findAncestorOpInBlock(*source);
        Operation *dstOrAncestor =
            commonBlock.findAncestorOpInBlock(*destination);
        if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
          continue;

        // Check if the src or its ancestor is before the dst or its ancestor.
        if (srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
          // Build dependence components for each loop depth.
          SmallVector<DependenceComponent> intraDeps;
          SmallVector<AffineForOp> enclosingLoops;
          getAffineForIVs(*destination, &enclosingLoops);

          bool hasDep = false;
          if (auto dst = dyn_cast<LoadInterface>(destination)) {
            hasDep = dst.hasDependence(source);
          } else if (auto dst = dyn_cast<StoreInterface>(destination)) {
            hasDep = dst.hasDependence(source);
          } else if (auto dst =
                         dyn_cast<SchedulableAffineInterface>(destination)) {
            hasDep = dst.hasDependence(source);
          } else if (isa<memref::LoadOp, memref::StoreOp, AffineLoadOp,
                         AffineStoreOp>(source)) {
            hasDep = getMemref(destination) == getMemref(source);
          }

          if (hasDep) {
            // Func
            DependenceComponent depComp;
            depComp.op = funcOp;
            depComp.lb = std::nullopt;
            depComp.ub = std::nullopt;
            intraDeps.push_back(depComp);

            for (size_t i = 0; i < depth; ++i) {
              DependenceComponent depComp;
              depComp.op = enclosingLoops[i];
              depComp.lb = 0;
              depComp.ub = 0;
              intraDeps.push_back(depComp);
            }

            results[dstOrAncestor].emplace_back(
                srcOrAncestor, DependenceResult::HasDependence, intraDeps);
          }
        }
      }
    }
  }
}

/// MemoryDependenceAnalysis traverses any AffineForOps in the FuncOp body and
/// checks for memory access dependences. Results are captured in a
/// MemoryDependenceResult, which can by queried by Operation.
circt::analysis::MemoryDependenceAnalysis::MemoryDependenceAnalysis(
    Operation *op) {
  auto funcOp = cast<func::FuncOp>(op);

  // Collect affine loops grouped by nesting depth.
  std::vector<SmallVector<AffineForOp, 2>> depthToLoops;
  mlir::affine::gatherLoops(funcOp, depthToLoops);

  // Collect load and store operations to check.
  SmallVector<Operation *> memrefOps;
  funcOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      memrefOps.push_back(op);
  });

  // For each depth, check memref accesses.
  for (unsigned depth = 1, e = depthToLoops.size(); depth <= e; ++depth)
    checkMemrefDependence(memrefOps, depth, results);

  SmallVector<Operation *> schedInterfaceOps;
  funcOp.walk([&](Operation *op) {
    if (isa<SchedulableAffineInterface>(op))
      schedInterfaceOps.push_back(op);
  });

  if (!schedInterfaceOps.empty())
    checkSchedInterfaceDependence(schedInterfaceOps, results);

  SmallVector<Operation *> memoryOps;
  funcOp.walk([&](Operation *op) {
    if (isa<LoadInterface, StoreInterface, SchedulableAffineInterface,
            memref::LoadOp, memref::StoreOp>(op))
      memoryOps.push_back(op);
  });

  if (!memoryOps.empty())
    checkNonAffineDependence(memoryOps, results);
}

/// Returns the dependences, if any, that the given Operation depends on.
ArrayRef<MemoryDependence>
circt::analysis::MemoryDependenceAnalysis::getDependences(Operation *op) {
  return results[op];
}

void dumpMap(MemoryDependenceResult &results) {
  llvm::errs() << "\ndump map\n=======================================\n";
  for (auto &it : results) {
    auto *op = it.first;
    llvm::errs() << "\nop\n";
    op->dump();
    // auto deps = it.getSecond();
    // if (!deps.empty()) {
    //   llvm::errs() << "deps\n";
    //   for (auto &dep : deps) {
    //     dep.source->dump();
    //   }
    // }
  }
}

/// Replaces the dependences, if any, from the oldOp to the newOp.
void circt::analysis::MemoryDependenceAnalysis::replaceOp(Operation *oldOp,
                                                          Operation *newOp) {
  // llvm::errs() << "\noldOp: ";
  // oldOp->dump();
  // llvm::errs() << "newOp: ";
  // newOp->dump();
  // llvm::errs() << "replace\n";
  // newOp->dump();
  // dumpMap(results);
  // If oldOp had any dependences.
  auto deps = results[oldOp];
  // llvm::errs() << "move dep\n";
  // Move the dependences to newOp.
  results[newOp] = deps;
  results.erase(oldOp);

  // auto test = results.find(newOp);
  // if (test != results.end()) {
  //   test->first->dump();
  // }
  // dumpMap(results);

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

bool circt::analysis::MemoryDependenceAnalysis::containsOp(Operation *op) {
  if (results.count(op) > 0 && !results[op].empty()) {
    llvm::errs() << "contains\n";
    op->dump();
    for (auto dep : results[op]) {
      llvm::errs() << "dep: ";
      dep.source->dump();
    }
    return true;
  }

  for (auto &it : results)
    for (auto &dep : it.second)
      // if (OperationEquivalence::isEquivalentTo(dep.source, op,
      // OperationEquivalence::IgnoreLocations)) {
      if (dep.source == op) {
        // llvm::errs() << "dep.dest\n";
        // it.first->dump();
        llvm::errs() << "dep.source\n";
        op->dump();
        return true;
      }

  return false;
}
