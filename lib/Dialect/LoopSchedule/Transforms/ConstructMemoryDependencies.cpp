//===- ConstructMemoryDependencies.cpp - Dependencies pass ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the ConstructMemoryDependencies pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Analysis/NameAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleAttributes.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/LoopSchedulePasses.h"
#include "circt/Scheduling/Algorithms.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include <set>

using namespace circt;
using namespace loopschedule;
using namespace mlir;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct ConstructMemoryDependenciesPass
    : public ConstructMemoryDependenciesBase<ConstructMemoryDependenciesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

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

namespace {
struct MemoryDependence {
  MemoryDependence(StringRef source, unsigned int distance)
      : source(source), distance(distance) {}
  StringRef source;
  unsigned int distance;

  friend bool operator<(const MemoryDependence &lhs,
                        const MemoryDependence &rhs) {
    return std::tie(lhs.source, lhs.distance) <
           std::tie(rhs.source, rhs.distance);
  }
};
} // namespace

using MemoryDependenceResult = std::map<StringRef, std::set<MemoryDependence>>;

static void checkAffineAccessPair(Operation *source, Operation *destination,
                                  MemoryDependenceResult &results,
                                  NameAnalysis nameAnalysis,
                                  bool isIntraIteration) {
  if (source == destination)
    return;

  auto sourceIsSchedInterface = isa<SchedulableAffineInterface>(source);
  auto destIsSchedInterface = isa<SchedulableAffineInterface>(destination);

  if (sourceIsSchedInterface != destIsSchedInterface)
    return;

  if (sourceIsSchedInterface && destIsSchedInterface) {
    auto srcInterface = cast<SchedulableAffineInterface>(source);
    if (!srcInterface.hasDependence(destination))
      return;
  }

  // source->dump();
  // destination->dump();
  // llvm::errs() << "isIntraIteration: " << isIntraIteration << "\n";

  // Look for inter-iteration dependences on the same memory location.
  MemRefAccess src(source);
  MemRefAccess dst(destination);

  if (!sourceIsSchedInterface && !destIsSchedInterface) {
    if (dst.memref != src.memref) {
      // llvm::errs() << "=======================================\n";
      return;
    }
  }

  // Hack to avoid having to copy the checkMemrefAccessDependence function
  dst.memref = src.memref;
  FlatAffineValueConstraints dependenceConstraints;
  SmallVector<DependenceComponent, 2> depComps;

  // Requested depth might not be a valid comparison if they do not belong
  // to the same loop nest
  auto numCommonLoops = getNumCommonSurroundingLoops(*source, *destination);

  if (!isIntraIteration && numCommonLoops < 1) {
    // llvm::errs() << "=======================================\n";
    return;
  }

  auto depth = isIntraIteration ? numCommonLoops + 1 : numCommonLoops;

  DependenceResult result = checkMemrefAccessDependence(
      src, dst, depth, &dependenceConstraints, &depComps, false);

  if (hasDependence(result)) {
    if (!isIntraIteration) {
      assert(!depComps.empty());
      results[nameAnalysis.getOperationName(destination)].emplace(
          nameAnalysis.getOperationName(source), depComps.back().lb.value());
    } else {
      results[nameAnalysis.getOperationName(destination)].emplace(
          nameAnalysis.getOperationName(source), 0);
    }
  }
}

static Value getMemoryValue(Operation *op) {
  if (auto memrefLoad = dyn_cast<memref::LoadOp>(op)) {
    return memrefLoad.getMemref();
  }
  if (auto memrefStore = dyn_cast<memref::StoreOp>(op)) {
    return memrefStore.getMemref();
  }
  if (auto affineLoad = dyn_cast<AffineLoadOp>(op)) {
    return affineLoad.getMemref();
  }
  if (auto affineStore = dyn_cast<AffineStoreOp>(op)) {
    return affineStore.getMemref();
  }
  op->dump();
  assert(false && "Op does not have memref");
}

static bool isLoad(Operation *op) {
  return isa<memref::LoadOp, AffineReadOpInterface, LoadInterface>(op);
}

static void checkNonAffineAccessPair(Operation *source, Operation *destination,
                                     MemoryDependenceResult &results,
                                     NameAnalysis nameAnalysis,
                                     bool isIntraIteration) {
  if (source == destination)
    return;

  auto sourceIsSchedInterface =
      isa<SchedulableAffineInterface, LoadInterface, StoreInterface>(source);
  auto destIsSchedInterface =
      isa<SchedulableAffineInterface, LoadInterface, StoreInterface>(
          destination);

  if (sourceIsSchedInterface != destIsSchedInterface)
    return;

  if (auto srcInterface = dyn_cast<StoreInterface>(source)) {
    if (auto dstInterface = dyn_cast<StoreInterface>(destination)) {
      if (isIntraIteration) {
        if (auto *commonBlock =
                getCommonBlockInAffineScope(source, destination)) {

          Operation *srcOrAncestor =
              commonBlock->findAncestorOpInBlock(*source);
          Operation *dstOrAncestor =
              commonBlock->findAncestorOpInBlock(*destination);
          if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
            return;

          // Check if the src or its ancestor is before the dst or its ancestor.
          if (srcOrAncestor->isBeforeInBlock(dstOrAncestor) &&
              dstInterface.hasControlDependence(srcInterface)) {
            results[nameAnalysis.getOperationName(destination)].emplace(
                nameAnalysis.getOperationName(source), 0);
          }
        }
      }
    }
  }

  if (isa<SchedulableAffineInterface>(source) &&
      isa<SchedulableAffineInterface>(destination))
    return;

  if (auto srcInterface = dyn_cast<SchedulableAffineInterface>(source)) {
    if (!srcInterface.hasDependence(destination))
      return;
  }

  if (auto srcInterface = dyn_cast<LoadInterface>(source)) {
    if (!srcInterface.hasDependence(destination))
      return;
  }

  if (auto srcInterface = dyn_cast<StoreInterface>(source)) {
    if (!srcInterface.hasDependence(destination))
      return;
  }

  if (!sourceIsSchedInterface) {
    if (getMemoryValue(source) != getMemoryValue(destination))
      return;
  }

  // Requested depth might not be a valid comparison if they do not belong
  // to the same loop nest
  auto numCommonLoops = getNumCommonSurroundingLoops(*source, *destination);

  if (!isIntraIteration && numCommonLoops < 1)
    return;

  // source->dump();
  // destination->dump();
  // llvm::errs() << "isIntraIteration: " << isIntraIteration << "\n";
  // llvm::errs() << "source is load: " << isLoad(source) << "\n";
  // llvm::errs() << "dest is load: " << isLoad(destination) << "\n";
  // llvm::errs() << "=======================================\n";

  // We don't care about RAR dependencies
  if (isLoad(source) && isLoad(destination))
    return;

  if (auto *commonBlock = getCommonBlockInAffineScope(source, destination)) {

    Operation *srcOrAncestor = commonBlock->findAncestorOpInBlock(*source);
    Operation *dstOrAncestor = commonBlock->findAncestorOpInBlock(*destination);
    if (srcOrAncestor == nullptr || dstOrAncestor == nullptr)
      return;

    // Check if the dst or its ancestor is before the src or its ancestor.
    // We want to dst to be before the src to insert iter-iteration deps.
    if (!isIntraIteration && dstOrAncestor->isBeforeInBlock(srcOrAncestor)) {
      results[nameAnalysis.getOperationName(destination)].emplace(
          nameAnalysis.getOperationName(source), 1);
    } else if (isIntraIteration &&
               srcOrAncestor->isBeforeInBlock(dstOrAncestor)) {
      results[nameAnalysis.getOperationName(destination)].emplace(
          nameAnalysis.getOperationName(source), 0);
    }
  }
}

static void insertDependencies(const MemoryDependenceResult &results,
                               ImplicitLocOpBuilder builder) {

  for (const auto &val : results) {
    auto destNameRef = val.first;
    auto destName = builder.getStringAttr(destNameRef);
    auto dependencies = val.second;
    auto dependsOn = builder.create<LoopScheduleDependsOnOp>(destName);
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(dependsOn.getBodyBlock());
    for (auto dep : dependencies) {
      auto sourceNameRef = dep.source;
      auto sourceName = builder.getStringAttr(sourceNameRef);
      auto dist =
          dep.distance > 0 ? builder.getI64IntegerAttr(dep.distance) : nullptr;
      builder.create<LoopScheduleAccessOp>(sourceName, dist);
    }
  }
}

void ConstructMemoryDependenciesPass::runOnOperation() {
  auto funcOp = getOperation();
  auto nameAnalysis = getAnalysis<NameAnalysis>();

  MemoryDependenceResult results;

  // Collect affine loops grouped by nesting depth.
  std::vector<SmallVector<AffineForOp, 2>> depthToLoops;
  mlir::affine::gatherLoops(funcOp, depthToLoops);

  // Collect load and store operations to check.
  SmallVector<Operation *> affineOps;
  funcOp.walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      affineOps.push_back(op);
  });

  // For each depth, check memref accesses.
  for (auto *source : affineOps) {
    for (auto *destination : affineOps) {
      checkAffineAccessPair(source, destination, results, nameAnalysis, false);
      checkAffineAccessPair(source, destination, results, nameAnalysis, true);
    }
  }

  // Collect load and store operations to check.
  SmallVector<Operation *> memrefOps;
  funcOp.walk([&](Operation *op) {
    if (isa<memref::LoadOp, memref::StoreOp, AffineLoadOp, AffineStoreOp,
            SchedulableAffineInterface, StoreInterface, LoadInterface>(op))
      memrefOps.push_back(op);
  });

  // For each depth, check memref accesses.
  for (auto *source : memrefOps) {
    for (auto *destination : memrefOps) {
      checkNonAffineAccessPair(source, destination, results, nameAnalysis,
                               false);
      checkNonAffineAccessPair(source, destination, results, nameAnalysis,
                               true);
    }
  }

  if (results.empty())
    return;

  ImplicitLocOpBuilder builder(funcOp.getLoc(), &getContext());
  builder.setInsertionPoint(funcOp);
  auto depOpNameAttr = builder.getStringAttr(funcOp.getName() + "_deps");
  auto dependencies = builder.create<LoopScheduleDependenciesOp>(depOpNameAttr);
  builder.setInsertionPointToEnd(dependencies.getBodyBlock());

  insertDependencies(results, builder);
  auto symbolName = builder.getAttr<FlatSymbolRefAttr>(depOpNameAttr);
  funcOp->setAttr("loopschedule.dependencies", symbolName);
}

std::unique_ptr<mlir::Pass>
circt::loopschedule::createConstructMemoryDependenciesPass() {
  return std::make_unique<ConstructMemoryDependenciesPass>();
}
