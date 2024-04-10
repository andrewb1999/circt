//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace circt;
using namespace circt::loopschedule;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace circt {

namespace loopschedule {

Value getMemref(Operation *op) {
  Value memref =
      isa<AffineStoreOp>(*op)     ? cast<AffineStoreOp>(*op).getMemRef()
      : isa<AffineLoadOp>(*op)    ? cast<AffineLoadOp>(*op).getMemRef()
      : isa<memref::StoreOp>(*op) ? cast<memref::StoreOp>(*op).getMemRef()
      : isa<memref::LoadOp>(*op)  ? cast<memref::LoadOp>(*op).getMemRef()
      : isa<LoopScheduleLoadOp>(*op)
          ? cast<LoopScheduleLoadOp>(*op).getMemRef()
          : cast<LoopScheduleStoreOp>(*op).getMemRef();
  return memref;
}

bool oneIsStore(Operation *op, Operation *otherOp) {
  auto firstIsStore = isa<AffineStoreOp, memref::StoreOp, StoreInterface>(*op);
  auto secondIsStore =
      isa<AffineStoreOp, memref::StoreOp, StoreInterface>(*otherOp);
  return firstIsStore || secondIsStore;
}

bool hasLoopScheduleDependence(Operation *op, Operation *otherOp) {
  if (isa<LoadInterface, StoreInterface>(op)) {
    if (!isa<LoadInterface, StoreInterface>(otherOp)) {
      return false;
    }

    if (auto load = dyn_cast<LoadInterface>(op)) {
      return load.hasDependence(otherOp);
    }

    auto store = dyn_cast<StoreInterface>(op);
    return store.hasDependence(otherOp);
  }

  if (isa<LoadInterface, StoreInterface>(otherOp)) {
    return false;
  }

  auto memref = getMemref(op);
  auto otherMemref = getMemref(otherOp);
  return memref == otherMemref && oneIsStore(op, otherOp);
}

ModuloProblem
getModuloProblem(scf::ForOp forOp,
                 LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  // Create a modulo scheduling problem.
  ModuloProblem problem = ModuloProblem::get(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (LoopScheduleDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      unsigned distance = memoryDep.distance;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  problem.insertOperation(anchor);
  forOp.getBody()->walk([&](Operation *op) {
    if (op == anchor || !problem.hasOperation(op))
      return;
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  // Handle explicitly computed loop-carried values, i.e. excluding the
  // induction variable. Insert inter-iteration dependences from the definers of
  // "iter_args" to their users.
  if (unsigned nIterArgs = anchor->getNumOperands(); nIterArgs > 0) {
    auto iterArgs = forOp.getRegionIterArgs();
    for (unsigned i = 0; i < nIterArgs; ++i) {
      Operation *iterArgDefiner = anchor->getOperand(i).getDefiningOp();
      // If it's not an operation, we don't need to model the dependence.
      if (!iterArgDefiner)
        continue;

      for (Operation *iterArgUser : iterArgs[i].getUsers()) {
        Dependence dep(iterArgDefiner, iterArgUser);
        auto depInserted = problem.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;

        // Values always flow between subsequent iterations.
        problem.setDistance(dep, 1);
      }
    }
  }

  return problem;
}

SharedOperatorsProblem
getSharedOperatorsProblem(scf::ForOp forOp,
                          LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(forOp);

  // Insert memory dependences into the problem.
  assert(forOp.getLoopRegions().size() == 1);
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopInterface>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    if (auto loop = dyn_cast<LoopInterface>(op)) {
      loop.getBodyBlock()->walk([&](Operation *innerOp) {
        for (auto &operand : innerOp->getOpOperands()) {
          auto *definingOp = operand.get().getDefiningOp();
          if (definingOp && definingOp->getParentOp() == forOp) {
            Dependence dep(definingOp, op);
            auto depInserted = problem.insertDependence(dep);
            assert(succeeded(depInserted));
            (void)depInserted;
          }
        }
      });
    }

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (const LoopScheduleDependence &memoryDep : dependences) {
      assert(memoryDep.source != nullptr);
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Do not consider inter-iteration deps for seq loops
      auto distance = memoryDep.distance;
      if (distance > 0)
        continue;

      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  assert(forOp.getLoopRegions().size() == 1);
  auto *anchor = forOp.getLoopRegions().front()->back().getTerminator();
  problem.insertOperation(anchor);
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
        !problem.hasOperation(op))
      return;
    if (!isa<AffineStoreOp, memref::StoreOp, StoreInterface>(op))
      return;
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

SharedOperatorsProblem
getSharedOperatorsProblem(func::FuncOp funcOp,
                          LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(funcOp);

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<LoopScheduleDependence> dependences =
        dependenceAnalysis.getDependencies(op);
    if (dependences.empty())
      return;

    for (const LoopScheduleDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!funcOp->isAncestor(memoryDep.source))
        continue;
      if (memoryDep.distance > 0)
        continue;
      // Insert a dependence into the problem.
      Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = funcOp.getBody().back().getTerminator();
  problem.insertOperation(anchor);
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
        !problem.hasOperation(op))
      return;
    Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

LogicalResult unrollSubLoops(scf::ForOp &forOp) {
  auto result = forOp.getBody()->walk<WalkOrder::PostOrder>([](scf::ForOp op) {
    std::optional<int64_t> lbCstOp = getConstantIntValue(op.getLowerBound());
    std::optional<int64_t> ubCstOp = getConstantIntValue(op.getUpperBound());
    std::optional<int64_t> stepCstOp = getConstantIntValue(op.getStep());
    if (!lbCstOp || !ubCstOp || !stepCstOp) {
      return WalkResult::interrupt();
    }
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);
    if (loopUnrollByFactor(op, tripCount).failed())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    forOp.emitOpError("Could not unroll sub loops");
    return failure();
  }

  return success();
}

struct SCFForIterationReduction : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    auto constantLB = op.getLowerBound().getDefiningOp<ConstantOp>();
    auto constantUB = op.getUpperBound().getDefiningOp<ConstantOp>();
    auto constantStep = op.getStep().getDefiningOp<ConstantOp>();
    if (constantLB == nullptr || constantUB == nullptr ||
        constantStep == nullptr) {
      auto inductionType = op.getInductionVar().getType();
      auto bitwidth = isa<IndexType>(inductionType)
                          ? 64
                          : inductionType.getIntOrFloatBitWidth();
      auto newType = rewriter.getIntegerType(bitwidth);
      if (op.getInductionVar().getType() == newType)
        return failure();

      op.getInductionVar().setType(newType);
      rewriter.setInsertionPointToStart(&op.getRegion().front());
      auto newExt = rewriter.create<arith::ExtSIOp>(
          op.getLoc(), rewriter.getI64Type(), op.getInductionVar());
      rewriter.replaceAllUsesExcept(op.getInductionVar(), newExt.getOut(),
                                    newExt);
      return success();
    }

    auto upperBoundAttr = dyn_cast<IntegerAttr>(constantUB.getValue());
    auto upperBound = upperBoundAttr.getValue();
    upperBound = upperBound + 1;
    auto bitwidth = upperBound.ceilLogBase2();

    auto induction = op.getInductionVar();
    auto newType = rewriter.getIntegerType(bitwidth);
    if (induction.getType() == newType)
      return failure();

    // Replace lowerBound, upperBound and step
    auto lbValue = cast<IntegerAttr>(constantLB.getValue()).getInt();
    auto newLBAttr = rewriter.getIntegerAttr(newType, lbValue);
    auto newLB = rewriter.create<ConstantOp>(op.getLoc(), newLBAttr);
    op.setLowerBound(newLB);

    auto ubValue = cast<IntegerAttr>(constantUB.getValue()).getInt();
    auto newUBAttr = rewriter.getIntegerAttr(newType, ubValue);
    auto newUB = rewriter.create<ConstantOp>(op.getLoc(), newUBAttr);
    op.setUpperBound(newUB);

    auto stepValue = cast<IntegerAttr>(constantStep.getValue()).getInt();
    auto newStepAttr = rewriter.getIntegerAttr(newType, stepValue);
    auto newStep = rewriter.create<ConstantOp>(op.getLoc(), newStepAttr);
    op.setStep(newStep);

    induction.setType(newType);
    rewriter.setInsertionPointToStart(&op.getRegion().front());
    auto newExt = rewriter.create<arith::ExtSIOp>(
        op.getLoc(), rewriter.getI64Type(), induction);
    // auto newCast = rewriter.create<arith::IndexCastOp>(op.getLoc(),
    // rewriter.getIndexType(), newExt.getOut());
    rewriter.replaceAllUsesExcept(induction, newExt.getOut(), newExt);
    return success();
  }
};

struct SCFForCleanupPattern : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    auto cast = op.getLowerBound().getDefiningOp<UnrealizedConversionCastOp>();
    if (cast != nullptr && cast.getInputs().size() == 1) {
      auto newVal = cast.getInputs().front();
      op.setLowerBound(newVal);
      changed = true;
    }

    cast = op.getUpperBound().getDefiningOp<UnrealizedConversionCastOp>();
    if (cast != nullptr && cast.getInputs().size() == 1) {
      auto newVal = cast.getInputs().front();
      op.setUpperBound(newVal);
      changed = true;
    }

    cast = op.getStep().getDefiningOp<UnrealizedConversionCastOp>();
    if (cast != nullptr && cast.getInputs().size() == 1) {
      auto newVal = cast.getInputs().front();
      op.setStep(newVal);
      changed = true;
    }

    if (!changed)
      return failure();
    return success();
  }
};

struct TruncCleanupPattern : OpRewritePattern<TruncIOp> {
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncIOp op,
                                PatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getIn()))
      return failure();
    auto *definingOp = op.getIn().getDefiningOp();
    auto outputType = op.getOut().getType();
    Operation *newOp = nullptr;
    if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
      newOp = rewriter.create<ExtUIOp>(op.getLoc(), outputType, extUI.getIn());
    } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
      newOp = rewriter.create<ExtSIOp>(op.getLoc(), outputType, extSI.getIn());
    }

    if (!newOp)
      return failure();

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct LoadCleanupPattern : OpRewritePattern<LoopScheduleLoadOp> {
  using OpRewritePattern<LoopScheduleLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct StoreCleanupPattern : OpRewritePattern<LoopScheduleStoreOp> {
  using OpRewritePattern<LoopScheduleStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct LoadAddressNarrowingPattern : OpRewritePattern<LoopScheduleLoadOp> {
  using OpRewritePattern<LoopScheduleLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoopScheduleLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto v : llvm::enumerate(indices)) {
      auto &idx = v.value();
      auto i = v.index();
      auto dimSize = op.getMemRefType().getDimSize(i);
      auto bitwidth = llvm::Log2_64_Ceil(dimSize);
      auto newType = rewriter.getIntegerType(bitwidth);
      auto oldType = dyn_cast_or_null<IntegerType>(idx.get().getType());
      if (oldType) {
        if (newType.getIntOrFloatBitWidth() < oldType.getIntOrFloatBitWidth()) {
          auto newIdx =
              rewriter.create<arith::TruncIOp>(op.getLoc(), newType, idx.get());
          idx.set(newIdx);
          updated = true;
        }
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct LoadInterfaceCleanupPattern : OpInterfaceRewritePattern<LoadInterface> {
  using OpInterfaceRewritePattern<LoadInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoadInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct StoreInterfaceCleanupPattern
    : OpInterfaceRewritePattern<StoreInterface> {
  using OpInterfaceRewritePattern<StoreInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(StoreInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto &idx : llvm::make_early_inc_range(indices)) {
      if (isa<BlockArgument>(idx.get()))
        continue;
      auto *definingOp = idx.get().getDefiningOp();
      if (auto extUI = dyn_cast<ExtUIOp>(definingOp)) {
        idx.set(extUI.getIn());
        updated = true;
      } else if (auto extSI = dyn_cast<ExtSIOp>(definingOp)) {
        idx.set(extSI.getIn());
        updated = true;
      } else if (auto unreal =
                     dyn_cast<UnrealizedConversionCastOp>(definingOp)) {
        if (unreal.getInputs().size() != 1)
          continue;
        idx.set(unreal.getInputs().front());
        updated = true;
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

struct LoadInterfaceAddressNarrowingPattern
    : OpInterfaceRewritePattern<LoadInterface> {
  using OpInterfaceRewritePattern<LoadInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LoadInterface op,
                                PatternRewriter &rewriter) const override {
    auto indices = op.getIndicesMutable();
    bool updated = false;
    for (auto v : llvm::enumerate(indices)) {
      auto &idx = v.value();
      auto i = v.index();
      auto bitwidth = op.getDimBitwidth(i);
      auto newType = rewriter.getIntegerType(bitwidth);
      auto oldType = dyn_cast_or_null<IntegerType>(idx.get().getType());
      if (oldType) {
        if (newType.getIntOrFloatBitWidth() < oldType.getIntOrFloatBitWidth()) {
          auto newIdx =
              rewriter.create<arith::TruncIOp>(op.getLoc(), newType, idx.get());
          idx.set(newIdx);
          updated = true;
        }
      }
    }

    if (!updated)
      return failure();

    return success();
  }
};

void populateIndexRemovalTypeConverter(TypeConverter &typeConverter) {
  typeConverter.addConversion(
      [](Type type) -> std::optional<Type> { return type; });
  typeConverter.addConversion([](IndexType type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 64);
  });
  typeConverter.addTargetMaterialization(
      [](OpBuilder &b, Type target, ValueRange input, Location loc) {
        return b.create<UnrealizedConversionCastOp>(loc, target, input)
            .getResult(0);
      });
  typeConverter.addSourceMaterialization(
      [](OpBuilder &b, Type source, ValueRange input, Location loc) {
        return b.create<UnrealizedConversionCastOp>(loc, source, input)
            .getResult(0);
      });
}

struct IndexRemovalRewritePattern final : ConversionPattern {
public:
  IndexRemovalRewritePattern(TypeConverter &converter, MLIRContext *context,
                             LoopScheduleDependenceAnalysis &dependenceAnalysis)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, 1, context),
        dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    const TypeConverter *converter = getTypeConverter();
    if (converter->isLegal(op))
      return rewriter.notifyMatchFailure(loc, "op already legal");

    OperationState newOp(loc, op->getName());
    newOp.addOperands(operands);

    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(), newResultTypes)))
      return rewriter.notifyMatchFailure(loc, "couldn't convert return types");
    newOp.addTypes(newResultTypes);
    newOp.addAttributes(op->getAttrs());
    Operation *legalized = rewriter.create(newOp);
    SmallVector<Value> results = legalized->getResults();
    dependenceAnalysis.replaceOp(op, legalized);
    rewriter.replaceOp(op, results);

    return success();
  }

private:
  LoopScheduleDependenceAnalysis &dependenceAnalysis;
};

struct ConstIndexRemovalRewritePattern final
    : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto val = op.getValue();
    if (!isa<IndexType>(val.getType()))
      return failure();

    auto integerAttr = dyn_cast_or_null<IntegerAttr>(val);
    if (!integerAttr)
      return failure();

    auto intVal = integerAttr.getInt();

    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            rewriter.getI64IntegerAttr(intVal));

    return success();
  }
};

struct IndexCastRemovalRewritePattern final
    : public OpConversionPattern<IndexCastOp> {
public:
  using OpConversionPattern<IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getOut().getType(), op.getIn());

    return success();
  }
};

LogicalResult bitwidthMinimization(
    mlir::MLIRContext &context, mlir::Operation *op,
    analysis::LoopScheduleDependenceAnalysis &dependenceAnalysis) {
  // Minimize SCFFor iteration argument bitwidth to enable further bitwidth
  // reduction
  RewritePatternSet patterns(&context);
  patterns.add<SCFForIterationReduction>(&context);

  GreedyRewriteConfig config;
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  // Remove index types to enable bitwidth reduction of indices
  TypeConverter typeConverter;
  populateIndexRemovalTypeConverter(typeConverter);
  ConversionTarget target(context);
  target.addDynamicallyLegalDialect<ArithDialect>(
      [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalDialect<LoopScheduleDialect>(
      [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  // target.addDynamicallyLegalDialect<scf::SCFDialect>(
  //     [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal(
      [&typeConverter](Operation *op) -> std::optional<bool> {
        if (!isa<LoadInterface>(op) && !isa<StoreInterface>(op))
          return std::nullopt;
        return typeConverter.isLegal(op);
      });
  target.addIllegalOp<arith::IndexCastOp>();
  patterns.clear();
  patterns.add<IndexRemovalRewritePattern>(typeConverter, &context,
                                           dependenceAnalysis);
  patterns.add<ConstIndexRemovalRewritePattern>(typeConverter, &context);
  patterns.add<IndexCastRemovalRewritePattern>(typeConverter, &context);
  populateReconcileUnrealizedCastsPatterns(patterns);
  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  // Apply the core integer narrowing pass
  patterns.clear();
  SmallVector<unsigned> bitwidthsSupported;
  for (unsigned i = 1; i <= 128; ++i) {
    bitwidthsSupported.push_back(i);
  }
  populateArithIntNarrowingPatterns(
      patterns, ArithIntNarrowingOptions{bitwidthsSupported});
  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<SCFForCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);
  patterns.add<LoadAddressNarrowingPattern>(&context);
  patterns.add<LoadInterfaceCleanupPattern>(&context);
  patterns.add<StoreInterfaceCleanupPattern>(&context);
  patterns.add<LoadInterfaceAddressNarrowingPattern>(&context);

  if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  auto res = op->walk([](Operation *op) {
    if (isa<UnrealizedConversionCastOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (res.wasInterrupted())
    return op->emitOpError(
        "Bitwidth minimization failed to remove UnrealizedConversionCastOps");
  // Perform dead code elimination again before scheduling
  mlir::IRRewriter rewriter(&context);
  (void)mlir::runRegionDCE(rewriter, op->getRegions());

  return success();
}

} // namespace loopschedule
} // namespace circt
