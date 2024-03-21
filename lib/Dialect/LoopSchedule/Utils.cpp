//===- Utils.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "to-loopschedule"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;
using namespace circt;
using namespace circt::loopschedule;
using namespace circt::analysis;
using namespace circt::scheduling;

namespace circt {

namespace loopschedule {

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
/// Also replaces the affine load with the memref load in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineLoadLowering : public OpConversionPattern<AffineLoadOp> {
public:
  AffineLoadLowering(MLIRContext *context,
                     MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands.has_value())
      return failure();

    // Build memref.load memref[expandedMap.results].
    auto memrefLoad = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, op.getMemRef(), *resultOperands);

    dependenceAnalysis.replaceOp(op, memrefLoad);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
/// Also replaces the affine store with the memref store in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineStoreLowering : public OpConversionPattern<AffineStoreOp> {
public:
  AffineStoreLowering(MLIRContext *context,
                      MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap.has_value())
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    auto memrefStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);

    dependenceAnalysis.replaceOp(op, memrefStore);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

class SchedulableAffineReadInterfaceLowering : public mlir::RewritePattern {
public:
  SchedulableAffineReadInterfaceLowering(
      MLIRContext *context, MemoryDependenceAnalysis &dependenceAnalysis)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(*op)) {
      // Expand affine map from 'affineWriteOpInterface'.
      SmallVector<Value, 8> indices(readOp.getMapOperands());
      auto maybeExpandedMap = expandAffineMap(rewriter, readOp.getLoc(),
                                              readOp.getAffineMap(), indices);
      if (!maybeExpandedMap.has_value())
        return failure();

      if (auto schedulableOp = dyn_cast<SchedulableAffineInterface>(*op)) {
        // Build memref.store valueToStore, memref[expandedMap.results].
        auto *newOp =
            schedulableOp.createNonAffineOp(rewriter, *maybeExpandedMap);
        rewriter.replaceOp(op, newOp->getResults());

        dependenceAnalysis.replaceOp(op, newOp);

        return success();
      }
    }
    return failure();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

class SchedulableAffineWriteInterfaceLowering : public mlir::RewritePattern {
public:
  SchedulableAffineWriteInterfaceLowering(
      MLIRContext *context, MemoryDependenceAnalysis &dependenceAnalysis)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context),
        dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (auto writeOp = dyn_cast<AffineWriteOpInterface>(*op)) {
      // Expand affine map from 'affineWriteOpInterface'.
      SmallVector<Value, 8> indices(writeOp.getMapOperands());
      auto maybeExpandedMap = expandAffineMap(rewriter, writeOp.getLoc(),
                                              writeOp.getAffineMap(), indices);
      if (!maybeExpandedMap.has_value())
        return failure();

      if (auto schedulableOp = dyn_cast<SchedulableAffineInterface>(*op)) {
        // Build memref.store valueToStore, memref[expandedMap.results].
        auto *newOp =
            schedulableOp.createNonAffineOp(rewriter, *maybeExpandedMap);
        rewriter.replaceOp(op, newOp->getResults());

        dependenceAnalysis.replaceOp(op, newOp);

        return success();
      }
    }
    return failure();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Helper to hoist computation out of scf::IfOp branches, turning it into a
/// mux-like operation, and exposing potentially concurrent execution of its
/// branches.
struct IfOpHoisting : OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
      }
    });

    return success();
  }
};

/// Helper to determine if an scf::IfOp is in mux-like form.
static bool ifOpLegalityCallback(scf::IfOp op) {
  return op.thenBlock()->without_terminator().empty() &&
         (!op.elseBlock() || op.elseBlock()->without_terminator().empty());
}

/// Helper to mark AffineYieldOp legal, unless it is inside a partially
/// converted scf::IfOp.
static bool yieldOpLegalityCallback(AffineYieldOp op) {
  return !op->getParentOfType<scf::IfOp>();
}

static bool schedulableAffineInterfaceLegalityCallback(Operation *op) {
  return !(
      isa<SchedulableAffineInterface>(*op) &&
      (isa<AffineReadOpInterface>(*op) || isa<AffineWriteOpInterface>(*op)));
}

/// After analyzing memory dependences, and before creating the schedule, we
/// want to materialize affine operations with arithmetic, scf, and memref
/// operations, which make the condition computation of addresses, etc.
/// explicit. This is important so the schedule can consider potentially complex
/// computations in the condition of ifs, or the addresses of loads and stores.
/// The dependence analysis will be updated so the dependences from the affine
/// loads and stores are now on the memref loads and stores.
LogicalResult
lowerAffineStructures(MLIRContext &context, Operation *op,
                      MemoryDependenceAnalysis &dependenceAnalysis) {

  ConversionTarget target(context);
  target.addLegalDialect<AffineDialect, ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();
  target.addIllegalOp<AffineIfOp, AffineLoadOp, AffineStoreOp>();
  target.markUnknownOpDynamicallyLegal(
      schedulableAffineInterfaceLegalityCallback);
  target.addDynamicallyLegalOp<scf::IfOp>(ifOpLegalityCallback);
  target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  auto *ctx = &context;
  RewritePatternSet patterns(ctx);
  patterns.add<AffineLoadLowering>(ctx, dependenceAnalysis);
  patterns.add<AffineStoreLowering>(ctx, dependenceAnalysis);
  patterns.add<IfOpHoisting>(ctx);
  patterns.add<SchedulableAffineReadInterfaceLowering>(ctx, dependenceAnalysis);
  patterns.add<SchedulableAffineWriteInterfaceLowering>(ctx,
                                                        dependenceAnalysis);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  patterns.clear();
  populateAffineToStdConversionPatterns(patterns);
  target.addIllegalOp<AffineApplyOp>();

  return applyPartialConversion(op, target, std::move(patterns));
}

template <typename OpTy>
struct FoldSign : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  static std::optional<IntegerType> operandIsExtended(Value operand) {
    auto *definingOp = operand.getDefiningOp();
    if (!definingOp)
      return std::nullopt;

    if (!isa<IntegerType>(operand.getType()))
      return std::nullopt;

    if (auto extOp = dyn_cast<arith::ExtSIOp>(*definingOp))
      return cast<IntegerType>(extOp->getOperand(0).getType());
    if (auto extOp = dyn_cast<arith::ExtUIOp>(*definingOp))
      return cast<IntegerType>(extOp->getOperand(0).getType());

    return std::nullopt;
  }

  static std::optional<IntegerType>
  valIsTruncated(TypedValue<IntegerType> val) {
    if (!val.hasOneUse())
      return std::nullopt;
    auto *op = *val.getUsers().begin();
    if (auto trunc = dyn_cast<arith::TruncIOp>(*op))
      if (auto truncType = dyn_cast<IntegerType>(trunc.getType()))
        return truncType;

    return std::nullopt;
  }

  static bool opIsLegal(OpTy op) {
    if (op->getNumResults() != 1)
      return true;
    if (op->getNumOperands() <= 0)
      return true;
    if (!isa<IntegerType>(op->getResultTypes().front()))
      return true;

    auto outType =
        valIsTruncated(cast<TypedValue<IntegerType>>(op->getResult(0)));
    if (!outType.has_value())
      return true;

    auto operandType = operandIsExtended(op->getOperand(0));
    if (!operandType.has_value() || operandType != outType)
      return true;

    // Extension and trunc should be opt away
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      auto oW = operandIsExtended(operand);
      return oW != operandType;
    });
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (opIsLegal(op))
      return failure();

    auto outType =
        valIsTruncated(cast<TypedValue<IntegerType>>(op->getResult(0)));

    // Extension and trunc should be opt away
    SmallVector<Value> operands;
    for (auto operand : op->getOperands())
      operands.push_back(operand.getDefiningOp()->getOperand(0));

    SmallVector<Type> resultTypes = {*outType};
    auto newOp = rewriter.create<OpTy>(op.getLoc(), resultTypes, operands);
    auto trunc = *op->getUsers().begin();
    rewriter.replaceAllUsesWith(trunc->getResult(0), newOp->getResult(0));
    rewriter.eraseOp(trunc);
    rewriter.eraseOp(op);

    return success();
  }
};

struct MulStrengthReduction : OpConversionPattern<MulIOp> {
  using OpConversionPattern<MulIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MulIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getRhs()))
      return failure();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto log = val.getValue().exactLogBase2();
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), log);
        auto shift = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::ShLIOp>(op, op.getLhs(),
                                                   shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

struct RemUIStrengthReduction : OpConversionPattern<RemUIOp> {
  using OpConversionPattern<RemUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getRhs()))
      return failure();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto shifted = val.getValue() - 1;
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), shifted);
        auto shift = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::AndIOp>(op, op.getLhs(),
                                                   shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

struct RemSIStrengthReduction : OpConversionPattern<RemSIOp> {
  using OpConversionPattern<RemSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RemSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::RemUIOp>(op, op.getLhs(), op.getRhs());

    return success();
  }
};

struct DivSIStrengthReduction : OpConversionPattern<DivSIOp> {
  using OpConversionPattern<DivSIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DivSIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<BlockArgument>(op.getRhs()))
      return failure();
    auto *rhsDef = op.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto val = cast<IntegerAttr>(constOp.getValue());
      if (llvm::isPowerOf2_32(val.getInt())) {
        auto log = val.getValue().exactLogBase2();
        auto attr = rewriter.getIntegerAttr(op.getRhs().getType(), log);
        auto shift = rewriter.create<arith::ConstantOp>(op.getLoc(), attr);
        rewriter.replaceOpWithNewOp<arith::ShRUIOp>(op, op.getLhs(),
                                                    shift.getResult());
        return success();
      }
    }

    return failure();
  }
};

static bool mulLegalityCallback(Operation *op) {
  if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
    if (isa<BlockArgument>(mulOp.getRhs()))
      return true;
    auto *rhsDef = mulOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
    return FoldSign<arith::MulIOp>::opIsLegal(mulOp);
  }
  return true;
}

static bool divSIOpLegalityCallback(Operation *op) {
  if (auto divOp = dyn_cast<arith::DivSIOp>(op)) {
    if (isa<BlockArgument>(divOp.getRhs()))
      return true;
    auto *rhsDef = divOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool remUILegalityCallback(Operation *op) {
  if (auto remOp = dyn_cast<arith::RemUIOp>(op)) {
    if (isa<BlockArgument>(remOp.getRhs()))
      return true;
    auto *rhsDef = remOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      if (cast<IntegerAttr>(constOp.getValue()).getValue().exactLogBase2() !=
          -1) {
        return false;
      }
    }
  }
  return true;
}

static bool remSILegalityCallback(Operation *op) {
  if (auto remOp = dyn_cast<arith::RemSIOp>(op)) {
    if (isa<BlockArgument>(remOp.getRhs()))
      return true;
    auto *rhsDef = remOp.getRhs().getDefiningOp();

    if (auto constOp = dyn_cast<arith::ConstantOp>(rhsDef)) {
      auto rhsValue = cast<IntegerAttr>(constOp.getValue());
      if (rhsValue.getValue().exactLogBase2() != -1) {
        if (rhsValue.getInt() >= 0)
          return false;
      }
    }
  }
  return true;
}

LogicalResult postLoweringOptimizations(mlir::MLIRContext &context,
                                        mlir::Operation *op) {
  // llvm::errs() << "post lowering opt\n";
  // op->getParentOfType<ModuleOp>().dump();
  ConversionTarget target(context);
  target.addLegalDialect<AffineDialect, ArithDialect, memref::MemRefDialect,
                         scf::SCFDialect>();

  auto *ctx = &context;
  RewritePatternSet patterns(ctx);

  patterns.add<FoldSign<arith::AddIOp>>(ctx);
  patterns.add<FoldSign<arith::SubIOp>>(ctx);
  patterns.add<FoldSign<arith::MulIOp>>(ctx);
  patterns.add<MulStrengthReduction>(ctx);
  patterns.add<DivSIStrengthReduction>(ctx);
  patterns.add<RemUIStrengthReduction>(ctx);
  patterns.add<RemSIStrengthReduction>(ctx);

  target.addDynamicallyLegalOp<AddIOp>(FoldSign<AddIOp>::opIsLegal);
  target.addDynamicallyLegalOp<SubIOp>(FoldSign<SubIOp>::opIsLegal);
  target.addDynamicallyLegalOp<MulIOp>(mulLegalityCallback);
  target.addDynamicallyLegalOp<DivSIOp>(divSIOpLegalityCallback);
  target.addDynamicallyLegalOp<RemUIOp>(remUILegalityCallback);
  target.addDynamicallyLegalOp<RemSIOp>(remSILegalityCallback);
  target.addLegalOp<LoopScheduleLoadOp>();
  target.addLegalOp<LoopScheduleStoreOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  // Loop invariant code motion to hoist produced constants out of loop
  op->walk(
      [&](LoopLikeOpInterface loopLike) { moveLoopInvariantCode(loopLike); });

  mlir::IRRewriter rewriter(&context);
  (void)mlir::runRegionDCE(rewriter, op->getRegions());

  return success();
}

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

bool hasMemoryDependence(Operation *op, Operation *otherOp) {
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

ModuloProblem getModuloProblem(affine::AffineForOp forOp,
                               MemoryDependenceAnalysis &dependenceAnalysis) {
  // Create a modulo scheduling problem.
  ModuloProblem problem = ModuloProblem::get(forOp);

  // Insert memory dependences into the problem.
  forOp.getBody()->walk([&](Operation *op) {
    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences =
        dependenceAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (MemoryDependence memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;

      // Use the lower bound of the innermost loop for this dependence. This
      // assumes outer loops execute sequentially, i.e. one iteration of the
      // inner loop completes before the next iteration is initiated. With
      // proper analysis and lowerings, this can be relaxed.
      unsigned distance = *memoryDep.dependenceComponents.back().lb;
      if (distance > 0)
        problem.setDistance(dep, distance);
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = forOp.getBody()->getTerminator();
  forOp.getBody()->walk([&](Operation *op) {
    if (op == anchor || !problem.hasOperation(op))
      return;
    Problem::Dependence dep(op, anchor);
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
        Problem::Dependence dep(iterArgDefiner, iterArgUser);
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
getSharedOperatorsProblem(affine::AffineForOp forOp,
                          MemoryDependenceAnalysis &dependenceAnalysis) {
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
            Problem::Dependence dep(definingOp, op);
            auto depInserted = problem.insertDependence(dep);
            assert(succeeded(depInserted));
            (void)depInserted;
          }
        }
      });
    }

    ArrayRef<MemoryDependence> dependences =
        dependenceAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (const MemoryDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;

      assert(memoryDep.source != nullptr);
      if (!forOp->isAncestor(memoryDep.source))
        continue;

      // Do not consider inter-iteration deps for seq loops
      auto distance = memoryDep.dependenceComponents.back().lb;
      if (distance.has_value())
        continue;

      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  assert(forOp.getLoopRegions().size() == 1);
  auto *anchor = forOp.getLoopRegions().front()->back().getTerminator();
  forOp.getLoopRegions().front()->walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;
    if (!isa<AffineStoreOp, memref::StoreOp, StoreInterface>(op))
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

SharedOperatorsProblem
getSharedOperatorsProblem(func::FuncOp funcOp,
                          MemoryDependenceAnalysis &dependenceAnalysis) {
  SharedOperatorsProblem problem = SharedOperatorsProblem::get(funcOp);

  // Insert memory dependences into the problem.
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;

    // Insert every operation into the problem.
    problem.insertOperation(op);

    ArrayRef<MemoryDependence> dependences =
        dependenceAnalysis.getDependences(op);
    if (dependences.empty())
      return;

    for (const MemoryDependence &memoryDep : dependences) {
      // Don't insert a dependence into the problem if there is no dependence.
      if (!hasDependence(memoryDep.dependenceType))
        continue;
      if (!funcOp->isAncestor(memoryDep.source))
        continue;
      if (memoryDep.dependenceComponents.back().lb.has_value())
        continue;
      // Insert a dependence into the problem.
      Problem::Dependence dep(memoryDep.source, op);
      auto depInserted = problem.insertDependence(dep);
      assert(succeeded(depInserted));
      (void)depInserted;
    }
  });

  // Set the anchor for scheduling. Insert dependences from all stores to the
  // terminator to ensure the problem schedules them before the terminator.
  auto *anchor = funcOp.getBody().back().getTerminator();
  funcOp.getBody().walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr)
      return;
    Problem::Dependence dep(op, anchor);
    auto depInserted = problem.insertDependence(dep);
    assert(succeeded(depInserted));
    (void)depInserted;
  });

  return problem;
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult
populateOperatorTypes(Operation *op, Region &loopBody,
                      SharedOperatorsProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType loopOpr = problem.getOrInsertOperatorType("loop");
  problem.setLatency(loopOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 4);

  Operation *unsupported;
  WalkResult result = loopBody.walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr) {
      return WalkResult::advance();
    }

    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<scf::IfOp, AffineYieldOp, arith::ConstantOp, arith::ExtSIOp,
              arith::ExtUIOp, arith::TruncIOp, CmpIOp, IndexCastOp,
              memref::AllocaOp, memref::AllocOp, loopschedule::AllocInterface,
              scf::YieldOp, func::ReturnOp, arith::SelectOp, AddIOp, SubIOp, CmpIOp,
              ShLIOp, AndIOp, ShRSIOp, ShRUIOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<LoopInterface>([&](Operation *loopOp) {
          // llvm::errs() << "loopOp\n";
          // loopOp->dump();
          // llvm::errs() << "loopOp after\n";
          problem.setLinkedOperatorType(loopOp, loopOpr);
          auto loop = cast<LoopInterface>(loopOp);
          loop.getBodyBlock()->walk([&](Operation *op) {
            if (isa<AffineLoadOp, AffineStoreOp, LoopScheduleLoadOp,
                    LoopScheduleStoreOp>(op)) {
              Value memRef = getMemref(op);
              Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
                  "mem_" + std::to_string(hash_value(memRef)));
              problem.setLatency(memOpr, 1);
              // External memories are 1 RW port
              problem.setLimit(memOpr, 1);
              problem.addExtraLimitingType(loopOp, memOpr);
            } else if (isa<LoadInterface>(op)) {
              auto loadOp = cast<loopschedule::LoadInterface>(*op);
              auto latency = loadOp.getLatency();
              auto limitOpt = loadOp.getLimit();
              Problem::OperatorType portOpr =
                  problem.getOrInsertOperatorType(loadOp.getUniqueId());
              problem.setLatency(portOpr, latency);
              if (limitOpt.has_value())
                problem.setLimit(portOpr, limitOpt.value());
              problem.addExtraLimitingType(loopOp, portOpr);
            } else if (isa<StoreInterface>(op)) {
              auto storeOp = cast<loopschedule::StoreInterface>(*op);
              auto latency = storeOp.getLatency();
              auto limitOpt = storeOp.getLimit();
              Problem::OperatorType portOpr =
                  problem.getOrInsertOperatorType(storeOp.getUniqueId());
              problem.setLatency(portOpr, latency);
              if (limitOpt.has_value())
                problem.setLimit(portOpr, limitOpt.value());
              problem.addExtraLimitingType(loopOp, portOpr);
            }
          });
          return WalkResult::advance();
        })
        .Case<LoopScheduleStoreOp, AffineStoreOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<LoopScheduleStoreOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLimit(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<LoopScheduleLoadOp, AffineLoadOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<LoopScheduleLoadOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLimit(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          return WalkResult::advance();
        })
        .Case<loopschedule::LoadInterface>([&](Operation *op) {
          auto loadOp = cast<loopschedule::LoadInterface>(*op);
          auto latency = loadOp.getLatency();
          auto limitOpt = loadOp.getLimit();
          Problem::OperatorType portOpr =
              problem.getOrInsertOperatorType(loadOp.getUniqueId());
          problem.setLatency(portOpr, latency);
          if (limitOpt.has_value())
            problem.setLimit(portOpr, limitOpt.value());
          problem.setLinkedOperatorType(op, portOpr);

          return WalkResult::advance();
        })
        .Case<loopschedule::StoreInterface>([&](Operation *op) {
          auto storeOp = cast<loopschedule::StoreInterface>(*op);
          auto latency = storeOp.getLatency();
          auto limitOpt = storeOp.getLimit();
          Problem::OperatorType portOpr =
              problem.getOrInsertOperatorType(storeOp.getUniqueId());
          problem.setLatency(portOpr, latency);
          if (limitOpt.has_value())
            problem.setLimit(portOpr, limitOpt.value());
          problem.setLinkedOperatorType(op, portOpr);

          return WalkResult::advance();
        })
        .Case<loopschedule::SchedulableInterface>([&](Operation *op) {
          auto schedOp = cast<SchedulableInterface>(op);
          auto latency = schedOp.getOpLatency();
          auto limitOpt = schedOp.getOpLimit();
          Problem::OperatorType opr =
              problem.getOrInsertOperatorType(schedOp.getUniqueId());
          problem.setLatency(opr, latency);
          if (limitOpt.has_value())
            problem.setLimit(opr, limitOpt.value());
          problem.setLinkedOperatorType(op, opr);

          return WalkResult::advance();
        })
        .Case<MulIOp, RemUIOp, RemSIOp, DivSIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return op->emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult solveModuloProblem(AffineForOp &loop,
                                 ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loop;

  LLVM_DEBUG(forOp.dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(for (auto *op
                  : problem.getOperations()) {
    if (auto parent = op->getParentOfType<LoopInterface>(); parent)
      continue;
    llvm::dbgs() << "Modulo scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    llvm::dbgs() << "\n  limit = " << problem.getLimit(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  });

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto parent = op->getParentOfType<LoopInterface>(); parent)
        return;
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult solveSharedOperatorsProblem(
    Region &region, SharedOperatorsProblem &problem) {

  LLVM_DEBUG(region.getParentOp()->dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto parent = op->getParentOfType<LoopInterface>(); parent)
      return;
    llvm::dbgs() << "Shared Operator scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr;
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    llvm::dbgs() << "\n  limit = " << problem.getLimit(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { "
                     << "source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = region.back().getTerminator();
  if (failed(scheduleSimplex(problem, anchor)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    region.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto parent = op->getParentOfType<LoopInterface>(); parent)
        return;
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

DenseMap<int64_t, SmallVector<Operation *>>
getOperationCycleMap(Problem &problem) {
  DenseMap<int64_t, SmallVector<Operation *>> map;

  for (auto *op : problem.getOperations()) {
    auto cycleOpt = problem.getStartTime(op);
    assert(cycleOpt.has_value());
    auto cycle = cycleOpt.value();
    auto vec = map.lookup(cycle);
    vec.push_back(op);
    map.insert(std::pair(cycle, vec));
  }

  return map;
}

LogicalResult unrollSubLoops(AffineForOp &forOp) {
  auto result = forOp.getBody()->walk<WalkOrder::PostOrder>([](AffineForOp op) {
    if (loopUnrollFull(op).failed())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    forOp.emitOpError("Could not unroll sub loops");
    return failure();
  }

  return success();
}

struct ReplaceMemrefLoad : OpConversionPattern<memref::LoadOp> {
public:
  ReplaceMemrefLoad(MLIRContext *context,
                    MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<LoopScheduleLoadOp>(
        op.getOperation(), op.getResult().getType(), op.getMemRef(),
        op.getIndices());
    dependenceAnalysis.replaceOp(op.getOperation(), newOp.getOperation());
    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

struct ReplaceMemrefStore : OpConversionPattern<memref::StoreOp> {
public:
  ReplaceMemrefStore(MLIRContext *context,
                     MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<LoopScheduleStoreOp>(
        op.getOperation(), op.getValueToStore(), op.getMemRef(),
        op.getIndices());
    dependenceAnalysis.replaceOp(op.getOperation(), newOp.getOperation());
    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

LogicalResult
replaceMemoryAccesses(mlir::MLIRContext &context, mlir::Operation *op,
                      analysis::MemoryDependenceAnalysis &dependenceAnalysis) {
  ConversionTarget target(context);
  target.addLegalDialect<AffineDialect, ArithDialect, LoopScheduleDialect,
                         scf::SCFDialect>();
  target.addIllegalOp<memref::LoadOp>();
  target.addIllegalOp<memref::StoreOp>();

  auto *ctx = &context;
  RewritePatternSet patterns(ctx);

  patterns.add<ReplaceMemrefLoad>(ctx, dependenceAnalysis);
  patterns.add<ReplaceMemrefStore>(ctx, dependenceAnalysis);

  return applyPartialConversion(op, target, std::move(patterns));
}

struct AffineForIterationReduction : OpRewritePattern<AffineForOp> {
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasConstantBounds()) {
      return failure();
    }

    auto upperBound = op.getConstantUpperBound();
    auto bitwidth = llvm::Log2_64_Ceil(upperBound + 1);

    auto induction = op.getInductionVar();
    if (induction.getType() == rewriter.getIntegerType(bitwidth))
      return failure();
    induction.setType(rewriter.getIntegerType(bitwidth));
    rewriter.setInsertionPointToStart(&op.getRegion().front());
    auto newExt = rewriter.create<arith::ExtSIOp>(
        op.getLoc(), rewriter.getI64Type(), induction);
    // auto newCast = rewriter.create<arith::IndexCastOp>(op.getLoc(),
    // rewriter.getIndexType(), newExt.getOut());
    rewriter.replaceAllUsesExcept(induction, newExt.getOut(), newExt);
    return success();
  }
};

bool insideScheduledLoop(Operation *op) {
  return op->getParentOfType<LoopSchedulePipelineOp>() != nullptr ||
         op->getParentOfType<LoopScheduleSequentialOp>() != nullptr;
}

struct TruncCleanupPattern : OpRewritePattern<TruncIOp> {
  using OpRewritePattern<TruncIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncIOp op,
                                PatternRewriter &rewriter) const override {
    if (insideScheduledLoop(op) || isa<BlockArgument>(op.getIn()))
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
    if (insideScheduledLoop(op))
      return failure();
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
    if (insideScheduledLoop(op))
      return failure();
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
    if (insideScheduledLoop(op))
      return failure();
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
    if (insideScheduledLoop(op))
      return failure();
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
    if (insideScheduledLoop(op))
      return failure();
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
    if (insideScheduledLoop(op))
      return failure();
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
                             MemoryDependenceAnalysis &dependenceAnalysis)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, 1, context),
        dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (insideScheduledLoop(op))
      return failure();
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
  MemoryDependenceAnalysis &dependenceAnalysis;
};

struct ConstIndexRemovalRewritePattern final
    : public OpConversionPattern<ConstantOp> {
public:
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (insideScheduledLoop(op))
      return failure();
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
    if (insideScheduledLoop(op))
      return failure();
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, op.getOut().getType(), op.getIn());

    return success();
  }
};

LogicalResult
bitwidthMinimization(mlir::MLIRContext &context, mlir::Operation *op,
                     analysis::MemoryDependenceAnalysis &dependenceAnalysis) {
  // Minimize AffineFor iteration argument bitwidth to enable further bitwidth
  // reduction
  RewritePatternSet patterns(&context);
  patterns.add<AffineForIterationReduction>(&context);

  op->getParentOfType<ModuleOp>().dump();

  SmallVector<Operation *> opsToSimplify;
  if (isa<AffineForOp>(op))
    opsToSimplify.push_back(op);
  op->walk([&](Operation *op) {
    if (isa<AffineForOp>(op)) {
      opsToSimplify.push_back(op);
    }
  });

  GreedyRewriteConfig config;
  if (failed(applyOpPatternsAndFold(opsToSimplify, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  // Remove index types to enable bitwidth reduction of indices
  TypeConverter typeConverter;
  populateIndexRemovalTypeConverter(typeConverter);
  ConversionTarget target(context);
  target.addDynamicallyLegalDialect<ArithDialect>(
      [&typeConverter](Operation *op) { return insideScheduledLoop(op) || 
                                               typeConverter.isLegal(op); });
  target.addDynamicallyLegalDialect<LoopScheduleDialect>(
      [&typeConverter](Operation *op) { return insideScheduledLoop(op) ||
                                               typeConverter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal(
      [&typeConverter](Operation *op) -> std::optional<bool> {
        if (!isa<LoadInterface>(op) && !isa<StoreInterface>(op))
          return std::nullopt;
        return insideScheduledLoop(op) || typeConverter.isLegal(op);
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

  op->getParentOfType<ModuleOp>().dump();

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);

  opsToSimplify.clear();
  op->walk([&](Operation *op) {
    if (isa<TruncIOp, LoopScheduleLoadOp, LoopScheduleStoreOp>(op)) {
      opsToSimplify.push_back(op);
    }
  });

  if (failed(applyOpPatternsAndFold(opsToSimplify, std::move(patterns), config))) {
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

  opsToSimplify.clear();
  op->walk([&](Operation *op) {
    if (isa<ArithDialect>(op->getDialect())) {
      opsToSimplify.push_back(op);
    }
  });

  if (failed(applyOpPatternsAndFold(opsToSimplify, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  // Cleanup extraneous casts after int narrowing
  patterns.clear();
  patterns.add<TruncCleanupPattern>(&context);
  patterns.add<LoadCleanupPattern>(&context);
  patterns.add<StoreCleanupPattern>(&context);
  patterns.add<LoadAddressNarrowingPattern>(&context);
  patterns.add<LoadInterfaceCleanupPattern>(&context);
  patterns.add<StoreInterfaceCleanupPattern>(&context);
  patterns.add<LoadInterfaceAddressNarrowingPattern>(&context);

  opsToSimplify.clear();
  op->walk([&](Operation *op) {
    if (isa<TruncIOp, LoopScheduleLoadOp, LoopScheduleStoreOp, 
            LoadInterface, StoreInterface>(op)) {
      opsToSimplify.push_back(op);
    }
  });

  if (failed(applyOpPatternsAndFold(opsToSimplify, std::move(patterns), config))) {
    op->emitOpError("Failed to perform bitwidth minimization conversions");
    return failure();
  }

  // Perform dead code elimination again before scheduling
  mlir::IRRewriter rewriter(&context);
  (void)mlir::runRegionDCE(rewriter, op->getRegions());

  return success();
}

} // namespace loopschedule
} // namespace circt
