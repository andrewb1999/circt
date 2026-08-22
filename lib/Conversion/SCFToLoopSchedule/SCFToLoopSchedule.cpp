//===- SCFToLoopSchedule.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToLoopSchedule.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Dialect/SSP/SSPInterfaces.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <math.h>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <utility>

#define DEBUG_TYPE "scf-to-loopschedule"

namespace circt {
#define GEN_PASS_DEF_SCFTOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::affine;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::loopschedule;

namespace {

// Forward declaration; full definition is just below the class. The class
// signature mentions `ScheduleStrategy &` so the type must at least be
// declared at this point.
struct ScheduleStrategy;

struct SCFToLoopSchedulePass
    : public circt::impl::SCFToLoopScheduleBase<SCFToLoopSchedulePass> {
  using SCFToLoopScheduleBase::SCFToLoopScheduleBase;
  void runOnOperation() override;

private:
  LogicalResult runOnFunc(FuncOp funcOp);
  LogicalResult populateOperatorTypes(Operation *op, Region &body,
                                      ChainingSharedOperatorsProblem &problem,
                                      bool sequentialRegion);
  LogicalResult solveChainingModuloProblem(scf::WhileOp &loop,
                                           ChainingModuloProblem &problem,
                                           float cycleTime);
  LogicalResult solveChainingSharedOperatorsProblem(
      Region &region, ChainingSharedOperatorsProblem &problem, float cycleTime);
  LogicalResult createLoopSchedulePipeline(scf::WhileOp &loop,
                                           CyclicProblem &problem,
                                           Value condValue);
  LogicalResult createLoopScheduleSequential(scf::WhileOp &loop,
                                             Problem &problem,
                                             Value condValue);
  LogicalResult createFuncLoopSchedule(FuncOp &funcOp, Problem &problem);

  /// Pipelined function-level lowering: produces a
  /// `loopschedule.func_pipeline` op directly. Day-1 restriction: no nested
  /// loops in the function body.
  LogicalResult createFuncLoopSchedulePipeline(FuncOp funcOp,
                                                ChainingModuloProblem &problem);

  /// Merged driver shared by `createLoopScheduleSequential` and
  /// `createFuncLoopSchedule`. The variant differences are encapsulated in
  /// `strategy`.
  LogicalResult lowerSchedule(ScheduleStrategy &strategy, Problem &problem);

  /// Walk the per-stage `startGroups` and compute the per-value pipe
  /// lifetimes (`pipeTimes`) plus the set of values that must be
  /// registered out of each stage so consumers in later stages can read
  /// them (`registerValues`). Shared between the loop-pipeline and
  /// func-pipeline drivers; the optional `funcReturnSentinel` lets the
  /// func variant treat `loopschedule.return` as an end-of-pipeline
  /// consumer without needing a `problem.getStartTime` entry for it.
  /// `startTimes` should be the sorted set of stage start times to walk.
  void computePipelineStageRegisters(
      ArrayRef<unsigned> startTimes,
      DenseMap<unsigned, SmallVector<Operation *>> &startGroups,
      CyclicProblem &problem, Operation *funcReturnSentinel,
      SmallVectorImpl<SmallVector<Value>> &registerValues,
      DenseMap<Value, std::pair<unsigned, unsigned>> &pipeTimes);

  /// Emit one `loopschedule.at offset(startTime)` stage at the builder's
  /// current insertion point. Clones each scheduled op into the stage,
  /// honouring `predicateMap` for if-guarded ops; updates the per-stage
  /// `valueMap` so later stages read the new values; populates the
  /// stage's terminator with the `registerValuesAtStage` operands and
  /// re-maps each registered value into its consumer-stage's value map.
  /// Returns the created stage so loop-pipeline callers can record its
  /// stage results (e.g. for the condition-result lookup).
  ///
  /// Shared between `createLoopSchedulePipeline` and
  /// `createFuncLoopSchedulePipeline`; the per-driver insertion-point
  /// difference is handled by the caller setting `builder` before the
  /// call.
  LoopScheduleAtOp emitOnePipelineStage(
      ImplicitLocOpBuilder &builder, unsigned startTime,
      ArrayRef<Operation *> group, ArrayRef<Value> registerValuesAtStage,
      ArrayRef<Type> registerTypesAtStage,
      MutableArrayRef<IRMapping> stageValueMaps, CyclicProblem &problem,
      DominanceInfo &dom);

  std::optional<LoopScheduleDependenceAnalysis> dependenceAnalysis;
  std::optional<OperatorLibraryAnalysis> operatorLibraryAnalysis;
  PredicateUse predicateUse;
  PredicateMap predicateMap;
};

/// Does `op` represent a dynamic-latency memory access? We detect via
/// `LoadInterface::isDynamic()` / `StoreInterface::isDynamic()` — ops
/// from downstream dialects (AMC's `amc.load`/`amc.store`) return true
/// when operating on a dynamic port. No attribute marker required.
static bool isDynamicLatencyOp(Operation &op) {
  if (auto load = dyn_cast<LoadInterface>(&op))
    return load.isDynamic();
  if (auto store = dyn_cast<StoreInterface>(&op))
    return store.isDynamic();
  return false;
}

/// Replace each dynamic-latency op inside `pipeline` with a
/// `loopschedule.launch` in its issue stage + a `loopschedule.expect` in
/// the stage `latency` cycles later. Only handles single-result ops for
/// now; multi-result dynamic ops emit a warning and are left alone.
static void wrapDynamicOpsInPipeline(LoopSchedulePipelineOp pipeline,
                                      scheduling::Problem &problem) {
  MLIRContext *ctx = pipeline.getContext();
  auto handleTy = HandleType::get(ctx);

  // Collect dynamic ops grouped by issue stage. The stage op is rebuilt
  // at most once per stage (with the union of handle-typed result slots),
  // so we must accumulate per-stage wrappings before touching IR.
  struct Wrapping {
    Operation *dynOp;
    unsigned latency;
  };
  // Keyed by stage OFFSET, not the `at`-op handle: planting an expect rebuilds
  // (and erases) a destination stage at-op, and that stage may itself be a
  // pending issue stage. Re-resolving by offset each iteration avoids
  // dereferencing a stale handle. The `dynOp` pointers in `Wrapping` stay valid
  // across a rebuild because `takeBody` reparents the op rather than erasing it.
  SmallVector<unsigned> stagesInOrder;
  llvm::SmallDenseMap<unsigned, SmallVector<Wrapping>, 8> byStage;
  // The set of existing stage offsets, used to tell whether an op's completion
  // stage (offset + latency) exists in the pipeline.
  llvm::SmallDenseSet<unsigned> stageOffsets;
  pipeline.walk(
      [&](LoopScheduleAtOp a) { stageOffsets.insert(a.getOffset()); });
  pipeline.walk([&](LoopScheduleAtOp atOp) {
    for (Operation &op : atOp.getBodyBlock().getOperations()) {
      if (!isDynamicLatencyOp(op))
        continue;
      // An explicit fire-and-forget store opts out of the launch/expect stall:
      // it fires and the pipeline continues without waiting for its `done`.
      if (auto store = dyn_cast<StoreInterface>(&op))
        if (!store.getWaitForCompletion())
          continue;
      // Phase 3 wraps single-result dynamic loads (the handle replaces the
      // load's data slot in the issue stage's at-yield) and zero-result
      // dynamic stores (the handle is appended as a fresh slot — there is no
      // result to thread, only a completion to stall on). Multi-result dynamic
      // ops are still unsupported.
      if (op.getNumResults() > 1) {
        op.emitWarning(
            "dynamic op with multiple results is not yet supported; "
            "skipping launch/expect wrap");
        continue;
      }
      // Place the expect at the op's declared latency so several
      // iterations' requests overlap on one port (the FSM attributes the
      // interleaved dones with per-expect counting + a data FIFO). The
      // walk sees CLONES (stage emission), which the problem does not
      // know, so fall back to the load/store interface latency when the
      // operator-type lookup fails.
      unsigned lat = 1;
      if (auto opr = problem.getLinkedOperatorType(&op))
        lat = problem.getLatency(*opr).value_or(1);
      else if (auto load = dyn_cast<LoadInterface>(&op))
        lat = load.getLatency();
      else if (auto store = dyn_cast<StoreInterface>(&op))
        lat = store.getLatency();
      if (lat == 0) {
        // A DECLARED-zero-latency dynamic load with a read-enable is an
        // FWFT stream pop: data is combinational, issue is stall-gated on
        // the port's beat-valid, and skipping the launch/expect wrap is
        // the designed lowering — stay silent. Anything else reaching
        // here is a misconfigured port worth warning about.
        auto hwLoad = dyn_cast<HWLoadLoweringInterface>(&op);
        if (!(hwLoad && hwLoad.getReadLatency() == 0 &&
              hwLoad.requiresReadEnable()))
          op.emitWarning("loopschedule.dynamic op has zero latency; skipping "
                         "launch/expect wrap");
        continue;
      }
      unsigned offset = atOp.getOffset();
      // A zero-result store has no downstream use forcing a later stage, so
      // `offset + lat` may not exist. Its expect doesn't need to sit exactly
      // at completion — the FSM's done counter stalls the planted stage until
      // the done arrives, however late — so clamp to the last existing stage.
      // If no later stage exists at all (a true last-stage store), leave it
      // unwrapped (fire-and-forget) rather than dangling its launch handle;
      // function-level control waits fence it. (Single-result loads always
      // have a later use, so their completion stage exists.)
      if (op.getNumResults() == 0) {
        while (lat > 0 && !stageOffsets.count(offset + lat))
          --lat;
        if (lat == 0)
          continue;
      }
      if (!byStage.count(offset))
        stagesInOrder.push_back(offset);
      byStage[offset].push_back({&op, lat});
    }
  });

  // For each issue stage, perform all launches, rebuild the stage once,
  // then plant expects in destination stages.
  for (unsigned issueOffset : stagesInOrder) {
    // Re-resolve the at op at this offset: a prior iteration's destination-
    // stage rebuild may have replaced it (same offset, fresh handle). Mirrors
    // the destStage lookup below.
    LoopScheduleAtOp issueStage;
    for (auto candidate :
         pipeline.getStagesBlock().getOps<LoopScheduleAtOp>()) {
      if (candidate.getOffset() == issueOffset) {
        issueStage = candidate;
        break;
      }
    }
    if (!issueStage)
      continue;
    auto &wrappings = byStage[issueOffset];

    // Create a launch per dynamic op, moving the op into the launch body.
    // Track each wrapping's launch so we can find the rewired slot below.
    struct LaunchInfo {
      Operation *dynOp;
      LoopScheduleLaunchOp launch;
      unsigned latency;
      unsigned slot = ~0u; // at-yield slot holding the handle
    };
    SmallVector<LaunchInfo> infos;
    infos.reserve(wrappings.size());
    for (auto &w : wrappings) {
      Operation *dynOp = w.dynOp;
      Location loc = dynOp->getLoc();
      OpBuilder b(dynOp);
      auto launch = b.create<LoopScheduleLaunchOp>(loc, handleTy);
      Block &launchBlock = launch.getBody().emplaceBlock();
      dynOp->moveBefore(&launchBlock, launchBlock.begin());
      b.setInsertionPointToEnd(&launchBlock);
      b.create<LoopScheduleYieldOp>(loc, dynOp->getResults());
      infos.push_back({dynOp, launch, w.latency});
    }

    // Rewire the issue stage's at-yield so the launch handle is threaded to
    // the destination stage. A LOAD (one result) replaces the slot carrying
    // its data result in place (index unchanged). A STORE (zero results) has
    // no slot to match, so a fresh handle slot is APPENDED. Loads only
    // overwrite in place and appends happen after the seed copy, so each
    // store's appended index is final and stable through the single rebuild.
    auto stageYield = issueStage.getYieldOp();
    SmallVector<Type> newStageResultTypes(issueStage.getResultTypes());
    for (auto &info : infos) {
      if (info.dynOp->getNumResults() == 1) {
        Value res = info.dynOp->getResult(0);
        for (auto [i, operand] : llvm::enumerate(stageYield->getOperands())) {
          if (operand == res) {
            stageYield->setOperand(i, info.launch.getHandle());
            newStageResultTypes[i] = handleTy;
            info.slot = i;
            break;
          }
        }
      } else {
        info.slot = newStageResultTypes.size();
        newStageResultTypes.push_back(handleTy);
        stageYield->insertOperands(stageYield->getNumOperands(),
                                   info.launch.getHandle());
      }
    }

    // Rebuild the issue stage once with the updated result types.
    LoopScheduleAtOp newStage = issueStage;
    if (newStageResultTypes != SmallVector<Type>(issueStage.getResultTypes())) {
      OpBuilder rebuild(issueStage);
      newStage = rebuild.create<LoopScheduleAtOp>(
          issueStage.getLoc(), newStageResultTypes,
          issueStage.getOffsetAttr());
      newStage.getBody().takeBody(issueStage.getBody());
      for (auto [oldRes, newRes] :
           llvm::zip(issueStage.getResults(), newStage.getResults()))
        oldRes.replaceAllUsesWith(newRes);
      issueStage.erase();
    }

    // Plant an expect in each destination stage, and if any cross-stage
    // use of the handle-typed slot exists past destStage, forward the
    // expect's result by adding a new slot to destStage and rewriting
    // those uses to it. Group by destStage so we rebuild each destStage
    // at most once even when multiple dynamic ops converge on it.
    struct DestWrap {
      Operation *dynOp;
      unsigned issueSlot; // index into newStage's results
    };
    llvm::SmallDenseMap<LoopScheduleAtOp, SmallVector<DestWrap>, 4> byDest;
    SmallVector<LoopScheduleAtOp> destsInOrder;
    for (auto &info : infos) {
      if (info.slot == ~0u)
        continue;
      unsigned destOffset = newStage.getOffset() + info.latency;
      LoopScheduleAtOp destStage;
      for (auto candidate :
           pipeline.getStagesBlock().getOps<LoopScheduleAtOp>()) {
        if (candidate.getOffset() == destOffset) {
          destStage = candidate;
          break;
        }
      }
      if (!destStage) {
        info.dynOp->emitWarning("no destination at-stage at offset ")
            << destOffset << " for dynamic op; stall handle left dangling";
        continue;
      }
      if (!byDest.count(destStage))
        destsInOrder.push_back(destStage);
      byDest[destStage].push_back({info.dynOp, info.slot});
    }

    for (LoopScheduleAtOp destStage : destsInOrder) {
      auto &dests = byDest[destStage];

      // Plant each expect at the top of destStage's body.
      struct Planted {
        Operation *dynOp;
        unsigned issueSlot;
        LoopScheduleExpectOp expect;
      };
      SmallVector<Planted> planted;
      planted.reserve(dests.size());
      OpBuilder b(&destStage.getBodyBlock(), destStage.getBodyBlock().begin());
      for (auto &d : dests) {
        auto expect = b.create<LoopScheduleExpectOp>(
            d.dynOp->getLoc(), d.dynOp->getResultTypes(),
            newStage.getResult(d.issueSlot));
        planted.push_back({d.dynOp, d.issueSlot, expect});
      }

      // Rewrite every use of the handle-typed issue-stage slot that sits
      // inside destStage's body to consume the expect's result. Uses
      // OUTSIDE destStage's body (in sibling at-stage bodies after
      // destStage) need the value forwarded via destStage's yield — we
      // record those and fix them up by extending destStage's yield with
      // a new value-typed slot.
      struct ForwardNeed {
        unsigned issueSlot;
        LoopScheduleExpectOp expect;
      };
      SmallVector<ForwardNeed> forwardNeeds;
      forwardNeeds.reserve(planted.size());
      for (auto &p : planted) {
        Value stageRes = newStage.getResult(p.issueSlot);
        bool needForward = false;
        for (OpOperand &use : llvm::make_early_inc_range(stageRes.getUses())) {
          Operation *user = use.getOwner();
          if (user == p.expect.getOperation())
            continue;
          if (destStage->isProperAncestor(user)) {
            use.set(p.expect.getResult(0));
            continue;
          }
          needForward = true;
        }
        if (needForward)
          forwardNeeds.push_back({p.issueSlot, p.expect});
      }
      if (forwardNeeds.empty())
        continue;

      // Extend destStage's yield with the expect results, and rebuild
      // destStage with the augmented result-type list. Cross-stage users
      // of the original handle-typed slot are rewritten to the new
      // value-typed slot on the rebuilt destStage.
      auto destYield = destStage.getYieldOp();
      SmallVector<Type> newDestTypes(destStage.getResultTypes());
      SmallVector<unsigned> addedSlots;
      addedSlots.reserve(forwardNeeds.size());
      for (auto &fn : forwardNeeds) {
        addedSlots.push_back(newDestTypes.size());
        newDestTypes.push_back(fn.expect.getResult(0).getType());
        destYield->insertOperands(destYield->getNumOperands(),
                                  fn.expect.getResult(0));
      }
      OpBuilder rebuildDest(destStage);
      auto newDest = rebuildDest.create<LoopScheduleAtOp>(
          destStage.getLoc(), newDestTypes, destStage.getOffsetAttr());
      newDest.getBody().takeBody(destStage.getBody());
      for (auto [oldRes, newRes] :
           llvm::zip(destStage.getResults(), newDest.getResults()))
        oldRes.replaceAllUsesWith(newRes);
      destStage.erase();

      // Now rewrite any remaining cross-stage handle-slot uses to the
      // new forwarded-value slot on newDest.
      for (auto [fn, addedSlot] : llvm::zip(forwardNeeds, addedSlots)) {
        Value stageRes = newStage.getResult(fn.issueSlot);
        Value forwarded = newDest.getResult(addedSlot);
        for (OpOperand &use : llvm::make_early_inc_range(stageRes.getUses())) {
          Operation *user = use.getOwner();
          if (user == fn.expect.getOperation())
            continue;
          if (newDest->isProperAncestor(user))
            continue; // already rewritten to the expect
          use.set(forwarded);
        }
      }
    }
  }
}

// === Shared types/state for the merged sequential+func lowering ===
//
// `createLoopScheduleSequential` and `createFuncLoopSchedule` both lower a
// scheduled region into a chain of `loopschedule.frame` ops via the same
// algorithm: partition start-times into phases (close-after-launch-bucket),
// then for each phase build one frame containing one `loopschedule.at` per
// bucket plus wrapping ats around launch ops. The two paths differ only in
// trim around the edges (terminator type, iter-arg threading, what to skip
// when grouping, what to preserve in cleanup, how to emit the tail). We
// capture those differences in `ScheduleStrategy` and share everything else
// in a single driver `lowerSchedule(strategy, problem)`.

struct BucketContent {
  uint32_t offset;
  SmallVector<Operation *> staticOps;
  // Dynamic-latency ops: emitted inside a `loopschedule.launch` and
  // produce a handle (await'd by a downstream frame). Includes loop ops
  // and `func.call` — see `isLaunchLikeOp`.
  SmallVector<Operation *> dynamicOps;
};

struct StaticExport {
  BucketContent *bucket;
  Operation *op;
  unsigned frameFirstIdx;
};

struct IterArgExport {
  BucketContent *bucket;
  Operation *op;
  Value iterArg;
  unsigned frameIdx;
};

struct PendingLaunch {
  Operation *origOp;
  Value currentHandle;
  SmallVector<std::pair<unsigned, Value>> forwards;
  DenseSet<size_t> remainingUserPhases;
  bool terminatorPending;
};

struct PendingService {
  size_t pendingIdx;
  bool awaitHere;
  bool forwardHere;
  SmallVector<std::pair<unsigned, Value>> forwards;
  std::optional<unsigned> forwardFrameIdx;
  // Parallel to `forwards`: whether each forwarded value needs to escape this
  // frame as a frame-result (because it's used by a later phase or by the
  // schedule-region terminator). Adopted from the sequential path.
  SmallVector<bool> forwardExportAsFrameResult;
  SmallVector<std::optional<unsigned>> forwardExportFrameIdx;
};

struct ServicedForwardInfo {
  size_t serviceIdx;
  unsigned firstBodyArgIdx;
};

struct IterArgSupport {
  // The original loop's after-region block args (one per iter-arg).
  // Used to detect whether an iter-arg has later uses and to seed valueMap.
  SmallVector<Value> afterArgs;
  // The new sequential's schedule-block args (one per iter-arg). Indices line
  // up with `afterArgs`.
  SmallVector<BlockArgument> scheduleBlockArgs;
  // Whether the induction-variable iter-arg (index 0) has any users, used for
  // the "trailing empty bucket if last bucket contains a launch" heuristic.
  bool inductionVarHasUsers = false;
};

struct ScheduleStrategy {
  // What we're lowering.
  Operation *root;            // scf::WhileOp or func::FuncOp
  Block *scheduleBlock;       // where top-level frames live
  Region *containingRegion;   // for findAncestorOpInRegion
  Operation *anchor;          // schedule-block terminator (scf.yield/func.return)

  // Variation hooks.
  std::function<bool(Operation *)> shouldSkipForGrouping;
  std::function<bool(Operation *)> isTerminator;
  std::function<void(OpBuilder &)> setFrameInsertion;
  std::function<void(IRMapping &)> seedValueMap;

  // Cond support — null Value if absent.
  Value condValue;

  // FUNCTION-level strategy only: wrap dynamic-latency memory accesses in
  // `loopschedule.launch` ops so the function FSM can give each one the
  // issue/completion handshake. Sequential-loop bodies keep them inline —
  // the loop FSM lowers them through the SeqDynCtx handshake path.
  bool dynAccessesAsLaunches = false;

  // Iter-arg support — std::nullopt for func.
  std::optional<IterArgSupport> iterArgs;

  // Tail emission. Called once after all phases are built.
  std::function<void(ImplicitLocOpBuilder &builder,
                     ArrayRef<PendingLaunch> pendingLaunches,
                     Value condResult,
                     ArrayRef<Value> termIterArgs,
                     ArrayRef<bool> iterArgUpdated,
                     IRMapping &valueMap)>
      finalize;
};

/// An op whose scheduler lowering emits a `loopschedule.launch` and produces
/// a handle (await'd by downstream frames). Today: loop ops plus `func.call`
/// (dynamic-latency — at least one cycle between start and done; the FSM
/// lowering stalls on the callee's done signal), plus barrier stores (e.g.
/// amc.control): data-less waits on a memory completion level, sampled by
/// the FSM through the same frame WAIT machinery.
static bool isLaunchLikeOp(Operation *op) {
  if (isa<LoopInterface, func::CallOp>(op))
    return true;
  if (auto store = dyn_cast<StoreInterface>(op))
    return store.isBarrier();
  return false;
}

/// The solver's SECONDARY objectives: among anchor-optimal schedules, pick
/// the one that is cheapest in the FRAME LOWERING's actual cost model —
/// lexicographically, FSM states first, capture registers second,
/// determinism last.
///
/// Phase 1 (the simplex) minimizes the ANCHOR (the terminator) alone. Any
/// op with slack against the anchor's binding constraints is a degenerate
/// variable: the LP has a family of equally-optimal solutions, and the
/// simplex returns whichever vertex its pivot order lands on. That is not
/// a choice, it is the absence of one, and it is not stable across
/// solver-internal changes. The phases below supply the missing criteria,
/// each spending only the currency it actually prices:
///
///  A. STATES — shrink the last occupied bucket. The FSM builds a state
///     for every cycle a real op occupies; the anchor itself is virtual
///     (a latency-1 tail store's commit edge pushes the anchor one slot
///     past the store, but the commit costs no state — spmv's fused tail
///     is the existence proof). An op drifting into that virtual slot
///     pins it down as a dead state. Moves are ALL-OR-NOTHING per bucket:
///     if every occupant of the max bucket can drop to its dependence
///     bound, the state disappears; if any occupant is pinned, nothing
///     moves — a partial move would leave the state standing AND stretch
///     the moved values' lifetimes toward it.
///  B. REGISTERS — pull remaining slack ops TOWARD their nearest real
///     consumer (never past any dependence, never past the makespan). A
///     value consumed more than one cycle after production mints a
///     `_latched` capture register downstream, so accidental-early
///     placement is a real area cost. Terminator edges are EXCLUDED from
///     the pull: the iter-arg registers' D inputs are combinational from
///     their producers in whatever cycle the advance fires, so a
///     "lifetime" against the anchor has no register semantics — pulling
///     toward it would re-mint the phase-A dead state to save a register
///     that does not exist.
///  C. DETERMINISM — ops with no real consumers at all (anchor-only:
///     zero-latency, memory-effect-free, every user the anchor or another
///     such op) sink to ASAP. No capture can exist for them and phase A
///     already took any state they pinned; ASAP just makes the placement
///     a stated rule instead of a pivot-order accident.
///
/// Every move is bounded by the op's own dependences on both sides, and
/// `verify()` is the backstop: on any violation the saved solution is
/// restored verbatim and the schedule is exactly what phase 1 produced.
static void applySecondaryScheduleObjectives(
    ChainingSharedOperatorsProblem &problem, Operation *anchor) {
  // Save the solution for the revert path.
  DenseMap<Operation *, unsigned> savedStart;
  DenseMap<Operation *, float> savedInCycle;
  for (auto *op : problem.getOperations()) {
    if (auto st = problem.getStartTime(op))
      savedStart[op] = *st;
    if (auto stc = problem.getStartTimeInCycle(op))
      savedInCycle[op] = *stc;
  }

  // A movable op: zero-latency pure comb, never a launch, with a solved
  // start. Everything else keeps its phase-1 placement.
  auto movable = [&](Operation *op) {
    if (op == anchor || isLaunchLikeOp(op) || !isMemoryEffectFree(op))
      return false;
    auto oprOpt = problem.getLinkedOperatorType(op);
    return problem.getStartTime(op).has_value() && oprOpt.has_value() &&
           problem.getLatency(*oprOpt).value_or(1) == 0;
  };

  // ASAP over every incoming dependence (SSA and auxiliary alike), or
  // nullopt when a source lacks scheduling info.
  auto asapOf = [&](Operation *op) -> std::optional<unsigned> {
    unsigned asap = 0;
    for (auto dep : problem.getDependences(op)) {
      Operation *src = dep.getSource();
      auto srcStart = problem.getStartTime(src);
      auto srcOpr = problem.getLinkedOperatorType(src);
      if (!srcStart || !srcOpr)
        return std::nullopt;
      asap = std::max(asap, *srcStart + problem.getLatency(*srcOpr).value_or(0));
    }
    return asap;
  };

  // In-cycle start at `cycle`: the latest arrival among sources whose
  // result lands exactly there. Combinational sources chain from their own
  // in-cycle end; latching sources deliver at their outgoing delay past the
  // cycle edge; earlier-finishing sources impose no in-cycle bound.
  auto inCycleAt = [&](Operation *op, unsigned cycle) {
    float inCycle = 0.0f;
    for (auto dep : problem.getDependences(op)) {
      Operation *src = dep.getSource();
      auto srcOpr = problem.getLinkedOperatorType(src);
      auto srcStart = problem.getStartTime(src);
      if (!srcOpr || !srcStart)
        continue;
      unsigned srcLat = problem.getLatency(*srcOpr).value_or(0);
      if (*srcStart + srcLat != cycle)
        continue;
      float srcOut = problem.getOutgoingDelay(*srcOpr).value_or(0.0f);
      float arrival =
          srcLat == 0
              ? problem.getStartTimeInCycle(src).value_or(0.0f) + srcOut
              : srcOut;
      inCycle = std::max(inCycle, arrival);
    }
    return inCycle;
  };

  auto moveTo = [&](Operation *op, unsigned cycle) {
    problem.setStartTime(op, cycle);
    problem.setStartTimeInCycle(op, inCycleAt(op, cycle));
  };

  // Auxiliary OUTGOING dependences (op as source) are not enumerable from
  // the op itself; build the reverse index once.
  DenseMap<Operation *, SmallVector<Operation *>> auxOut;
  for (auto *op : problem.getOperations())
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        auxOut[dep.getSource()].push_back(op);

  // The anchor-only class (phase C's domain, and phase B's exclusion): ops
  // whose every user is the anchor or another member. Reverse block order
  // resolves chains (addi -> cmpi -> terminator) in one pass.
  DenseSet<Operation *> anchorOnly;
  Block *block = anchor->getBlock();
  for (Operation &opRef : llvm::reverse(*block)) {
    Operation *op = &opRef;
    if (!movable(op))
      continue;
    bool all = true;
    for (Operation *user : op->getUsers())
      if (user != anchor && !anchorOnly.contains(user)) {
        all = false;
        break;
      }
    if (all)
      anchorOnly.insert(op);
  }

  bool changed = false;

  // --- Phase A: states. Shrink the last occupied bucket, all-or-nothing
  // per bucket, until an occupant is pinned.
  for (;;) {
    unsigned maxBucket = 0;
    SmallVector<Operation *> occupants;
    for (auto *op : problem.getOperations()) {
      if (op == anchor)
        continue;
      auto st = problem.getStartTime(op);
      if (!st)
        continue;
      if (*st > maxBucket) {
        maxBucket = *st;
        occupants.clear();
      }
      if (*st == maxBucket)
        occupants.push_back(op);
    }
    if (maxBucket == 0 || occupants.empty())
      break;
    SmallVector<std::pair<Operation *, unsigned>> moves;
    bool allMove = true;
    for (Operation *op : occupants) {
      auto asap = movable(op) ? asapOf(op) : std::nullopt;
      if (!asap || *asap >= maxBucket) {
        allMove = false;
        break;
      }
      moves.push_back({op, *asap});
    }
    if (!allMove)
      break;
    for (auto &[op, asap] : moves)
      moveTo(op, asap);
    changed = true;
  }

  // Recompute the (possibly shrunk) makespan as phase B's ceiling.
  unsigned makespan = 0;
  for (auto *op : problem.getOperations())
    if (op != anchor)
      if (auto st = problem.getStartTime(op))
        makespan = std::max(makespan, *st);

  // --- Phase B: registers. Pull ops with REAL consumers toward the
  // nearest one, so a value's lifetime stays within the one-cycle window
  // that needs no capture register. Bounded by every outgoing dependence
  // and by the makespan (never re-grow a bucket phase A vacated).
  for (Operation &opRef : *block) {
    Operation *op = &opRef;
    if (!movable(op) || anchorOnly.contains(op))
      continue;
    auto curOpt = problem.getStartTime(op);
    if (!curOpt)
      continue;
    unsigned upper = makespan;
    unsigned nearestReal = UINT_MAX;
    bool boundsOk = true;
    for (Operation *user : op->getUsers()) {
      auto us = problem.getStartTime(user);
      if (!us) {
        boundsOk = false;
        break;
      }
      upper = std::min(upper, *us); // zero-latency: may share the cycle
      if (user != anchor && !anchorOnly.contains(user))
        nearestReal = std::min(nearestReal, *us);
    }
    if (!boundsOk || nearestReal == UINT_MAX)
      continue;
    if (auto it = auxOut.find(op); it != auxOut.end())
      for (Operation *dst : it->second) {
        auto ds = problem.getStartTime(dst);
        if (!ds) {
          boundsOk = false;
          break;
        }
        upper = std::min(upper, *ds);
      }
    if (!boundsOk)
      continue;
    // Ideal: one cycle before the nearest real consumer (its data arrives
    // combinationally valid in the consumer's read window without a
    // capture); clamp to the hard upper bound.
    unsigned target = std::min(upper, nearestReal > 0 ? nearestReal - 1 : 0u);
    if (target > *curOpt) {
      moveTo(op, target);
      changed = true;
    }
  }

  // --- Phase C: determinism. Anchor-only ops sink to ASAP: no real
  // consumer means no capture register can exist, phase A already took any
  // state they pinned, and a stated rule beats a pivot-order accident.
  for (Operation &opRef : llvm::reverse(*block)) {
    Operation *op = &opRef;
    if (!anchorOnly.contains(op))
      continue;
    auto cur = problem.getStartTime(op);
    auto asap = asapOf(op);
    if (!cur || !asap || *asap >= *cur)
      continue;
    moveTo(op, *asap);
    changed = true;
  }

  if (!changed)
    return;
  if (failed(problem.verify())) {
    // Restore the solver's solution verbatim; the tie-break is a pure
    // optimization and must never turn a valid schedule invalid.
    for (auto &[op, st] : savedStart)
      problem.setStartTime(op, st);
    for (auto &[op, stc] : savedInCycle)
      problem.setStartTimeInCycle(op, stc);
  }
}

/// Partition start-times into phases by the close-after-launch-bucket rule:
/// a bucket containing a launch-like op closes the current phase. This
/// guarantees that sibling launches at different start times land in separate
/// frames so the second frame can await the first's handle (the only correct
/// ordering for two launches with a memref dependency but no SSA dep).
static SmallVector<SmallVector<unsigned>>
partitionPhasesByLaunch(ArrayRef<unsigned> startTimes,
                        const DenseMap<unsigned, SmallVector<Operation *>>
                            &startGroups,
                        llvm::function_ref<bool(Operation *)> isLaunchLike) {
  SmallVector<SmallVector<unsigned>> phases;
  SmallVector<unsigned> currentPhase;
  for (auto t : startTimes) {
    currentPhase.push_back(t);
    bool hasLaunch = false;
    auto it = startGroups.find(t);
    if (it != startGroups.end())
      for (auto *op : it->second)
        if (isLaunchLike(op)) {
          hasLaunch = true;
          break;
        }
    if (hasLaunch) {
      phases.push_back(currentPhase);
      currentPhase.clear();
    }
  }
  if (!currentPhase.empty())
    phases.push_back(currentPhase);
  return phases;
}

/// Clone the `before` body's non-terminator ops into the start of the `after`
/// body so they participate in the scheduling problem. The `before` body
/// computes the loop condition from the iter_args; by inlining these ops into
/// the `after` body, the scheduler handles them as normal operations instead of
/// special-casing them. Returns the cloned condition value in the `after` body.
static Value inlineBeforeBodyOps(scf::WhileOp loop) {
  auto scfCond = cast<scf::ConditionOp>(loop.getBeforeBody()->getTerminator());
  Block *afterBody = loop.getAfterBody();
  IRMapping mapping;
  // before block args and after block args both represent the same iter_args.
  for (auto [beforeArg, afterArg] :
       llvm::zip(loop.getBeforeArguments(), loop.getAfterArguments()))
    mapping.map(beforeArg, afterArg);

  OpBuilder builder(loop.getContext());
  builder.setInsertionPointToStart(afterBody);
  for (auto &op : loop.getBeforeBody()->getOperations()) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    builder.clone(op, mapping);
  }
  return mapping.lookup(scfCond.getCondition());
}

/// Check that a sequential loop's continuation is something the loop FSM can
/// actually evaluate when it has to.
///
/// `loopschedule.sequential` decides whether to run another iteration in a
/// COND state that is entered BEFORE the first one, and the hardware reads
/// that decision off the loop's carried registers (or, with the COND bypass,
/// off their D wires). So the condition must be a function of the iteration
/// arguments and of values that dominate the loop. A condition that READS
/// MEMORY inside the body has no value at all on loop entry — the FSM would
/// sample whatever the memory's output register happened to hold.
///
/// This is exactly the shape a source `while (A[i] != 0)` has, and the shape
/// `--promote-scf-scalars` removes for a search state kept in a scratch
/// memref by making it a carried value. Diagnose what it could not reach,
/// with the source loop's location, instead of lowering it into an FSM that
/// would be quietly wrong.
static LogicalResult checkSequentialCondition(scf::WhileOp loop, Value cond) {
  SmallVector<Value> stack{cond};
  DenseSet<Value> visited;
  while (!stack.empty()) {
    Value v = stack.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    // A block argument is either an iteration argument (a register the FSM
    // reads directly) or belongs to an enclosing region — both fine.
    if (isa<BlockArgument>(v))
      continue;
    Operation *def = v.getDefiningOp();
    // Defined outside the loop body: computed once, before the loop runs.
    if (!def || !loop.getAfter().isAncestor(def->getParentRegion()))
      continue;
    if (!isMemoryEffectFree(def) || def->getNumRegions() != 0) {
      InFlightDiagnostic diag = loop.emitOpError(
          "`while` condition is computed from something the loop FSM cannot "
          "read before an iteration runs");
      diag.attachNote(def->getLoc())
          << "'" << def->getName()
          << "' contributes to the condition but executes inside the loop "
             "body";
      diag.attachNote()
          << "a sequential loop evaluates its continuation from the values it "
             "CARRIES, once before every iteration including the first; make "
             "the searched-for state a loop-carried value rather than reading "
             "it out of memory inside the loop";
      return failure();
    }
    for (Value operand : def->getOperands())
      stack.push_back(operand);
  }
  return success();
}

} // namespace

void SCFToLoopSchedulePass::runOnOperation() {
  // Now a ModuleOp pass: walk each top-level func.func and lower it. The
  // pass replaces each func.func with a `loopschedule.func_sequential`, so
  // it cannot run as `Pass<func::FuncOp>` (the framework requires the
  // root op to remain valid). Collect funcs first to avoid mutating the
  // module while iterating.
  ModuleOp moduleOp = getOperation();
  SmallVector<FuncOp> funcs;
  moduleOp.walk([&](FuncOp f) { funcs.push_back(f); });
  for (auto funcOp : funcs)
    if (failed(runOnFunc(funcOp))) {
      signalPassFailure();
      return;
    }

  // After all funcs have been wrapped into loopschedule.func_sequential /
  // loopschedule.func_pipeline, rewrite any surviving func.call whose callee
  // resolves to a schedule-level container. Done in a second pass so the
  // callee is guaranteed to be converted regardless of module iteration order.
  SymbolTable symbolTable(moduleOp);
  SmallVector<func::CallOp> calls;
  moduleOp.walk([&](func::CallOp call) { calls.push_back(call); });
  for (auto call : calls) {
    Operation *callee = symbolTable.lookup(call.getCallee());
    if (!callee ||
        !isa<LoopScheduleFuncSequentialOp, LoopScheduleFuncPipelineOp>(callee))
      continue;
    OpBuilder builder(call);
    auto newCall = builder.create<LoopScheduleCallOp>(
        call.getLoc(), call.getCalleeAttr(), call.getResultTypes(),
        call.getOperands());
    call.replaceAllUsesWith(newCall.getResults());
    call.erase();
  }
}

LogicalResult SCFToLoopSchedulePass::runOnFunc(FuncOp funcOp) {
  float cycleTime = cycleTimeNs;

  // Reset per-func state so we don't leak entries from a prior func.
  predicateMap.clear();
  predicateUse.clear();
  dependenceAnalysis.reset();
  operatorLibraryAnalysis.reset();

  // Collect loops to pipeline and work on them.
  SmallVector<scf::WhileOp> loops;

  auto hasPipelinedParent = [](Operation *op) {
    Operation *currentOp = op;

    while (!isa<ModuleOp>(currentOp->getParentOp())) {
      if (currentOp->getParentOp()->hasAttr("hls.pipeline"))
        return true;
      currentOp = currentOp->getParentOp();
    }

    return false;
  };

  auto res = funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<scf::WhileOp>(op) || !op->hasAttr("hls.pipeline"))
      return WalkResult::advance();

    if (hasPipelinedParent(op))
      return WalkResult::interrupt();

    loops.push_back(cast<scf::WhileOp>(op));
    return WalkResult::advance();
  });

  if (res.wasInterrupted())
    return funcOp.emitOpError(
        "Loops marked for pipelining cannot contain other loops");

  // func.call is a dynamic-latency op; it cannot appear inside a
  // pipelined context (modulo-problem scheduling assumes bounded per-op
  // latency). Diagnose early.
  for (auto loop : loops) {
    func::CallOp badCall;
    loop.getAfter().walk([&](func::CallOp call) {
      badCall = call;
      return WalkResult::interrupt();
    });
    if (badCall)
      return badCall.emitOpError(
          "func.call is not allowed inside a pipelined loop "
          "(dynamic-latency op)");
  }

  // A barrier store (e.g. amc.control) is a whole-memory completion wait;
  // inside a pipelined loop it would have to drain per iteration, which
  // defeats pipelining and is unsupported. Diagnose early.
  for (auto loop : loops) {
    Operation *badBarrier = nullptr;
    loop.getAfter().walk([&](Operation *op) {
      if (auto store = dyn_cast<StoreInterface>(op))
        if (store.isBarrier()) {
          badBarrier = op;
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    });
    if (badBarrier)
      return badBarrier->emitOpError(
          "memory control waits are not allowed inside a pipelined loop; "
          "place the wait between loops");
  }

  // Per-function analyses obtained via the analysis manager scoped to this
  // child func.
  dependenceAnalysis = getChildAnalysis<LoopScheduleDependenceAnalysis>(funcOp);
  operatorLibraryAnalysis = getChildAnalysis<OperatorLibraryAnalysis>(funcOp);

  // Map from while op to the cloned condition value in the after body.
  DenseMap<Operation *, Value> loopCondValues;

  // Schedule all pipelined loops first
  for (auto loop : llvm::make_early_inc_range(loops)) {
    // Inline the before-body condition ops into the after body so they
    // participate in the scheduling problem like any other operation.
    loopCondValues[loop] = inlineBeforeBodyOps(loop);

    ResourceMap resourceMap;
    ResourceLimits resourceLimits;
    if (failed(recordMemoryResources(loop.getOperation(), loop.getAfter(),
                                     resourceMap, resourceLimits)))
      return failure();

    if (failed(
            ifOpConversion(loop.getOperation(), loop.getAfter(), predicateMap)))
      return failure();

    // Populate the target operator types.
    ChainingModuloProblem moduloProblem =
        getChainingModuloProblem(loop, *dependenceAnalysis);

    if (failed(populateOperatorTypes(loop.getOperation(), loop.getAfter(),
                                     moduloProblem,
                                     /*sequentialRegion=*/false)))
      return failure();

    if (failed(addMemoryResources(loop.getOperation(), loop.getAfter(),
                                  moduloProblem, resourceMap, resourceLimits)))
      return failure();

    addPredicateDependencies(loop.getOperation(), loop.getAfter(),
                             moduloProblem, predicateMap, predicateUse);

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveChainingModuloProblem(loop, moduloProblem, cycleTime))) {
      llvm::errs() << "Failed to solve ChainingModuloProblem\n";
      return failure();
    }

    // Convert the IR.
    if (failed(createLoopSchedulePipeline(loop, moduloProblem,
                                          loopCondValues[loop])))
      return failure();
  }

  // Schedule all remaining loops
  SmallVector<scf::WhileOp> seqLoops;

  funcOp.walk([&](scf::WhileOp loop) {
    seqLoops.push_back(loop);
    return WalkResult::advance();
  });

  // Schedule loops
  for (auto loop : seqLoops) {
    // Inline the before-body condition ops into the after body so they
    // participate in the scheduling problem like any other operation.
    loopCondValues[loop] = inlineBeforeBodyOps(loop);

    // The continuation has to be readable before an iteration runs. Checked
    // here, on the still-recognisable loop, so an unsupported `while` gets a
    // diagnostic pointing at the offending op instead of a schedule the FSM
    // lowering cannot honour.
    if (failed(checkSequentialCondition(loop, loopCondValues[loop])))
      return failure();

    ResourceMap resourceMap;
    ResourceLimits resourceLimits;
    if (failed(recordMemoryResources(loop.getOperation(), loop.getAfter(),
                                     resourceMap, resourceLimits)))
      return failure();

    if (failed(
            ifOpConversion(loop.getOperation(), loop.getAfter(), predicateMap)))
      return failure();

    // Un-CSE yield operands that also feed same-iteration consumers. The
    // shared-operators problem places a WAR dependence from every reader of
    // an iter arg onto the producer of its NEXT value (the iter-arg register
    // may update mid-frame), so a producer that doubles as, say, an address
    // for a load whose result reaches a store closes a positive-latency
    // cycle and the schedule is declared infeasible — even though the
    // hardware is fine, since the consumer reads the comb value while only
    // the register update carries the WAR. Giving the yield a private clone
    // of any pure producer with other in-loop users separates the two roles
    // (this is the shape the IR had before upstream CSE learned to merge
    // them).
    {
      Operation *terminator = loop.getAfterBody()->getTerminator();
      OpBuilder cloneBuilder(terminator);
      for (OpOperand &yielded : terminator->getOpOperands()) {
        Operation *def = yielded.get().getDefiningOp();
        if (!def || def->getBlock() != terminator->getBlock() ||
            !isMemoryEffectFree(def) || def->getNumResults() != 1)
          continue;
        bool hasOtherUsers = llvm::any_of(
            def->getResult(0).getUsers(),
            [&](Operation *user) { return user != terminator; });
        if (!hasOtherUsers)
          continue;
        Operation *clone = cloneBuilder.clone(*def);
        yielded.set(clone->getResult(0));
      }
    }

    auto problem = getChainingSharedOperatorsProblem(loop, *dependenceAnalysis);

    // Populate the target operator types.
    if (failed(populateOperatorTypes(loop.getOperation(), loop.getAfter(),
                                     problem, /*sequentialRegion=*/true)))
      return failure();

    if (failed(addMemoryResources(loop.getOperation(), loop.getAfter(), problem,
                                  resourceMap, resourceLimits)))
      return failure();

    addPredicateDependencies(loop.getOperation(), loop.getAfter(), problem,
                             predicateMap, predicateUse);

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveChainingSharedOperatorsProblem(loop.getAfter(), problem,
                                                   cycleTime)))
      return failure();

    // Convert the IR.
    if (failed(createLoopScheduleSequential(loop, problem,
                                            loopCondValues[loop])))
      return failure();
  }

  // === Schedule whole function ===
  // Branch on `hls.pipeline` attr on the func itself: pipelined functions
  // get a `loopschedule.func_pipeline`, others get `loopschedule.func_sequential`.
  bool funcIsPipelined = funcOp->hasAttr("hls.pipeline");

  ResourceMap resourceMap;
  ResourceLimits resourceLimits;
  if (failed(recordMemoryResources(funcOp.getOperation(), funcOp.getRegion(),
                                   resourceMap, resourceLimits)))
    return failure();

  if (failed(ifOpConversion(funcOp.getOperation(), funcOp.getRegion(),
                            predicateMap)))
    return failure();

  // Pipelined functions cannot contain nested loops on day one (mirrors the
  // `LoopSchedulePipelineOp`-on-while restriction). Surfacing this here
  // gives a clean diagnostic before scheduling burns cycles on an
  // unsolvable shape.
  if (funcIsPipelined) {
    bool hasNestedLoop = false;
    funcOp.walk([&](LoopInterface) { hasNestedLoop = true; });
    if (hasNestedLoop)
      return funcOp.emitOpError(
          "func-level pipelining does not yet support nested loops");

    // Pipelined functions overlap in-flight transactions, but a memory
    // argument carries no per-transaction identity: nothing at the
    // interface says which transaction a read or write belongs to, so
    // each overlapped transaction would need its own image of the
    // memory maintained by the environment with cycle-exact timing.
    // Until that contract is architectural (e.g. ping-pong buffers or
    // per-transaction base addresses), reject memory arguments up front.
    for (BlockArgument arg : funcOp.getArguments())
      if (!arg.getType().isIntOrIndexOrFloat())
        return funcOp.emitOpError(
            "func-level pipelining does not support memory arguments");

    func::CallOp badCall;
    funcOp.walk([&](func::CallOp call) {
      badCall = call;
      return WalkResult::interrupt();
    });
    if (badCall)
      return badCall.emitOpError(
          "func.call is not allowed inside a pipelined func "
          "(dynamic-latency op)");
  }

  if (funcIsPipelined) {
    auto problem = getChainingModuloProblem(funcOp, *dependenceAnalysis);
    if (failed(populateOperatorTypes(funcOp.getOperation(), funcOp.getBody(),
                                     problem, /*sequentialRegion=*/false)))
      return failure();
    if (failed(addMemoryResources(funcOp.getOperation(), funcOp.getRegion(),
                                  problem, resourceMap, resourceLimits)))
      return failure();
    addPredicateDependencies(funcOp.getOperation(), funcOp.getRegion(),
                             problem, predicateMap, predicateUse);
    // Solve the modulo problem. The anchor is the func.return op (already
    // inserted into the problem by the modulo-problem builder).
    std::optional<int32_t> requestedII;
    if (auto iiAttr = funcOp->getAttrOfType<IntegerAttr>("hls.pipeline"))
      requestedII = iiAttr.getInt();
    auto *anchor = funcOp.getBody().back().getTerminator();
    if (failed(problem.check()))
      return failure();
    if (failed(scheduling::scheduleSimplex(problem, anchor, cycleTime,
                                           requestedII)))
      return funcOp.emitOpError(
          "failed to solve ChainingModuloProblem for pipelined func");
    if (failed(problem.verify()))
      return failure();
    if (requestedII.has_value() &&
        problem.getInitiationInterval() != requestedII) {
      return funcOp.emitError(
          "Failed to schedule for desired II of " +
          std::to_string(*requestedII) + ", minimum II is " +
          std::to_string(problem.getInitiationInterval().value()));
    }

    // Pre-cleanup (same as sequential path) before we replace funcOp.
    if (funcOp->hasAttrOfType<SymbolRefAttr>("loopschedule.dependencies")) {
      auto depSymbol =
          funcOp->getAttrOfType<SymbolRefAttr>("loopschedule.dependencies");
      auto *depOp = SymbolTable::lookupNearestSymbolFrom(funcOp, depSymbol);
      depOp->walk([](LoopScheduleAccessOp op) { op.erase(); });
      depOp->walk([](LoopScheduleDependsOnOp op) { op.erase(); });
      depOp->erase();
      funcOp->removeAttr("loopschedule.dependencies");
    }
    funcOp->walk([](Operation *op) {
      if (op->hasAttrOfType<StringAttr>(NameAnalysis::getAttributeName()))
        op->removeAttr(NameAnalysis::getAttributeName());
    });

    if (failed(createFuncLoopSchedulePipeline(funcOp, problem)))
      return failure();
    return success();
  }

  auto problem = getChainingSharedOperatorsProblem(funcOp, *dependenceAnalysis);

  // Populate the target operator types.
  if (failed(populateOperatorTypes(funcOp.getOperation(), funcOp.getBody(),
                                   problem, /*sequentialRegion=*/true)))
    return failure();

  if (failed(addMemoryResources(funcOp.getOperation(), funcOp.getRegion(),
                                problem, resourceMap, resourceLimits)))
    return failure();

  addPredicateDependencies(funcOp.getOperation(), funcOp.getRegion(), problem,
                           predicateMap, predicateUse);

  // Solve the scheduling problem computed by the analysis.
  if (failed(solveChainingSharedOperatorsProblem(funcOp.getBody(), problem,
                                                 cycleTime)))
    return failure();

  // Pre-cleanup that needs the live `func.func`. createFuncLoopSchedule
  // wraps the body in a loopschedule.func_sequential and erases the
  // original funcOp, so anything that touches funcOp must run first.
  if (funcOp->hasAttrOfType<SymbolRefAttr>("loopschedule.dependencies")) {
    auto depSymbol =
        funcOp->getAttrOfType<SymbolRefAttr>("loopschedule.dependencies");
    auto *depOp = SymbolTable::lookupNearestSymbolFrom(funcOp, depSymbol);
    depOp->walk([](LoopScheduleAccessOp op) { op.erase(); });
    depOp->walk([](LoopScheduleDependsOnOp op) { op.erase(); });
    depOp->erase();
    funcOp->removeAttr("loopschedule.dependencies");
  }

  // Strip access-name attrs while the body is still under the func.
  funcOp->walk([](Operation *op) {
    if (op->hasAttrOfType<StringAttr>(NameAnalysis::getAttributeName())) {
      op->removeAttr(NameAnalysis::getAttributeName());
    }
  });

  // Convert the IR (this erases funcOp and replaces it with a
  // loopschedule.func_sequential at the same module-level location).
  if (failed(createFuncLoopSchedule(funcOp, problem)))
    return failure();

  return success();
}

/// True iff `v` reaches, through pure ops in the SAME region level, an operand
/// of a memory access — an op whose address or write data the FSM drives
/// combinationally in the very cycle it is scheduled in.
///
/// That is the one consumer kind for which a sequential frame's completion-
/// clocked `_cap` register is a cycle too late (see the caller). Users inside a
/// nested `loopschedule.sequential` / `loopschedule.pipeline` are deliberately
/// NOT followed: those reach the value through a launch, whose body runs the
/// cycle after the start pulse, by which time `_cap` holds it.
static bool feedsSameCycleAccess(Value v) {
  SmallVector<Value, 4> work{v};
  llvm::SmallDenseSet<Operation *, 8> seen;
  while (!work.empty()) {
    for (Operation *user : work.pop_back_val().getUsers()) {
      if (user->getParentOfType<LoopScheduleSequentialOp>() ||
          user->getParentOfType<LoopSchedulePipelineOp>())
        continue; // reached through a launch, a cycle later
      if (isa<LoadInterface, StoreInterface, AffineLoadOp, AffineStoreOp,
              memref::LoadOp, memref::StoreOp>(user))
        return true;
      // Only pure single-result forwarding is followed; anything else is not
      // an address computation.
      if (user->getNumResults() != 1 || !isMemoryEffectFree(user))
        continue;
      if (seen.insert(user).second)
        work.push_back(user->getResult(0));
    }
  }
  return false;
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult SCFToLoopSchedulePass::populateOperatorTypes(
    Operation *op, Region &loopBody, ChainingSharedOperatorsProblem &problem,
    bool sequentialRegion) {
  // Scheduling analyis only considers the innermost loop nest for now.

  // Set an operator type's delays, clamped to the cycle budget. The
  // chaining analysis hard-fails when a single operator's delay exceeds
  // the cycle time; the physically honest fallback is to give such an
  // operator a full cycle to itself, so clamp and warn instead.
  float cycleTime = cycleTimeNs;
  auto setDelays = [&](Problem::OperatorType opr, float inc, float out,
                       Operation *locOp) {
    if (inc > cycleTime || out > cycleTime)
      locOp->emitWarning("operator '")
          << opr.getAttr().getValue() << "' delays (incoming " << inc
          << ", outgoing " << out << " ns) exceed the cycle time of "
          << cycleTime << " ns; clamping to the full cycle";
    problem.setIncomingDelay(opr, std::min(inc, cycleTime));
    problem.setOutgoingDelay(opr, std::min(out, cycleTime));
  };

  // Resolve a memory access's scheduling timing from the per-memory
  // `oplib.operator` stamped by OperatorAllocation, when present. Returns
  // {operator name, latency, incoming delay, outgoing delay}; the delay
  // defaults match the historical hardcoded memory constants.
  struct MemTiming {
    StringRef name;
    unsigned latency;
    float incomingDelay;
    float outgoingDelay;
  };
  auto memAccessTiming = [&](Operation *memOp) -> std::optional<MemTiming> {
    auto sym = memOp->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    if (!sym)
      return std::nullopt;
    StringRef name = sym.getLeafReference();
    if (!operatorLibraryAnalysis->hasOperator(name))
      return std::nullopt;
    return MemTiming{
        name, operatorLibraryAnalysis->getOperatorLatency(name),
        operatorLibraryAnalysis->getOperatorIncomingDelay(name).value_or(0.5f),
        operatorLibraryAnalysis->getOperatorOutgoingDelay(name).value_or(
            0.5f)};
  };

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType freeOpr = problem.getOrInsertOperatorType("free");
  problem.setLatency(freeOpr, 0);
  problem.setIncomingDelay(freeOpr, 0);
  problem.setOutgoingDelay(freeOpr, 0);
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  problem.setIncomingDelay(combOpr, 0.2);
  problem.setOutgoingDelay(combOpr, 0.2);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  problem.setIncomingDelay(seqOpr, 0.5);
  problem.setOutgoingDelay(seqOpr, 0.5);
  Problem::OperatorType loopOpr = problem.getOrInsertOperatorType("loop");
  problem.setLatency(loopOpr, 1);
  problem.setIncomingDelay(loopOpr, 0.0);
  problem.setOutgoingDelay(loopOpr, 0.0);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 4);
  problem.setIncomingDelay(mcOpr, 0.5);
  problem.setOutgoingDelay(mcOpr, 0.5);
  Problem::OperatorType divOpr = problem.getOrInsertOperatorType("divider");
  problem.setLatency(divOpr, 36);
  problem.setIncomingDelay(divOpr, 0.5);
  problem.setOutgoingDelay(divOpr, 0.5);

  Operation *unsupported;
  WalkResult result = loopBody.walk([&](Operation *op) {
    if (op->getParentOfType<LoopScheduleSequentialOp>() != nullptr ||
        op->getParentOfType<LoopSchedulePipelineOp>() != nullptr) {
      return WalkResult::advance();
    }

    auto potentialOperators =
        operatorLibraryAnalysis->getPotentialOperators(op);
    if (!potentialOperators.empty()) {
      // Just pick the first potential operator for now
      // TODO: Perform better allocation
      auto selectedOperator = potentialOperators.front();
      Problem::OperatorType libOpr =
          problem.getOrInsertOperatorType(selectedOperator);
      problem.setLatency(libOpr, operatorLibraryAnalysis->getOperatorLatency(
                                     selectedOperator));
      setDelays(
          libOpr,
          operatorLibraryAnalysis->getOperatorIncomingDelay(selectedOperator)
              .value_or(0.0),
          operatorLibraryAnalysis->getOperatorOutgoingDelay(selectedOperator)
              .value_or(0.0),
          op);
      problem.setLinkedOperatorType(op, libOpr);
      op->setAttr("loopschedule.operator",
                  SymbolRefAttr::get(libOpr.getAttr()));
      return WalkResult::advance();
    }

    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::TruncIOp, IndexCastOp, memref::AllocaOp, memref::AllocOp,
              loopschedule::AllocInterface, YieldOp, func::ReturnOp,
              comb::ExtractOp>([&](Operation *freeOp) {
          // Some known free ops.
          problem.setLinkedOperatorType(freeOp, freeOpr);
          return WalkResult::advance();
        })
        .Case<func::CallOp>([&](Operation *callOp) {
          // Dynamic-latency op. Scheduling sees it as at least 1 cycle
          // between start and done — the FSM lowering stalls on the real
          // done signal, so the actual latency can be longer. Each callee
          // gets its own resource type so two call sites to the same
          // callee serialize by default (sharing the single hardware
          // instance); duplicating the instance is a future extension.
          auto call = cast<func::CallOp>(callOp);
          std::string id = ("call_" + call.getCallee()).str();
          Problem::OperatorType opr = problem.getOrInsertOperatorType(id);
          problem.setLatency(opr, 1);
          problem.setIncomingDelay(opr, 0.0);
          problem.setOutgoingDelay(opr, 0.0);
          auto rsrc = problem.getOrInsertResourceType(id);
          problem.setLimit(rsrc, 1);
          problem.addLinkedResourceType(callOp, rsrc);
          problem.setLinkedOperatorType(callOp, opr);
          return WalkResult::advance();
        })
        .Case<ShLIOp, ShRSIOp, ShRUIOp>([&](Operation *shOp) {
          bool constant =
              llvm::any_of(shOp->getOpOperands(), [](auto &operand) {
                if (isa<BlockArgument>(operand.get())) {
                  return false;
                }
                return isa<arith::ConstantOp>(operand.get().getDefiningOp());
              });
          // Constant shifts are free
          if (constant) {
            problem.setLinkedOperatorType(shOp, freeOpr);
          } else {
            problem.setLinkedOperatorType(shOp, combOpr);
          }
          return WalkResult::advance();
        })
        .Case<CmpIOp, arith::SelectOp, AddIOp, SubIOp, AndIOp, XOrIOp>(
            [&](Operation *combOp) {
              // Some known combinational ops.
              problem.setLinkedOperatorType(combOp, combOpr);
              return WalkResult::advance();
            })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Multiplier
          problem.setLinkedOperatorType(mcOp, mcOpr);
          return WalkResult::advance();
        })
        .Case<LoopScheduleBufferOp>([&](Operation *op) {
          problem.setLinkedOperatorType(op, seqOpr);
          return WalkResult::advance();
        })
        .Case<RemUIOp, RemSIOp, DivSIOp>([&](Operation *op) {
          // Divider ops
          auto bitwidth = op->getResult(0).getType().getIntOrFloatBitWidth();
          if (bitwidth != 32 && bitwidth != 64) {
            unsupported = op;
            return WalkResult::interrupt();
          }
          problem.setLinkedOperatorType(op, divOpr);
          return WalkResult::advance();
        })
        .Case<LoopScheduleStoreOp, AffineStoreOp>([&](Operation *memOp) {
          if (auto timing = memAccessTiming(memOp)) {
            Problem::OperatorType memOpr =
                problem.getOrInsertOperatorType(timing->name);
            problem.setLatency(memOpr, timing->latency);
            problem.setLinkedOperatorType(memOp, memOpr);
            setDelays(memOpr, timing->incomingDelay, timing->outgoingDelay,
                      memOp);
            return WalkResult::advance();
          }
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<LoopScheduleStoreOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          problem.setIncomingDelay(memOpr, 0.5);
          problem.setOutgoingDelay(memOpr, 0.5);
          return WalkResult::advance();
        })
        .Case<LoopScheduleLoadOp, AffineLoadOp>([&](Operation *memOp) {
          if (auto timing = memAccessTiming(memOp)) {
            Problem::OperatorType memOpr =
                problem.getOrInsertOperatorType(timing->name);
            problem.setLatency(memOpr, timing->latency);
            problem.setLinkedOperatorType(memOp, memOpr);
            setDelays(memOpr, timing->incomingDelay, timing->outgoingDelay,
                      memOp);
            return WalkResult::advance();
          }
          Value memRef = getMemref(memOp);
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);
          problem.setIncomingDelay(memOpr, 0.5);
          problem.setOutgoingDelay(memOpr, 0.5);
          return WalkResult::advance();
        })
        .Case<LoadInterface, StoreInterface>([&](Operation *op) {
          unsigned latency;
          std::string uniqueId;
          float incomingDelay = 0.0;
          float outgoingDelay = 0.0;
          if (auto loadOp = dyn_cast<loopschedule::LoadInterface>(*op)) {
            latency = loadOp.getLatency();
            uniqueId = loadOp.getUniqueId();
            incomingDelay = loadOp.getIncomingDelay();
            outgoingDelay = loadOp.getOutgoingDelay();
          } else {
            auto storeOp = cast<loopschedule::StoreInterface>(*op);
            latency = storeOp.getLatency();
            uniqueId = storeOp.getUniqueId();
            incomingDelay = storeOp.getIncomingDelay();
          }
          // Prefer characterized delays from the per-memory operator when
          // OperatorAllocation stamped one; the interface methods are the
          // uncalibrated fallback. Latency stays with the port type (it is
          // authoritative for the FSM lowering), and the operator-type key
          // stays the per-port uniqueId so port/resource semantics are
          // unchanged.
          if (auto timing = memAccessTiming(op)) {
            incomingDelay = timing->incomingDelay;
            if (isa<loopschedule::LoadInterface>(*op))
              outgoingDelay = timing->outgoingDelay;
          }
          // A DECLARED-ZERO-LATENCY dynamic load is an FWFT stream pop
          // (`amc.burst_pop`): the beat is combinationally valid on the pop
          // cycle, and a PIPELINE forwards that raw port data straight to a
          // same-stage consumer, so latency 0 is exactly right there.
          //
          // In a SEQUENTIAL frame `lowerSeqDynAccess` maps the result to a
          // `_cap` register instead, because the frame may stall on a SIBLING
          // access in the same cycle and the beat has to survive it. `_cap` is
          // written at the END of the pop cycle, which is fine for the
          // consumers a sequential frame usually has: a launched child's start
          // pulse fires in the pop cycle and its body runs the next one, so it
          // reads the register after it has been written. It is NOT fine for a
          // consumer that COMMITS IN ITS OWN CYCLE, and the one that does is
          // another memory access in the same frame cycle: its address is
          // driven combinationally out of `_cap`, so it issues against the
          // PREVIOUS beat. That is exactly how the in-loop dynamic-base
          // request took each row's address from the row before it — silently,
          // every iteration, `out[i] = sum(A[idx[i-1]])`.
          //
          // Give the pop a scheduling latency of 1 in that case, which puts
          // the consumer one cycle later where the register is readable — the
          // invariant every latency >= 1 dynamic access already has (consumers
          // land at `offset + latency`, `_cap` is readable at
          // `offset + latency`). Deliberately NOT applied when no such
          // consumer exists: bumping unconditionally also moves launches that
          // were correct at cycle 0, costing a cycle per outer iteration and
          // rewriting the RTL of kernels this has nothing to do with
          // (outer_product's `a[i]`).
          //
          // The FSM's budget arithmetic (`lastC = offset + (lat ? lat - 1 : 0)`)
          // is identical at 0 and 1, so the stall and capture hardware do not
          // change shape; only the consumer's cycle moves.
          if (sequentialRegion && latency == 0)
            if (auto hwLoad = dyn_cast<HWLoadLoweringInterface>(op))
              if (hwLoad.getReadLatency() == 0 && hwLoad.requiresReadEnable() &&
                  op->getNumResults() == 1 &&
                  feedsSameCycleAccess(op->getResult(0)))
                latency = 1;
          Problem::OperatorType portOpr =
              problem.getOrInsertOperatorType(uniqueId);
          problem.setLatency(portOpr, latency);
          problem.setLinkedOperatorType(op, portOpr);
          setDelays(portOpr, incomingDelay, outgoingDelay, op);

          return WalkResult::advance();
        })
        .Case<loopschedule::SchedulableInterface>([&](Operation *op) {
          auto schedOp = cast<SchedulableInterface>(op);
          auto latency = schedOp.getOpLatency();
          auto limitOpt = schedOp.getOpLimit();
          Problem::OperatorType opr =
              problem.getOrInsertOperatorType(schedOp.getUniqueId());
          problem.setLatency(opr, latency);
          if (limitOpt.has_value()) {
            auto rsrc = problem.getOrInsertResourceType(schedOp.getUniqueId());
            problem.setLimit(rsrc, limitOpt.value());
            problem.addLinkedResourceType(op, rsrc);
          }
          problem.setLinkedOperatorType(op, opr);
          setDelays(opr, schedOp.getIncomingDelay(),
                    schedOp.getOutgoingDelay(), op);

          return WalkResult::advance();
        })
        .Case<LoopInterface>([&](Operation *loopOp) {
          problem.setLinkedOperatorType(loopOp, loopOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return op->emitError("unsupported operation ") << *unsupported;

  // Second walk: if any op carries `loopschedule.operator = @name` and
  // `@name`'s `oplib.operator` entry declares a `limit`, attach a resource
  // of that name with the declared instance limit so the modulo/shared-
  // operator scheduler respects the cap. This is the single place where
  // oplib-declared resource limits enter the scheduling problem — covers
  // both arith operators (e.g. `i32_muli_l4 limit<2>`) and memory
  // operators (`mem_<func>_<idx> limit<2>` emitted by
  // `OperatorAllocation`).
  loopBody.walk([&](Operation *op) {
    auto oprAttr = op->getAttrOfType<SymbolRefAttr>("loopschedule.operator");
    if (!oprAttr)
      return;
    StringRef name = oprAttr.getLeafReference();
    auto limit = operatorLibraryAnalysis->getOperatorLimit(name);
    if (!limit)
      return;
    auto rsrc = problem.getOrInsertResourceType(name);
    problem.setLimit(rsrc, *limit);
    problem.addLinkedResourceType(op, rsrc);
  });

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult SCFToLoopSchedulePass::solveChainingModuloProblem(
    scf::WhileOp &loop, ChainingModuloProblem &problem, float cycleTime) {
  // Scheduling analyis only considers the innermost loop nest for now.

  std::optional<int32_t> ii;
  if (auto iiAttr = loop->getAttrOfType<IntegerAttr>("hls.pipeline"))
    ii = iiAttr.getInt();

  LLVM_DEBUG(loop.dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(for (auto *op
                  : problem.getOperations()) {
    // if (auto parent = op->getParentOfType<LoopInterface>(); parent)
    //   continue;
    llvm::dbgs() << "Chaining Modulo scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr->getAttr();
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    auto maybeRsrcs = problem.getLinkedResourceTypes(op);
    if (maybeRsrcs.has_value()) {
      for (auto rsrc : *maybeRsrcs)
        llvm::dbgs() << "\n  resource = " << rsrc.getAttr()
                     << " limit = " << problem.getLimit(rsrc);
    }
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  });

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = loop.getAfterBody()->getTerminator();
  if (failed(scheduleSimplex(problem, anchor, cycleTime, ii)))
    return failure();

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Verify II
  if (ii.has_value() && problem.getInitiationInterval() != ii) {
    return loop.emitError(
        "Failed to schedule for desired II of " + std::to_string(ii.value()) +
        ", minimum II is " +
        std::to_string(problem.getInitiationInterval().value()));
  }

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    for (auto *op : problem.getOperations()) {
      if (auto parent = op->getParentOfType<LoopInterface>(); parent)
        continue;
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    }
  });

  return success();
}

LogicalResult SCFToLoopSchedulePass::solveChainingSharedOperatorsProblem(
    Region &region, ChainingSharedOperatorsProblem &problem, float cycleTime) {

  LLVM_DEBUG(region.getParentOp()->dump());

  // Optionally debug problem inputs.
  LLVM_DEBUG(for (auto op
                  : problem.getOperations()) {
    llvm::dbgs() << "Chaining Shared Operator scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr->getAttr();
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    auto maybeRsrcs = problem.getLinkedResourceTypes(op);
    if (maybeRsrcs.has_value()) {
      for (auto rsrc : *maybeRsrcs)
        llvm::dbgs() << "\n  resource = " << rsrc.getAttr()
                     << " limit = " << problem.getLimit(rsrc);
    }
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { "
                     << "source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  });

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = region.back().getTerminator();
  if (failed(scheduleSimplex(problem, anchor, cycleTime)))
    return failure();

  // The simplex only optimizes the anchor; complete the solution with the
  // lowering-aware secondary objectives (states, then registers, then
  // determinism) so the rest of the schedule is chosen by stated criteria
  // rather than by the final simplex basis.
  applySecondaryScheduleObjectives(problem, anchor);

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

/// Create the pipeline op for a loop nest.
void SCFToLoopSchedulePass::computePipelineStageRegisters(
    ArrayRef<unsigned> startTimes,
    DenseMap<unsigned, SmallVector<Operation *>> &startGroups,
    CyclicProblem &problem, Operation *funcReturnSentinel,
    SmallVectorImpl<SmallVector<Value>> &registerValues,
    DenseMap<Value, std::pair<unsigned, unsigned>> &pipeTimes) {
  for (auto startTime : startTimes) {
    auto &group = startGroups[startTime];

    for (unsigned i = registerValues.size(); i <= startTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());

    for (auto *op : group) {
      // Walk users (including predicate users) and find the latest one;
      // values consumed past their producer's stage need to be registered
      // forward through every intervening stage.
      SmallVector<Operation *> users(op->getUsers().begin(),
                                      op->getUsers().end());
      for (auto res : op->getResults())
        users.append(predicateUse.lookup(res));

      bool consumedByReturn = false;
      unsigned pipeEndTime = 0;
      for (auto *user : users) {
        if (funcReturnSentinel && user == funcReturnSentinel) {
          consumedByReturn = true;
          continue;
        }
        auto utOpt = problem.getStartTime(user);
        if (!utOpt.has_value())
          continue;
        unsigned userStartTime = *utOpt;
        if (userStartTime > startTime)
          pipeEndTime = std::max(pipeEndTime, userStartTime);
      }

      if (op->getUsers().empty() && !consumedByReturn &&
          llvm::none_of(op->getResults(),
                        [&](Value v) { return predicateUse.contains(v); }))
        continue;

      for (auto res : op->getResults())
        pipeTimes[res] = std::pair(startTime, pipeEndTime);

      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Each result needs to live in registerValues[startTime] if any of
      // its users — direct or predicate — is outside this stage's group.
      for (auto result : op->getResults()) {
        bool registered = false;
        for (auto *user : result.getUsers()) {
          if (!llvm::is_contained(group, user)) {
            registerValues[startTime].push_back(result);
            registered = true;
            break;
          }
        }
        if (!registered) {
          for (auto *user : predicateUse.lookup(result)) {
            if (!llvm::is_contained(group, user)) {
              registerValues[startTime].push_back(result);
              break;
            }
          }
        }
      }

      // Forward through the intermediate stages so the value is visible
      // past its producer's latency window.
      unsigned firstUse = std::max(
          startTime + 1,
          startTime +
              *problem.getLatency(*problem.getLinkedOperatorType(op)));
      for (unsigned i = firstUse; i < pipeEndTime; ++i)
        for (auto result : op->getResults())
          registerValues[i].push_back(result);
    }
  }
}

LoopScheduleAtOp SCFToLoopSchedulePass::emitOnePipelineStage(
    ImplicitLocOpBuilder &builder, unsigned startTime,
    ArrayRef<Operation *> group, ArrayRef<Value> registerValuesAtStage,
    ArrayRef<Type> registerTypesAtStage,
    MutableArrayRef<IRMapping> stageValueMaps, CyclicProblem &problem,
    DominanceInfo &dom) {
  SmallVector<Operation *> sortedGroup(group.begin(), group.end());
  llvm::sort(sortedGroup, [&](Operation *a, Operation *b) {
    return dom.dominates(a, b);
  });

  auto startTimeAttr =
      builder.getIntegerAttr(builder.getIntegerType(64), startTime);
  auto stage = builder.create<LoopScheduleAtOp>(
      SmallVector<Type>(registerTypesAtStage.begin(),
                        registerTypesAtStage.end()),
      startTimeAttr);
  auto &stageBlock = stage.getBodyBlock();
  auto *stageTerminator = stageBlock.getTerminator();
  builder.setInsertionPointToStart(&stageBlock);

  // Clone each scheduled op into the stage. If the op is predicated,
  // wrap the clone in a `loopschedule.if` sourced from the predicate.
  for (auto *op : sortedGroup) {
    OpBuilder::InsertionGuard g(builder);
    LoopScheduleIfOp ifOp;
    if (predicateMap.contains(op)) {
      Value cond = predicateMap.lookup(op);
      if (stageValueMaps[startTime].contains(cond))
        cond = stageValueMaps[startTime].lookup(cond);
      ifOp = builder.create<LoopScheduleIfOp>(op->getLoc(),
                                               op->getResultTypes(), cond);
      builder.setInsertionPointToStart(&ifOp.getBody().front());
    }
    auto *newOp = builder.clone(*op, stageValueMaps[startTime]);
    dependenceAnalysis->replaceOp(op, newOp);
    if (predicateMap.contains(op)) {
      if (!newOp->getResults().empty())
        builder.create<LoopScheduleYieldOp>(op->getLoc(),
                                             newOp->getResults());
      newOp = ifOp;
    }
    // Subsequent ops in this stage must consume the cloned values.
    for (auto result : op->getResults())
      stageValueMaps[startTime].map(
          result, newOp->getResult(result.getResultNumber()));
  }

  // Forward registered values into the stage terminator and re-map them
  // for the destination stage. The destination is the next stage by
  // default; for multi-cycle ops whose result is yielded by their
  // producing stage, the destination skips ahead by `latency` cycles so
  // consumers see the wrapper output rather than the just-issued input.
  SmallVector<Value> stageOperands;
  unsigned resIndex = 0;
  for (auto res : registerValuesAtStage) {
    stageOperands.push_back(stageValueMaps[startTime].lookup(res));
    unsigned destTime = startTime + 1;
    if (!isa<BlockArgument>(res)) {
      unsigned latency = *problem.getLatency(
          *problem.getLinkedOperatorType(res.getDefiningOp()));
      if (*problem.getStartTime(res.getDefiningOp()) == startTime &&
          latency > 1)
        destTime = startTime + latency;
    }
    destTime =
        std::min((unsigned)(stageValueMaps.size() - 1), destTime);
    stageValueMaps[destTime].map(res, stage.getResult(resIndex++));
  }
  stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                  stageOperands);
  return stage;
}

/// A DRAINABLE STREAM POP: an FWFT beat consumer (dynamic, latency 0,
/// ordered reads) on a face whose only other users are pops and BARRIER
/// stores outside the loop — the abort/drain (amc.burst_abort) that
/// retires whatever tail an early exit strands. Such a pop is legal in a
/// pipelined data-dependent-exit while: the ghost iteration issues
/// nothing new on the bus (the request is hoisted — an in-loop request is
/// a dynamic STORE in the body and still refuses), a ghost pop merely
/// drains one beat early, and the face's abort retires the rest before
/// anything else touches the bundle.
static bool isDrainableStreamPop(Operation *op, mlir::scf::WhileOp loop) {
  auto load = dyn_cast<loopschedule::LoadInterface>(op);
  if (!load || !load.isDynamic() || load.getLatency() != 0 ||
      !load.readsAreOrdered())
    return false;
  bool sawAbort = false;
  for (Operation *user : load.getMemoryValue().getUsers()) {
    if (auto st = dyn_cast<loopschedule::StoreInterface>(user)) {
      if (!st.isBarrier() || loop->isAncestor(user))
        return false;
      sawAbort = true;
    } else if (auto ld = dyn_cast<loopschedule::LoadInterface>(user)) {
      if (!(ld.isDynamic() && ld.getLatency() == 0))
        return false;
    } else {
      return false;
    }
  }
  return sawAbort;
}

LogicalResult
SCFToLoopSchedulePass::createLoopSchedulePipeline(scf::WhileOp &loop,
                                                  CyclicProblem &problem,
                                                  Value condValue) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);

  // A pipeline launches a new iteration every II cycles from cycle 0, and
  // the continuation is the launch decision. A condition scheduled at cycle
  // 0 gates its own iteration's entry, so nothing speculative ever runs. A
  // condition scheduled at cycle c, 0 < c <= II, is decided in the same
  // cycle the NEXT launch happens: the lowering gates that launch on it
  // combinationally, so exactly ONE iteration — the "ghost" launched past
  // the loop's first decided-false — is ever in flight speculatively, and
  // the lowering's kill chain squashes its observable effects with its own
  // (false) condition (see the late-condition handling in
  // LoopScheduleToFSM's lowerPipelineChild). find_first's data-dependent
  // exit — `found` set from a latency-1 memory read, condition at cycle 1,
  // II 1 — is the canonical legal case.
  //
  // That leaves three things the kill chain cannot save, refused here with
  // the schedule in hand (start times exist nowhere else):
  //
  //  * c > II — a second speculative iteration launches before the first
  //    decided-false, and ITS condition is computed from iter_args the
  //    ghost already corrupted, so it cannot be trusted to squash anything.
  //  * an observable effect scheduled at a cycle < c — it commits before
  //    the condition that should have squashed it exists. (Reads from
  //    static ports are harmless and stay legal; this is about stores.)
  //  * dynamic-latency accesses anywhere in the body — the ghost's request
  //    is already on the bus when the exit resolves, and an unconsumed
  //    response beat wedges the bundle. Abandonment/quiescence is engine
  //    work that does not exist yet (RESULTS.md gap 4).
  //
  // The ghost also updates the carried values before its condition
  // resolves, so the loop's RESULTS are corrupted when the condition is
  // late; refuse when they are used. (`--promote-scf-scalars` loops carry
  // scratch state whose final values nothing reads, so this costs no
  // real kernel today.)
  if (Operation *condOp = condValue.getDefiningOp())
    if (problem.hasOperation(condOp)) {
      auto start = problem.getStartTime(condOp);
      unsigned condStart = start ? *start : 0;
      unsigned ii = problem.getInitiationInterval().value_or(1);
      // Root cause first: with a dynamic-latency access in the body the
      // continuation is late by the BUS's latency, so the cycle-count
      // refusal below would fire too — with advice ("restructure the exit")
      // that no restructuring can follow. Name the access instead.
      if (condStart > 0) {
        WalkResult dynCheck = loop.getAfter().walk([&](Operation *op) {
          if (!isDynamicLatencyOp(*op))
            return WalkResult::advance();
          if (isDrainableStreamPop(op, loop))
            return WalkResult::advance();
          InFlightDiagnostic diag = loop.emitOpError(
              "cannot pipeline a loop with dynamic-latency memory "
              "accesses and a late-deciding continuation");
          diag.attachNote(op->getLoc())
              << "a speculative iteration's request would already be "
                 "on the bus when the exit resolves; abandoning it "
                 "needs engine support that does not exist";
          return WalkResult::interrupt();
        });
        if (dynCheck.wasInterrupted())
          return failure();
      }
      if (condStart > ii) {
        InFlightDiagnostic diag = loop.emitOpError(
            "cannot pipeline a loop whose continuation is not decided in "
            "time to gate the next launch");
        diag.attachNote(condOp->getLoc())
            << "the condition is scheduled at cycle " << condStart
            << " but a new iteration launches every " << ii
            << " cycle(s), so more than one speculative iteration would be "
               "in flight past the exit — and only the first one's "
               "condition is trustworthy";
        diag.attachNote()
            << "run this loop sequentially instead, or restructure it so "
               "the exit condition is decided within one initiation "
               "interval";
        return failure();
      }
      if (condStart > 0) {
        // Observable effects must not commit before the condition that
        // squashes the ghost exists.
        auto isObservableEffect = [](Operation &op) {
          if (isa<StoreInterface>(&op))
            return true;
          if (isa<HWStoreLoweringInterface>(&op))
            return true;
          if (auto memOp = dyn_cast<MemoryEffectOpInterface>(&op))
            return memOp.hasEffect<MemoryEffects::Write>();
          return false;
        };
        WalkResult effectCheck =
            loop.getAfter().walk([&](Operation *op) {
              if (!isObservableEffect(*op) || !problem.hasOperation(op))
                return WalkResult::advance();
              // A drainable pop's "effect" is FIFO advance — the ghost's
              // pop drains one beat the abort would otherwise retire.
              if (isDrainableStreamPop(op, loop))
                return WalkResult::advance();
              auto effectStart = problem.getStartTime(op);
              if (effectStart && *effectStart < condStart) {
                InFlightDiagnostic diag = loop.emitOpError(
                    "cannot pipeline a loop with an observable effect "
                    "scheduled before its continuation is decided");
                diag.attachNote(op->getLoc())
                    << "this effect commits at cycle " << *effectStart
                    << ", before the cycle-" << condStart
                    << " condition could squash the speculative iteration "
                       "that issues it";
                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            });
        if (effectCheck.wasInterrupted())
          return failure();
        if (!llvm::all_of(loop.getResults(),
                          [](Value r) { return r.use_empty(); })) {
          InFlightDiagnostic diag = loop.emitOpError(
              "cannot pipeline a loop whose results are used when its "
              "continuation is decided late");
          diag.attachNote()
              << "the speculative iteration past the exit updates the "
                 "carried values before its condition resolves, so the "
                 "loop's results would hold corrupted values";
          return failure();
        }
      }
    }

  builder.setInsertionPointToStart(
      &loop->getParentOfType<FuncOp>().getBody().front());

  // Create Values for the loop's lower and upper bounds.
  // Value lowerBound = loop.getLowerBound();
  // Value upperBound = loop.getUpperBound();
  // Value step = loop.getStep();

  builder.setInsertionPoint(loop);

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = loop.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  // iterArgs.push_back(lowerBound);
  iterArgs.append(loop.getInits().begin(), loop.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount =
          loop->getAttrOfType<IntegerAttr>("loopschedule.trip_count"))
    tripCountAttr = tripCount;

  auto pipeline = builder.create<LoopSchedulePipelineOp>(
      resultTypes, ii, tripCountAttr, iterArgs);

  // Record the schedule's total iteration latency: the anchor (terminator)
  // is constrained after every op's end time — including result-less store
  // commits — so its solved start time is the cycle by which an iteration
  // has fully committed. The FSM sizes the pipeline's `done` from this
  // rather than from the stage-list length (tail stages carry no live
  // values and are not load-bearing).
  if (auto anchorTime =
          problem.getStartTime(loop.getAfterBody()->getTerminator()))
    pipeline.setLatencyAttr(builder.getI64IntegerAttr(*anchorTime));

  // Add the non-yield and non-if operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<YieldOp, IfOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // Nested loops are not supported yet.
  assert(iterArgs.size() == loop.getAfterBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    valueMap.map(loop.getAfterBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));
  }

  // Create the stages.
  Block &stagesBlock = pipeline.getStagesBlock();
  builder.setInsertionPointToStart(&stagesBlock);

  // Iterate in order of the start times.
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());

  // Keys for translating values in each stage
  SmallVector<SmallVector<Value>> registerValues;
  SmallVector<SmallVector<Type>> registerTypes;

  // The maps that ensure a stage uses the correct version of a value
  SmallVector<IRMapping> stageValueMaps;

  // For storing the range of stages an operation's results need to be valid for
  DenseMap<Value, std::pair<unsigned, unsigned>> pipeTimes;

  DenseSet<unsigned> newStartTimes(startTimes.begin(), startTimes.end());
  computePipelineStageRegisters(startTimes, startGroups, problem,
                                 /*funcReturnSentinel=*/nullptr,
                                 registerValues, pipeTimes);

  for (auto it : enumerate(loop.getAfter().getArguments())) {
    auto iterArg = it.value();
    if (iterArg.getUsers().empty())
      continue;

    unsigned startPipeTime = 0;

    auto *term = loop.getAfterBody()->getTerminator();
    auto &termOperand = term->getOpOperand(it.index());
    auto *definingOp = termOperand.get().getDefiningOp();
    assert(definingOp != nullptr);
    startPipeTime = *problem.getStartTime(definingOp);

    unsigned pipeEndTime = 0;
    for (auto *user : iterArg.getUsers()) {
      unsigned userStartTime = *problem.getStartTime(user);
      if (userStartTime >= startPipeTime) {
        pipeEndTime = std::max(pipeEndTime, userStartTime);
      }
    }

    // Do not need to pipe result if there are no later uses
    if (startPipeTime >= pipeEndTime)
      continue;

    // Make sure a stage exists for every time between startTime and
    // pipeEndTime
    // for (unsigned i = startTime; i < pipeEndTime; ++i)
    //   newStartTimes.insert(i);

    // Insert the range of pipeline stages the value needs to be valid for
    pipeTimes[iterArg] = std::pair(startPipeTime, pipeEndTime);

    // Add register stages for each time slice we need to pipe to
    for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
      registerValues.push_back(SmallVector<Value>());

    // Keep a collection of this stages results as keys to our valueMaps
    registerValues[startPipeTime].push_back(iterArg);

    // Other stages that use the value will need these values as keys too
    unsigned firstUse = startPipeTime + 1;
    for (unsigned i = firstUse; i < pipeEndTime; ++i) {
      registerValues[i].push_back(iterArg);
    }
  }

  // Ensure the condition value gets registered as a stage result. Its only
  // consumer is the loopschedule.terminator (not yet created), so the forward-
  // piping loop above may not have added it.
  {
    Operation *condOp = condValue.getDefiningOp();
    unsigned condStartTime = *problem.getStartTime(condOp);
    for (unsigned i = registerValues.size(); i <= condStartTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());
    if (!llvm::is_contained(registerValues[condStartTime], condValue))
      registerValues[condStartTime].push_back(condValue);
    pipeTimes[condValue] = std::pair(condStartTime, condStartTime);
  }

  // Now make register Types and stageValueMaps
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    if (!registerValues[i].empty()) {
      newStartTimes.insert(i);
    }
    SmallVector<mlir::Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());

    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  startTimes.clear();
  startTimes.append(newStartTimes.begin(), newStartTimes.end());
  llvm::sort(startTimes);
  SmallVector<bool> iterArgNeedsForwarding;
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    iterArgNeedsForwarding.push_back(false);
  }
  // Track which stage result holds the condition value, set during stage
  // creation below once the condition op's stage is lowered.
  Value pipelineCondResult;

  // Create stages along with maps
  for (auto startTime : startTimes) {
    builder.setInsertionPointToEnd(&stagesBlock);
    auto stage = emitOnePipelineStage(
        builder, startTime, startGroups[startTime], registerValues[startTime],
        registerTypes[startTime], stageValueMaps, problem, dom);

    // If this stage contains the condition value, record its stage result
    // for the terminator built after the stages loop completes.
    for (unsigned regIdx = 0; regIdx < registerValues[startTime].size();
         ++regIdx) {
      if (registerValues[startTime][regIdx] == condValue) {
        pipelineCondResult = stage.getResult(regIdx);
        break;
      }
    }
  }

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;

  for (auto it :
       llvm::enumerate(loop.getAfterBody()->getTerminator()->getOperands())) {
    unsigned i = it.index();
    Value value = it.value();
    unsigned lookupTime =
        std::min((unsigned)(stageValueMaps.size() - 1),
                 (unsigned)(pipeTimes[value].first + ii.getInt()));

    Value newValue = stageValueMaps[lookupTime].lookup(value);
    termIterArgs.push_back(newValue);
    termResults.push_back(newValue);

    // Emit an iter_arg_update inside the stage that produced the new value.
    // The LHS is the pipeline's iter-arg block argument for position `i`; the
    // RHS is the register-op operand that corresponds to `newValue` (the
    // stage result is not visible inside the stage itself).
    if (auto stage = newValue.getDefiningOp<LoopScheduleAtOp>()) {
      auto yieldOp = stage.getYieldOp();
      unsigned resultIdx = cast<OpResult>(newValue).getResultNumber();
      Value inside = yieldOp->getOperand(resultIdx);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(yieldOp);
      builder.create<LoopScheduleIterArgUpdateOp>(
          stage.getLoc(), pipeline.getStagesBlock().getArgument(i), inside);
    }
  }

  // Build the loopschedule.terminator with the condition and results.
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<LoopScheduleTerminatorOp>(pipelineCondResult, termResults,
                                           ValueRange{});

  // Post-processing: wrap any dynamic-latency op (ops carrying a
  // `loopschedule.dynamic` unit attr) in a `loopschedule.launch` at its
  // issue stage, and insert a `loopschedule.expect` in the stage that
  // sits `latency(op)` cycles later. The dynamic marker tells downstream
  // FSM lowering to stall the pipeline on that expect's handle if the
  // op's completion signal isn't asserted by the expected cycle.
  wrapDynamicOpsInPipeline(pipeline, problem);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < loop.getNumResults(); ++i)
    loop.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  dependenceAnalysis->replaceOp(loop, pipeline);

  loop.walk([&](Operation *op) {
    if (!isa<scf::IfOp>(op)) {
      assert(!dependenceAnalysis->containsOp(op));
    }
  });

  // Remove the loop nest from the IR.
  loop.walk([&](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

/// Build the pipeline stages for a function-level pipeline. Adapted from
/// `createLoopSchedulePipeline` with the loop-only pieces stripped (no iter
/// args, no condition value, no `LoopSchedulePipelineOp` wrapper, no
/// `LoopScheduleTerminatorOp`). Stages live as direct children of the
/// `loopschedule.func_pipeline` body, between the preserved-static prelude
/// (allocations / constants / init ops) and the `loopschedule.return`.
LogicalResult SCFToLoopSchedulePass::createFuncLoopSchedulePipeline(
    FuncOp funcOp, ChainingModuloProblem &problem) {
  ImplicitLocOpBuilder builder(funcOp.getLoc(), funcOp);
  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  // Create the new func_pipeline op alongside the original.
  builder.setInsertionPointAfter(funcOp);
  auto newOp = builder.create<LoopScheduleFuncPipelineOp>(
      funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType(), ii);
  for (auto attr : funcOp->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name == SymbolTable::getSymbolAttrName() ||
        name == funcOp.getFunctionTypeAttrName() || name == "II" ||
        name == "hls.pipeline")
      continue;
    // The frontend's discardable `amc.control_interface` marker becomes the
    // op's inherent `control` clause.
    if (name == "amc.control_interface") {
      newOp.setControlInterfaceAttr(cast<StringAttr>(attr.getValue()));
      continue;
    }
    newOp->setAttr(attr.getName(), attr.getValue());
  }

  // Move the function body into the new op's body. Block args (function
  // args) travel with the block.
  newOp.getBody().takeBody(funcOp.getBody());
  Block &body = newOp.getBody().front();
  auto *funcReturn = body.getTerminator();

  // Replace func.return → loopschedule.return up front so the stages we
  // emit can be placed before it.
  auto returnLoc = funcReturn->getLoc();
  SmallVector<Value> returnOperands(funcReturn->getOperands().begin(),
                                     funcReturn->getOperands().end());
  builder.setInsertionPoint(funcReturn);
  auto retOp = builder.create<LoopScheduleReturnOp>(returnLoc, returnOperands);
  funcReturn->erase();
  funcReturn = retOp.getOperation();

  // Group scheduled ops by start time. Skip ops that don't get scheduled
  // (yields, predicate carriers, allocations, init ops).
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<YieldOp, IfOp, func::ReturnOp, LoopScheduleReturnOp,
            memref::AllocaOp, arith::ConstantOp, memref::AllocOp,
            AllocInterface>(op))
      continue;
    if (auto schedOp = dyn_cast<SchedulableInterface>(op))
      if (schedOp.isInitOp())
        continue;
    auto startTime = problem.getStartTime(op);
    if (!startTime.has_value())
      continue;
    startGroups[*startTime].push_back(op);
  }

  // valueMap holds rewrites from original op-results / block args to
  // pipeline-stage results. Function args trivially map to themselves
  // (they're the same block args after takeBody).
  IRMapping valueMap;

  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  DominanceInfo dom(getOperation());

  SmallVector<SmallVector<Value>> registerValues;
  SmallVector<SmallVector<Type>> registerTypes;
  SmallVector<IRMapping> stageValueMaps;
  DenseMap<Value, std::pair<unsigned, unsigned>> pipeTimes;

  DenseSet<unsigned> newStartTimes(startTimes.begin(), startTimes.end());
  computePipelineStageRegisters(startTimes, startGroups, problem,
                                 /*funcReturnSentinel=*/funcReturn,
                                 registerValues, pipeTimes);

  // Make sure values consumed by the return op are also registered through
  // to the last stage. Without this, the return's operands have no
  // valueMap entry past the producing stage.
  for (Value retOperand : funcReturn->getOperands()) {
    Operation *defOp = retOperand.getDefiningOp();
    if (!defOp || !problem.hasOperation(defOp))
      continue;
    unsigned defStart = *problem.getStartTime(defOp);
    unsigned lastStage = startTimes.back();
    pipeTimes[retOperand] = std::pair(defStart, lastStage);
    for (unsigned i = registerValues.size(); i <= lastStage; ++i)
      registerValues.push_back(SmallVector<Value>());
    if (!llvm::is_contained(registerValues[defStart], retOperand))
      registerValues[defStart].push_back(retOperand);
    unsigned lat =
        *problem.getLatency(*problem.getLinkedOperatorType(defOp));
    unsigned firstUse = std::max(defStart + 1, defStart + lat);
    for (unsigned i = firstUse; i < lastStage; ++i)
      if (!llvm::is_contained(registerValues[i], retOperand))
        registerValues[i].push_back(retOperand);
  }

  // Now make register Types and stageValueMaps.
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    if (!registerValues[i].empty())
      newStartTimes.insert(i);
    SmallVector<Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());
    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  startTimes.clear();
  startTimes.append(newStartTimes.begin(), newStartTimes.end());
  llvm::sort(startTimes);

  // Emit each stage as a `loopschedule.at offset(...)` op directly inside
  // the func_pipeline body, just before the `loopschedule.return`.
  for (auto startTime : startTimes) {
    builder.setInsertionPoint(funcReturn);
    emitOnePipelineStage(builder, startTime, startGroups[startTime],
                          registerValues[startTime], registerTypes[startTime],
                          stageValueMaps, problem, dom);
  }

  // Rewrite the loopschedule.return operands to consume the final stage's
  // forwarded values.
  for (auto [i, operand] : llvm::enumerate(funcReturn->getOperands())) {
    if (auto mapped = stageValueMaps.back().lookupOrNull(operand))
      funcReturn->setOperand(i, mapped);
  }

  // Erase scheduled-op originals from the func body. Anything we cloned
  // into a stage has been replaced via the dependence-analysis update;
  // walking in reverse-iteration order avoids touching freed parents.
  SmallVector<Operation *> toErase;
  for (auto &startTime : startTimes)
    for (auto *op : startGroups[startTime])
      toErase.push_back(op);
  for (auto *op : llvm::reverse(toErase)) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  }

  funcOp.erase();
  return success();
}

/// Create loopschedule seq op for a sequential loop. Builds a
/// `LoopScheduleSequentialOp`, sets up the variant strategy, and delegates
/// the body lowering to `lowerSchedule`.
LogicalResult
SCFToLoopSchedulePass::createLoopScheduleSequential(scf::WhileOp &loop,
                                                    Problem &problem,
                                                    Value condValue) {
  ImplicitLocOpBuilder builder(loop.getLoc(), loop);
  builder.setInsertionPoint(loop);

  auto *anchor = loop.getAfterBody()->getTerminator();
  TypeRange resultTypes = loop.getResultTypes();
  SmallVector<Value> iterArgs(loop.getInits().begin(), loop.getInits().end());

  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount =
          loop->getAttrOfType<IntegerAttr>("loopschedule.trip_count"))
    tripCountAttr = tripCount;

  auto sequential = builder.create<LoopScheduleSequentialOp>(
      loop.getLoc(), resultTypes, tripCountAttr, iterArgs);

  ScheduleStrategy S;
  S.root = loop.getOperation();
  S.scheduleBlock = &sequential.getScheduleBlock();
  S.containingRegion = &loop.getAfter();
  S.anchor = anchor;
  S.condValue = condValue;

  S.shouldSkipForGrouping = [](Operation *op) {
    return isa<YieldOp, IfOp>(op);
  };
  S.isTerminator = [loop](Operation *op) {
    return isa<YieldOp>(op) && op->getParentOp() == loop;
  };
  S.setFrameInsertion = [&sequential](OpBuilder &b) {
    b.setInsertionPointToStart(&sequential.getScheduleBlock());
  };

  IterArgSupport ia;
  for (auto arg : loop.getAfter().getArguments())
    ia.afterArgs.push_back(arg);
  for (auto arg : sequential.getScheduleBlock().getArguments())
    ia.scheduleBlockArgs.push_back(arg);
  // A `for`-derived while carries its induction variable as iter-arg 0; a
  // SOURCE `scf.while` may carry nothing at all (its state lives in memory
  // or in the condition's own operands). Guard the probe rather than
  // indexing an empty argument list.
  ia.inductionVarHasUsers =
      loop.getAfter().getNumArguments() > 0 &&
      !loop.getAfter().getArgument(0).getUsers().empty();
  S.iterArgs = std::move(ia);

  S.seedValueMap = [&](IRMapping &valueMap) {
    for (size_t i = 0; i < iterArgs.size(); ++i)
      valueMap.map(loop.getAfter().getArgument(i),
                   sequential.getScheduleBlock().getArgument(i));
  };

  S.finalize = [&](ImplicitLocOpBuilder &b,
                   ArrayRef<PendingLaunch> pendingLaunches,
                   Value condResult, ArrayRef<Value> termIterArgs,
                   ArrayRef<bool> /*iterArgUpdated*/, IRMapping & /*vm*/) {
    SmallVector<Value> termAwaitHandles;
    for (auto &pl : pendingLaunches)
      termAwaitHandles.push_back(pl.currentHandle);
    b.setInsertionPointToEnd(&sequential.getScheduleBlock());
    b.create<LoopScheduleTerminatorOp>(condResult, termIterArgs,
                                       termAwaitHandles);

    for (size_t i = 0; i < loop.getNumResults(); ++i)
      loop.getResult(i).replaceAllUsesWith(sequential.getResult(i));

    dependenceAnalysis->replaceOp(loop, sequential);

    loop.walk(
        [&](Operation *op) { assert(!dependenceAnalysis->containsOp(op)); });

    loop.walk([&](Operation *op) {
      op->dropAllUses();
      op->dropAllDefinedValueUses();
      op->dropAllReferences();
      op->erase();
    });
  };

  return lowerSchedule(S, problem);
}

int64_t opOrParentStartTime(Problem &problem, Operation *op) {
  Operation *currentOp = op;

  while (!isa<func::FuncOp>(currentOp)) {
    if (problem.hasOperation(currentOp)) {
      return problem.getStartTime(currentOp).value();
    }
    currentOp = currentOp->getParentOp();
  }
  op->emitOpError("Operation or parent does not have start time");
  return -1;
}

/// Merged driver shared by `createLoopScheduleSequential` and
/// `createFuncLoopSchedule`. The structure mirrors the previous sequential
/// implementation; variant points (terminator type, iter-arg threading, cond
/// op tracking, op-skip filter, and tail emission) flow through `S`.
LogicalResult SCFToLoopSchedulePass::lowerSchedule(ScheduleStrategy &S,
                                                    Problem &problem) {
  ImplicitLocOpBuilder builder(S.root->getLoc(), S.root);
  Operation *anchor = S.anchor;
  IRMapping valueMap;

  // Position the builder where new top-level frames go.
  S.setFrameInsertion(builder);
  S.seedValueMap(valueMap);

  // === Build start groups ===
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  unsigned endTime = 0;
  for (auto *op : problem.getOperations()) {
    if (S.shouldSkipForGrouping(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
    if (startTime > endTime)
      endTime = *startTime;
  }

  auto usedByOperation = [&](Operation *op, Operation *maybeUser) {
    return llvm::any_of(maybeUser->getOperands(), [&](Value operand) {
      if (operand.getDefiningOp() == op)
        return true;
      if (predicateUse.contains(operand))
        return true;
      return false;
    });
  };

  // hasLaterUse: walks into nested ops so it correctly accounts for users
  // that live inside an already-lowered Frame (the func-level case).
  auto hasLaterUse = [&](Operation *op, uint32_t resTime) {
    for (uint32_t i = resTime + 1; i <= endTime; ++i) {
      auto it = startGroups.find(i);
      if (it == startGroups.end())
        continue;
      for (auto *operation : it->second) {
        if (usedByOperation(op, operation))
          return true;
        auto wasInterrupted = operation->walk([&](Operation *inner) {
          if (usedByOperation(op, inner))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
        if (wasInterrupted.wasInterrupted())
          return true;
      }
    }
    return false;
  };

  auto valueHasLaterUse = [&](Value v, uint32_t resTime) {
    for (uint32_t i = resTime + 1; i <= endTime; ++i) {
      auto it = startGroups.find(i);
      if (it == startGroups.end())
        continue;
      for (auto *operation : it->second) {
        for (auto &operand : operation->getOpOperands()) {
          if (operand.get() == v)
            return true;
          if (predicateUse.contains(operand.get()))
            return true;
        }
      }
    }
    return false;
  };

  // === Reregister late-use load results ===
  for (auto *op : problem.getOperations()) {
    if (isa<LoopScheduleLoadOp>(op)) {
      auto startTime = problem.getStartTime(op);
      auto resTime = *startTime + 1;
      if (hasLaterUse(op, resTime) && !startGroups.contains(resTime))
        startGroups[resTime] = SmallVector<Operation *>();
    }
    if (auto load = dyn_cast<LoadInterface>(op)) {
      auto startTime = problem.getStartTime(op);
      auto latency = load.getLatency();
      auto resTime = *startTime + latency;
      if (hasLaterUse(op, resTime) && !startGroups.contains(resTime))
        startGroups[resTime] = SmallVector<Operation *>();
    }
  }

  // Sequential-only: if the induction-var iter-arg has users and the last
  // bucket contains a launch, append a trailing empty bucket so iter-arg
  // forwards have a landing site.
  if (S.iterArgs && S.iterArgs->inductionVarHasUsers) {
    bool containsLoop = false;
    for (auto *op : startGroups[endTime])
      if (isLaunchLikeOp(op)) {
        containsLoop = true;
        break;
      }
    if (containsLoop)
      startGroups[endTime + 1] = SmallVector<Operation *>();
  }

  // === Order start times and partition into phases ===
  SmallVector<unsigned> startTimes;
  for (const auto &group : startGroups)
    startTimes.push_back(group.first);
  llvm::sort(startTimes);

  auto isLaunchLike = [&](Operation *op) {
    return isLaunchLikeOp(op) ||
           (S.dynAccessesAsLaunches && isDynamicLatencyOp(*op));
  };
  auto phases = partitionPhasesByLaunch(startTimes, startGroups, isLaunchLike);

  DenseMap<uint32_t, size_t> bucketTimeToPhase;
  for (auto phaseIdx : llvm::seq<size_t>(0, phases.size()))
    for (auto t : phases[phaseIdx])
      bucketTimeToPhase[t] = phaseIdx;

  auto computeDynamicOpUserPhases = [&](Operation *dop,
                                         DenseSet<size_t> &userPhases,
                                         bool &consumedByTerminator) {
    userPhases.clear();
    consumedByTerminator = false;
    for (auto res : dop->getResults()) {
      for (auto *user : res.getUsers()) {
        if (S.isTerminator(user)) {
          consumedByTerminator = true;
          continue;
        }
        auto userStart = opOrParentStartTime(problem, user);
        if (userStart < 0)
          continue;
        auto it = bucketTimeToPhase.find((uint32_t)userStart);
        if (it != bucketTimeToPhase.end())
          userPhases.insert(it->second);
      }
    }
  };

  DominanceInfo dom(getOperation());
  DenseMap<uint32_t, SmallVector<Value>> reregisterValues;

  Value scheduleCondResult;
  Operation *condOp = S.condValue ? S.condValue.getDefiningOp() : nullptr;

  SmallVector<PendingLaunch> pendingLaunches;

  // === Per-phase loop ===
  for (auto phaseIdx : llvm::seq<size_t>(0, phases.size())) {
    auto &bucketTimes = phases[phaseIdx];
    uint32_t phaseBase = bucketTimes.front();

    SmallVector<BucketContent> buckets;
    for (auto t : bucketTimes) {
      BucketContent bc;
      bc.offset = t - phaseBase;
      for (auto *op : startGroups[t]) {
        if (isLaunchLike(op))
          bc.dynamicOps.push_back(op);
        else
          bc.staticOps.push_back(op);
      }
      buckets.push_back(std::move(bc));
    }

    // ===== Plan exports =====
    SmallVector<StaticExport> staticExports;
    SmallVector<IterArgExport> iterArgExports;
    SmallVector<std::pair<uint32_t, Value>> reregExports;
    SmallVector<unsigned> launchHandleIdx;
    SmallVector<Type> stepTypes;
    bool phaseHasCondOp = false;

    for (auto &bc : buckets) {
      for (auto *op : bc.staticOps) {
        bool needsReturn = false;
        SmallVector<Operation *, 10> users;
        users.append(op->getUsers().begin(), op->getUsers().end());
        for (auto res : op->getResults()) {
          auto predUsers = predicateUse.lookup(res);
          users.append(predUsers.begin(), predUsers.end());
        }
        for (auto *user : users) {
          auto *userOrAncestor =
              S.containingRegion->findAncestorOpInRegion(*user);
          if (!userOrAncestor)
            continue;
          if (!llvm::is_contained(bc.staticOps, userOrAncestor)) {
            needsReturn = true;
            break;
          }
        }
        if (op == condOp) {
          needsReturn = true;
          phaseHasCondOp = true;
        }
        if (needsReturn) {
          StaticExport se{&bc, op, (unsigned)stepTypes.size()};
          staticExports.push_back(se);
          stepTypes.append(op->getResultTypes().begin(),
                           op->getResultTypes().end());
        }

        if (S.iterArgs) {
          for (auto &operand : anchor->getOpOperands()) {
            auto iterArgNum = operand.getOperandNumber();
            auto iterArg = S.iterArgs->afterArgs[iterArgNum];
            if (operand.get().getDefiningOp() == op &&
                valueHasLaterUse(iterArg, phaseBase + bc.offset)) {
              IterArgExport iae;
              iae.bucket = &bc;
              iae.op = op;
              iae.iterArg = iterArg;
              iae.frameIdx = stepTypes.size();
              iterArgExports.push_back(iae);
              stepTypes.push_back(iterArg.getType());
            }
          }
        }
      }
    }

    for (auto &bc : buckets) {
      uint32_t bt = phaseBase + bc.offset;
      for (auto val : reregisterValues[bt]) {
        reregExports.emplace_back(bt, val);
        stepTypes.push_back(val.getType());
      }
    }

    SmallVector<std::pair<BucketContent *, Operation *>> launchesInPhase;
    for (auto &bc : buckets)
      for (auto *dop : bc.dynamicOps)
        launchesInPhase.emplace_back(&bc, dop);

    SmallVector<DenseSet<size_t>> launchUserPhasesInPhase;
    SmallVector<bool> launchTerminatorConsumedInPhase;
    for (auto &bl : launchesInPhase) {
      DenseSet<size_t> userPhases;
      bool terminatorConsumed = false;
      computeDynamicOpUserPhases(bl.second, userPhases, terminatorConsumed);
      launchUserPhasesInPhase.push_back(userPhases);
      launchTerminatorConsumedInPhase.push_back(terminatorConsumed);
    }

    SmallVector<std::optional<unsigned>> launchHandleIdxOpt;
    for (auto &bl : launchesInPhase) {
      (void)bl;
      launchHandleIdx.push_back(stepTypes.size());
      launchHandleIdxOpt.push_back(stepTypes.size());
      stepTypes.push_back(HandleType::get(builder.getContext()));
    }

    // === Pending services ===
    SmallVector<PendingService> pendingServices;
    for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
      PendingService ps;
      ps.pendingIdx = pi;
      ps.awaitHere = true;
      ps.forwardHere = false;
      for (auto &fwd : pl.forwards) {
        bool useInThisPhase = false;
        bool useOutsideThisPhase = false;
        for (auto *user : fwd.second.getUsers()) {
          if (S.isTerminator(user)) {
            useOutsideThisPhase = true;
            continue;
          }
          auto userStart = opOrParentStartTime(problem, user);
          if (userStart < 0)
            continue;
          auto it = bucketTimeToPhase.find((uint32_t)userStart);
          if (it == bucketTimeToPhase.end())
            continue;
          if (it->second == phaseIdx)
            useInThisPhase = true;
          else if (it->second > phaseIdx)
            useOutsideThisPhase = true;
        }
        if (!useInThisPhase && !useOutsideThisPhase)
          continue;
        ps.forwards.push_back(fwd);
        ps.forwardExportAsFrameResult.push_back(useOutsideThisPhase);
        ps.forwardExportFrameIdx.push_back(std::nullopt);
      }
      pendingServices.push_back(ps);
    }

    // Reserve frame-result slots for forwards that need to escape this frame.
    for (auto &ps : pendingServices) {
      if (!ps.awaitHere)
        continue;
      for (auto [idx, fwd] : llvm::enumerate(ps.forwards)) {
        if (!ps.forwardExportAsFrameResult[idx])
          continue;
        ps.forwardExportFrameIdx[idx] = stepTypes.size();
        stepTypes.push_back(fwd.second.getType());
      }
    }

    // === Build the frame ===
    auto frame = builder.create<LoopScheduleFrameOp>(stepTypes);

    SmallVector<Value> awaitYieldValues;
    SmallVector<Type> bodyBlockArgTypes;
    SmallVector<ServicedForwardInfo> servicedForwards;
    {
      Block &awaitBlock = frame.getAwaitRegion().emplaceBlock();
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&awaitBlock);
      for (auto [si, ps] : llvm::enumerate(pendingServices)) {
        if (!ps.awaitHere)
          continue;
        auto &pl = pendingLaunches[ps.pendingIdx];
        SmallVector<Type> forwardTypes;
        for (auto &fwd : ps.forwards)
          forwardTypes.push_back(fwd.second.getType());
        auto awaitOp = builder.create<LoopScheduleAwaitOp>(
            pl.currentHandle.getLoc(), forwardTypes,
            ValueRange{pl.currentHandle});
        if (!ps.forwards.empty()) {
          ServicedForwardInfo sfi;
          sfi.serviceIdx = si;
          sfi.firstBodyArgIdx = bodyBlockArgTypes.size();
          servicedForwards.push_back(sfi);
          for (auto r : awaitOp.getResults()) {
            awaitYieldValues.push_back(r);
            bodyBlockArgTypes.push_back(r.getType());
          }
        }
      }
      builder.create<LoopScheduleYieldOp>(awaitYieldValues);
    }

    Block &bodyBlock = frame.getBodyRegion().emplaceBlock();
    for (auto t : bodyBlockArgTypes)
      bodyBlock.addArgument(t, frame.getLoc());
    for (auto &sfi : servicedForwards) {
      auto &ps = pendingServices[sfi.serviceIdx];
      for (auto [i, fwd] : llvm::enumerate(ps.forwards)) {
        Value bodyArg = bodyBlock.getArgument(sfi.firstBodyArgIdx + i);
        valueMap.map(fwd.second, bodyArg);
      }
    }
    LoopScheduleYieldOp bodyYield;
    {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(&bodyBlock);
      bodyYield = builder.create<LoopScheduleYieldOp>();
    }

    SmallVector<Value> bodyYieldOperands(stepTypes.size(), Value());

    for (auto &ps : pendingServices) {
      if (!ps.forwardHere)
        continue;
      auto &pl = pendingLaunches[ps.pendingIdx];
      bodyYieldOperands[*ps.forwardFrameIdx] = pl.currentHandle;
    }

    for (auto &sfi : servicedForwards) {
      auto &ps = pendingServices[sfi.serviceIdx];
      for (auto [i, fwd] : llvm::enumerate(ps.forwards)) {
        if (!ps.forwardExportAsFrameResult[i])
          continue;
        Value bodyArg = bodyBlock.getArgument(sfi.firstBodyArgIdx + i);
        bodyYieldOperands[*ps.forwardExportFrameIdx[i]] = bodyArg;
      }
    }

    // ===== Per-bucket at emission =====
    enum AtSlotKind { Static, IterArg, Rereg };
    struct AtSlot {
      AtSlotKind kind;
      unsigned frameIdx;
      unsigned atIdx;
      StaticExport *se = nullptr;
      IterArgExport *iae = nullptr;
      Value reregVal;
    };

    for (auto &bc : buckets) {
      SmallVector<Type> atTypes;
      SmallVector<AtSlot> slots;

      for (auto &se : staticExports) {
        if (se.bucket != &bc)
          continue;
        AtSlot s;
        s.kind = Static;
        s.frameIdx = se.frameFirstIdx;
        s.atIdx = atTypes.size();
        s.se = &se;
        for (auto t : se.op->getResultTypes())
          atTypes.push_back(t);
        slots.push_back(s);
      }
      if (S.iterArgs) {
        for (auto &iae : iterArgExports) {
          if (iae.bucket != &bc)
            continue;
          AtSlot s;
          s.kind = IterArg;
          s.frameIdx = iae.frameIdx;
          s.atIdx = atTypes.size();
          s.iae = &iae;
          atTypes.push_back(iae.iterArg.getType());
          slots.push_back(s);
        }
      }
      uint32_t bt = phaseBase + bc.offset;
      for (auto &re : reregExports) {
        if (re.first != bt)
          continue;
        unsigned base = 0;
        for (auto &s : staticExports)
          base += s.op->getNumResults();
        if (S.iterArgs)
          base += iterArgExports.size();
        unsigned idx = 0;
        unsigned rrFrameIdx = 0;
        for (auto &re2 : reregExports) {
          if (&re2 == &re) {
            rrFrameIdx = base + idx;
            break;
          }
          ++idx;
        }
        AtSlot s;
        s.kind = Rereg;
        s.frameIdx = rrFrameIdx;
        s.atIdx = atTypes.size();
        s.reregVal = re.second;
        atTypes.push_back(re.second.getType());
        slots.push_back(s);
      }

      // A bucket with no static ops can still owe the frame re-registered
      // or iter-arg slots (e.g. a value that must pass through a phase whose
      // only compute is dynamic launches); its `at` is then a pure
      // pass-through yield. Skipping it would leave those frame slots as
      // null yield operands. Only buckets that contribute nothing at all
      // are skipped.
      if (bc.staticOps.empty() && slots.empty())
        continue;

      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(bodyYield);
      auto atOp = builder.create<LoopScheduleAtOp>(
          atTypes, builder.getI64IntegerAttr(bc.offset));
      auto &atBlock = atOp.getBodyBlock();
      auto *atTerm = atBlock.getTerminator();
      builder.setInsertionPointToStart(&atBlock);

      auto staticOps = bc.staticOps;
      llvm::sort(staticOps, [&](Operation *a, Operation *b) {
        return dom.dominates(a, b);
      });

      DenseMap<Operation *, Operation *> oldToNew;
      for (auto *op : staticOps) {
        OpBuilder::InsertionGuard g2(builder);
        LoopScheduleIfOp ifOp;
        if (predicateMap.contains(op)) {
          Value cond = predicateMap.lookup(op);
          cond = valueMap.lookupOrDefault(cond);
          ifOp = builder.create<LoopScheduleIfOp>(op->getLoc(),
                                                  op->getResultTypes(), cond);
          builder.setInsertionPointToStart(&ifOp.getBody().front());
        }
        auto *newOp = builder.clone(*op, valueMap);
        dependenceAnalysis->replaceOp(op, newOp);
        if (auto opr = problem.getLinkedOperatorType(op)) {
          unsigned lat = problem.getLatency(*opr).value_or(1);
          if (lat > 1)
            newOp->setAttr("loopschedule.cycle_latency",
                           builder.getI64IntegerAttr(lat));
        }
        if (predicateMap.contains(op)) {
          if (!newOp->getResults().empty())
            builder.create<LoopScheduleYieldOp>(op->getLoc(),
                                                newOp->getResults());
          newOp = ifOp;
        }
        oldToNew[op] = newOp;
        for (auto result : op->getResults())
          valueMap.map(result, newOp->getResult(result.getResultNumber()));
      }

      SmallVector<Value> atYieldOperands(atTypes.size(), Value());
      for (auto &s : slots) {
        if (s.kind == Static) {
          auto *newOp = oldToNew.lookup(s.se->op);
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            atYieldOperands[s.atIdx + i] = newOp->getResult(i);
        } else if (s.kind == IterArg) {
          atYieldOperands[s.atIdx] = valueMap.lookup(s.iae->iterArg);
        } else {
          atYieldOperands[s.atIdx] = valueMap.lookup(s.reregVal);
        }
      }
      atTerm->setOperands(atYieldOperands);

      if (S.iterArgs) {
        for (auto &s : slots) {
          if (s.kind != IterArg)
            continue;
          OpBuilder::InsertionGuard ig(builder);
          builder.setInsertionPoint(atTerm);
          Value insideAt = atTerm->getOperand(s.atIdx);
          unsigned argIdx =
              cast<BlockArgument>(s.iae->iterArg).getArgNumber();
          builder.create<LoopScheduleIterArgUpdateOp>(
              atOp.getLoc(), S.iterArgs->scheduleBlockArgs[argIdx], insideAt);
        }
      }

      for (auto &s : slots) {
        if (s.kind == Static) {
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            valueMap.map(s.se->op->getResult(i), atOp.getResult(s.atIdx + i));
        } else if (s.kind == Rereg) {
          valueMap.map(s.reregVal, atOp.getResult(s.atIdx));
        }
      }

      for (auto &s : slots) {
        if (s.kind == Static) {
          for (unsigned i = 0, e = s.se->op->getNumResults(); i < e; ++i)
            bodyYieldOperands[s.frameIdx + i] = atOp.getResult(s.atIdx + i);
        } else {
          bodyYieldOperands[s.frameIdx] = atOp.getResult(s.atIdx);
        }
      }
    }

    // ===== Per-launch wrapping at =====
    for (auto [launchIdx, blPair] : llvm::enumerate(launchesInPhase)) {
      BucketContent *bc = blPair.first;
      Operation *dop = blPair.second;
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPoint(bodyYield);

      SmallVector<Type> atResultTypes{HandleType::get(builder.getContext())};
      auto atOp = builder.create<LoopScheduleAtOp>(
          dop->getLoc(), TypeRange(atResultTypes),
          builder.getI64IntegerAttr(bc->offset));
      Block &atBlock = atOp.getBody().front();
      builder.setInsertionPointToStart(&atBlock);

      auto launch = builder.create<LoopScheduleLaunchOp>(
          dop->getLoc(), HandleType::get(builder.getContext()));
      Block &launchBlock = launch.getBody().emplaceBlock();
      {
        OpBuilder::InsertionGuard gg(builder);
        builder.setInsertionPointToEnd(&launchBlock);
        builder.create<LoopScheduleYieldOp>();
      }
      auto *launchYield = launchBlock.getTerminator();
      builder.setInsertionPointToStart(&launchBlock);

      auto *newOp = builder.clone(*dop, valueMap);
      dependenceAnalysis->replaceOp(dop, newOp);
      if (auto opr = problem.getLinkedOperatorType(dop)) {
        unsigned lat = problem.getLatency(*opr).value_or(1);
        if (lat > 1)
          newOp->setAttr("loopschedule.cycle_latency",
                         builder.getI64IntegerAttr(lat));
      }

      std::queue<Operation *> oldOps;
      dop->walk([&](Operation *op) { oldOps.push(op); });
      if (isa<LoopInterface>(newOp)) {
        newOp->walk([&](Operation *op) {
          Operation *oldOp = oldOps.front();
          dependenceAnalysis->replaceOp(oldOp, op);
          oldOps.pop();
        });
      }

      if (newOp->getNumResults() > 0)
        launchYield->setOperands(newOp->getResults());

      {
        OpBuilder::InsertionGuard gg(builder);
        auto *atYield = atBlock.getTerminator();
        atYield->setOperands(ValueRange{launch.getResult()});
      }

      for (auto [orig, clone] :
           llvm::zip(dop->getResults(), newOp->getResults()))
        valueMap.map(orig, clone);

      bodyYieldOperands[*launchHandleIdxOpt[launchIdx]] = atOp.getResult(0);
    }

    bodyYield->setOperands(bodyYieldOperands);

    // Reorder at children by offset.
    {
      SmallVector<Operation *> children;
      for (Operation &op : bodyBlock.without_terminator())
        children.push_back(&op);
      std::stable_sort(children.begin(), children.end(),
                       [](Operation *a, Operation *b) {
                         return cast<LoopScheduleAtOp>(a).getOffset() <
                                cast<LoopScheduleAtOp>(b).getOffset();
                       });
      for (Operation *op : children)
        op->moveBefore(bodyYield);
    }

    // Post-frame valueMap rebinding.
    for (auto &se : staticExports) {
      for (unsigned i = 0, e = se.op->getNumResults(); i < e; ++i)
        valueMap.map(se.op->getResult(i),
                     frame->getResult(se.frameFirstIdx + i));
      if (S.condValue && phaseHasCondOp && se.op == condOp) {
        unsigned condResNum = cast<OpResult>(S.condValue).getResultNumber();
        scheduleCondResult = frame->getResult(se.frameFirstIdx + condResNum);
      }
    }
    if (S.iterArgs) {
      for (auto &iae : iterArgExports)
        valueMap.map(iae.iterArg, frame->getResult(iae.frameIdx));
    }
    for (auto &ps : pendingServices) {
      if (!ps.awaitHere)
        continue;
      for (auto [idx, fwd] : llvm::enumerate(ps.forwards)) {
        if (!ps.forwardExportAsFrameResult[idx])
          continue;
        valueMap.map(fwd.second,
                     frame->getResult(*ps.forwardExportFrameIdx[idx]));
      }
    }
    {
      unsigned base = 0;
      for (auto &s : staticExports)
        base += s.op->getNumResults();
      if (S.iterArgs)
        base += iterArgExports.size();
      unsigned idx = 0;
      for (auto &re : reregExports) {
        valueMap.map(re.second, frame->getResult(base + idx));
        ++idx;
      }
    }

    // Update pendingLaunches for the next phase.
    SmallVector<PendingLaunch> nextPending;
    for (auto &ps : pendingServices) {
      if (ps.awaitHere)
        continue;
      auto pl = pendingLaunches[ps.pendingIdx];
      pl.currentHandle = frame->getResult(*ps.forwardFrameIdx);
      pl.remainingUserPhases.erase(phaseIdx);
      nextPending.push_back(pl);
    }
    for (auto [launchIdx, blPair] : llvm::enumerate(launchesInPhase)) {
      PendingLaunch pl;
      pl.origOp = blPair.second;
      pl.currentHandle = frame->getResult(*launchHandleIdxOpt[launchIdx]);
      for (auto res : blPair.second->getResults())
        pl.forwards.push_back({res.getResultNumber(), res});
      for (auto p : launchUserPhasesInPhase[launchIdx])
        if (p > phaseIdx)
          pl.remainingUserPhases.insert(p);
      pl.terminatorPending = launchTerminatorConsumedInPhase[launchIdx];
      nextPending.push_back(pl);
    }
    pendingLaunches = std::move(nextPending);

    // Compute reregister values for the next bucket time.
    for (auto &bc : buckets) {
      uint32_t bt = phaseBase + bc.offset;
      for (auto *op : bc.staticOps) {
        if (auto load = dyn_cast<LoopScheduleLoadOp>(op)) {
          if (hasLaterUse(op, bt + 1))
            reregisterValues[bt + 1].push_back(load.getResult());
        } else if (auto load = dyn_cast<LoadInterface>(op)) {
          auto latency = load.getLatency();
          if (hasLaterUse(op, bt + latency))
            reregisterValues[bt + latency].push_back(load.getResult());
        }
      }
    }
  }

  // === Iter-arg term collection (sequential only) ===
  SmallVector<Value> termIterArgs;
  SmallVector<bool> iterArgUpdated;

  if (S.iterArgs) {
    iterArgUpdated.assign(anchor->getNumOperands(), false);
    for (int i = 0, vals = anchor->getNumOperands(); i < vals; ++i) {
      auto value = anchor->getOperand(i);
      Value newValue = valueMap.lookup(value);
      termIterArgs.push_back(newValue);

      if (auto frameOp = newValue.getDefiningOp<LoopScheduleFrameOp>()) {
        auto bYield = frameOp.getBodyYield();
        unsigned frameResultIdx = cast<OpResult>(newValue).getResultNumber();
        Value insideFrame = bYield->getOperand(frameResultIdx);
        if (auto at = insideFrame.getDefiningOp<LoopScheduleAtOp>()) {
          auto atYield = at.getYieldOp();
          unsigned atResultIdx = cast<OpResult>(insideFrame).getResultNumber();
          Value insideAt = atYield->getOperand(atResultIdx);
          bool alreadyEmitted = false;
          at.getBodyBlock().walk([&](LoopScheduleIterArgUpdateOp u) {
            if (u.getIterArg() == S.iterArgs->scheduleBlockArgs[i]) {
              alreadyEmitted = true;
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          if (!alreadyEmitted) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(atYield);
            builder.create<LoopScheduleIterArgUpdateOp>(
                at.getLoc(), S.iterArgs->scheduleBlockArgs[i], insideAt);
          }
          iterArgUpdated[i] = true;
        }
      }
    }

    auto findAnyAtInSchedule = [&]() -> LoopScheduleAtOp {
      for (auto &op : *S.scheduleBlock) {
        if (auto at = dyn_cast<LoopScheduleAtOp>(op))
          return at;
        if (auto frame = dyn_cast<LoopScheduleFrameOp>(op)) {
          for (auto &inner : frame.getBodyBlock())
            if (auto at = dyn_cast<LoopScheduleAtOp>(inner))
              return at;
        }
      }
      return nullptr;
    };
    for (int i = 0, vals = anchor->getNumOperands(); i < vals; ++i) {
      if (iterArgUpdated[i])
        continue;
      Value iterArg = S.iterArgs->scheduleBlockArgs[i];
      auto hostAt = findAnyAtInSchedule();
      if (!hostAt)
        continue;
      bool alreadyEmitted = false;
      hostAt.getBodyBlock().walk([&](LoopScheduleIterArgUpdateOp u) {
        if (u.getIterArg() == iterArg) {
          alreadyEmitted = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!alreadyEmitted) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(hostAt.getYieldOp());
        builder.create<LoopScheduleIterArgUpdateOp>(hostAt.getLoc(), iterArg,
                                                    iterArg);
      }
    }
  }

  // === Tail emission & cleanup ===
  S.finalize(builder, pendingLaunches, scheduleCondResult, termIterArgs,
             iterArgUpdated, valueMap);

  return success();
}

/// Create the loopschedule ops for an entire function. Builds a strategy
/// targeting the function body and delegates to `lowerSchedule`. The tail
/// emits a trailing barrier frame to await any pending launches and forwards
/// return-consumed values out through that frame.
LogicalResult SCFToLoopSchedulePass::createFuncLoopSchedule(FuncOp &funcOp,
                                                            Problem &problem) {
  auto *funcReturn = funcOp.getBody().back().getTerminator();

  ScheduleStrategy S;
  S.root = funcOp.getOperation();
  S.scheduleBlock = &funcOp.getBody().front();
  S.containingRegion = &funcOp.getBody();
  S.anchor = funcReturn;

  S.shouldSkipForGrouping = [](Operation *op) {
    if (isa<YieldOp, func::ReturnOp, memref::AllocaOp, arith::ConstantOp,
            memref::AllocOp, AllocInterface, IfOp>(op))
      return true;
    if (auto schedOp = dyn_cast<SchedulableInterface>(op))
      if (schedOp.isInitOp())
        return true;
    return false;
  };
  S.isTerminator = [funcOp](Operation *op) {
    return isa<func::ReturnOp>(op) && op->getParentOp() == funcOp;
  };
  S.setFrameInsertion = [funcReturn](OpBuilder &b) {
    b.setInsertionPoint(funcReturn);
  };
  S.seedValueMap = [](IRMapping &) {};
  S.dynAccessesAsLaunches = true;

  S.finalize = [&](ImplicitLocOpBuilder &b,
                   ArrayRef<PendingLaunch> pendingLaunches,
                   Value /*condResult*/, ArrayRef<Value> /*termIterArgs*/,
                   ArrayRef<bool> /*iterArgUpdated*/, IRMapping &valueMap) {
    auto isFuncTerminator = [funcOp](Operation *op) {
      return isa<func::ReturnOp>(op) && op->getParentOp() == funcOp;
    };

    if (!pendingLaunches.empty()) {
      struct ReturnForward {
        size_t pendingIdx;
        unsigned forwardIdx;
        Type resultType;
      };
      SmallVector<ReturnForward> returnForwards;
      SmallVector<Type> barrierResultTypes;
      for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
        for (auto [fi, fwd] : llvm::enumerate(pl.forwards)) {
          for (auto *user : fwd.second.getUsers()) {
            if (isFuncTerminator(user)) {
              returnForwards.push_back(
                  {pi, (unsigned)fi, fwd.second.getType()});
              barrierResultTypes.push_back(fwd.second.getType());
              break;
            }
          }
        }
      }

      b.setInsertionPoint(funcReturn);
      auto barrier = b.create<LoopScheduleFrameOp>(barrierResultTypes);

      {
        Block &awaitBlock = barrier.getAwaitRegion().emplaceBlock();
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToEnd(&awaitBlock);

        SmallVector<Value> awaitYieldOperands;
        for (auto [pi, pl] : llvm::enumerate(pendingLaunches)) {
          SmallVector<Type> awaitRetTypes;
          SmallVector<const ReturnForward *> forwardsForThis;
          for (auto &rf : returnForwards) {
            if (rf.pendingIdx == pi) {
              awaitRetTypes.push_back(rf.resultType);
              forwardsForThis.push_back(&rf);
            }
          }
          auto awaitOp = b.create<LoopScheduleAwaitOp>(
              pl.currentHandle.getLoc(), awaitRetTypes,
              ValueRange{pl.currentHandle});
          for (auto [i, rf] : llvm::enumerate(forwardsForThis)) {
            (void)rf;
            awaitYieldOperands.push_back(awaitOp.getResult(i));
          }
        }
        b.create<LoopScheduleYieldOp>(funcOp.getLoc(), awaitYieldOperands);
      }

      {
        Block &bodyBlock = barrier.getBodyRegion().emplaceBlock();
        for (Type t : barrierResultTypes)
          bodyBlock.addArgument(t, funcOp.getLoc());
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPointToEnd(&bodyBlock);
        b.create<LoopScheduleYieldOp>(
            funcOp.getLoc(),
            SmallVector<Value>(bodyBlock.getArguments().begin(),
                               bodyBlock.getArguments().end()));
      }

      for (auto [i, rf] : llvm::enumerate(returnForwards)) {
        Value origValue =
            pendingLaunches[rf.pendingIdx].forwards[rf.forwardIdx].second;
        origValue.replaceAllUsesWith(barrier.getResult(i));
      }
    }

    auto *returnOp = funcOp.getBody().back().getTerminator();
    int numOperands = returnOp->getNumOperands();
    for (int i = 0; i < numOperands; ++i) {
      auto operand = returnOp->getOperand(i);
      auto newValue = valueMap.lookupOrDefault(operand);
      returnOp->setOperand(i, newValue);
    }

    std::function<bool(Operation *)> inTopLevelFrameOp = [&](Operation *op) {
      auto parent = op->getParentOfType<LoopScheduleFrameOp>();
      if (!parent)
        return false;
      if (isa<func::FuncOp>(parent->getParentOp()))
        return true;
      return inTopLevelFrameOp(parent);
    };

    funcOp.getBody().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if ((isa<LoopScheduleFrameOp>(op) && isa<FuncOp>(op->getParentOp())) ||
          inTopLevelFrameOp(op) ||
          isa<func::ReturnOp, memref::AllocaOp, arith::ConstantOp,
              memref::AllocOp, AllocInterface>(op))
        return;
      if (auto schedOp = dyn_cast<SchedulableInterface>(op))
        if (schedOp.isInitOp())
          return;
      op->dropAllUses();
      op->dropAllDefinedValueUses();
      op->dropAllReferences();
      op->erase();
    });

    // Wrap the lowered body in a `loopschedule.func_sequential` op. The new
    // op carries FunctionOpInterface so downstream passes can keep using
    // function-like APIs. We splice the func body's region into the new op,
    // rewrite `func.return` → `loopschedule.return`, then erase the
    // `func.func` shell.
    {
      OpBuilder builder(funcOp);
      auto newOp = builder.create<LoopScheduleFuncSequentialOp>(
          funcOp.getLoc(), funcOp.getName(), funcOp.getFunctionType());
      // Carry over all attributes except the ones managed by the new op's
      // own operand slots (symbol name + function type).
      for (auto attr : funcOp->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name == SymbolTable::getSymbolAttrName() ||
            name == funcOp.getFunctionTypeAttrName())
          continue;
        // The frontend's discardable `amc.control_interface` marker becomes
        // the op's inherent `control` clause.
        if (name == "amc.control_interface") {
          newOp.setControlInterfaceAttr(cast<StringAttr>(attr.getValue()));
          continue;
        }
        newOp->setAttr(attr.getName(), attr.getValue());
      }
      newOp.getBody().takeBody(funcOp.getBody());

      // Convert the moved func.return to loopschedule.return.
      newOp.getBody().walk([&](func::ReturnOp ret) {
        OpBuilder rb(ret);
        rb.create<LoopScheduleReturnOp>(ret.getLoc(), ret.getOperands());
        ret.erase();
      });

      funcOp.erase();
    }
  };

  return lowerSchedule(S, problem);
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::createSCFToLoopSchedulePass(const SCFToLoopScheduleOptions &options) {
  return std::make_unique<SCFToLoopSchedulePass>(options);
}
