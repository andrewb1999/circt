//===- LoopScheduleOps.cpp - LoopSchedule CIRCT Operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the LoopSchedule ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

// True if `block` is terminated by a `loopschedule.yield` with no operands.
// Used by custom printers to elide the trivial terminator (mirrors the
// SingleBlockImplicitTerminator round-trip contract: a trivial yield is
// implicit and does not need to appear in source).
static bool hasTrivialYield(Block &block) {
  auto yield = dyn_cast<LoopScheduleYieldOp>(block.getTerminator());
  return yield && yield.getOperands().empty();
}

//===----------------------------------------------------------------------===//
// LoopInterface
//===----------------------------------------------------------------------===//

LogicalResult loopschedule::verifyLoop(Operation *op) {
  if (!isa<LoopInterface>(op))
    return failure();

  auto loop = cast<LoopInterface>(op);

  // Verify the body block contains at least one phase and a terminator.
  Block *stagesBlock = loop.getBodyBlock();
  if (stagesBlock->getOperations().size() < 2)
    return loop.emitOpError("body must contain at least one phase");

  // Verify the loop's condition value is produced by some phase in the body.
  auto firstPhaseRange = stagesBlock->getOps<PhaseInterface>();
  if (firstPhaseRange.empty())
    return loop.emitOpError("body must contain at least one phase");
  Value condValue = loop.getConditionValue();
  if (!condValue)
    return loop.emitOpError("missing condition value on terminator");
  if (!condValue.getType().isInteger(1))
    return loop.emitOpError("loop condition must be i1, found ")
           << condValue.getType();
  if (!isa<PhaseInterface>(condValue.getDefiningOp()))
    return loop.emitOpError(
        "loop condition must be produced by a phase in the body");

  // Verify exactly one iter_arg_update exists per iter-arg block argument.
  DenseMap<Value, LoopScheduleIterArgUpdateOp> updates;
  WalkResult wr = loop.getBodyBlock()->walk(
      [&](LoopScheduleIterArgUpdateOp u) {
        auto [it, inserted] = updates.try_emplace(u.getIterArg(), u);
        if (!inserted) {
          u.emitOpError("duplicate iter_arg_update for iter-arg");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (wr.wasInterrupted())
    return failure();
  for (BlockArgument ba : loop.getBodyArgs()) {
    if (!updates.count(ba))
      return loop.emitOpError("missing iter_arg_update for iter-arg #")
             << ba.getArgNumber();
  }

  // Verify iter-arg reads only happen in phases at or before the phase that
  // hosts the iter-arg's update. This preserves the invariant that a new
  // iter-arg value is not consumed in the same iteration by phases that
  // scheduling ordered earlier than the update-hosting phase.
  for (auto it : llvm::enumerate(loop.getBodyArgs())) {
    BlockArgument bodyArg = it.value();
    auto upd = updates.lookup(bodyArg);
    // Walk up to the phase that lives directly in the loop's body block, so
    // the monotonic ordering check below operates on operations from the same
    // block as the other phases in `loop.getBodyBlock()`.
    Operation *updParent = upd->getParentOp();
    while (updParent && !(isa<PhaseInterface>(updParent) &&
                          updParent->getBlock() == loop.getBodyBlock()))
      updParent = updParent->getParentOp();
    if (!updParent)
      continue;
    auto updPhase = cast<PhaseInterface>(updParent);
    SmallVector<PhaseInterface> validPhases;
    auto phases = loop.getBodyBlock()->getOps<PhaseInterface>();
    llvm::copy_if(phases, std::back_inserter(validPhases),
                  [&](PhaseInterface phase) {
                    return phase == updPhase ||
                           phase->isBeforeInBlock(updPhase);
                  });
    for (auto &use : bodyArg.getUses()) {
      auto *user = use.getOwner();
      // Skip assignment-target uses: `iter_arg_update %iv = ...` names the
      // iter-arg slot, not a read of its current value.
      if (auto u = dyn_cast<LoopScheduleIterArgUpdateOp>(user))
        if (use.getOperandNumber() == 0)
          continue;
      // The loop terminator reads iter-args as part of naming the iteration
      // boundary (particularly for pass-through iter-args that are returned
      // unchanged). This use is always safe.
      if (isa<LoopScheduleTerminatorOp>(user))
        continue;
      bool inValidPhase = false;
      for (auto phase : validPhases)
        if (phase->isAncestor(user))
          inValidPhase = true;
      if (!inValidPhase)
        return loop.emitOpError("Iter arg can only be used before new value is "
                                "produced, found use in: ");
    }
  }

  // Verify that the terminator produces the expected number of results
  auto termOp =
      cast<LoopScheduleTerminatorOp>(loop.getBodyBlock()->getTerminator());
  if (termOp.getResults().size() != loop->getNumResults())
    return loop.emitOpError(
        "TerminatorOp does not produce the expected number of results");

  return success();
}

//===----------------------------------------------------------------------===//
// LoopSchedulePipelineOp
//===----------------------------------------------------------------------===//

ParseResult LoopSchedulePipelineOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  // Parse initiation interval.
  IntegerAttr ii;
  if (parser.parseKeyword("II") || parser.parseEqual() ||
      parser.parseAttribute(ii))
    return failure();
  result.addAttribute("II", ii);

  // Parse optional trip count.
  if (succeeded(parser.parseOptionalKeyword("trip_count"))) {
    IntegerAttr tripCount;
    if (parser.parseEqual() || parser.parseAttribute(tripCount))
      return failure();
    result.addAttribute("tripCount", tripCount);
  }

  // Parse optional iteration latency.
  if (succeeded(parser.parseOptionalKeyword("latency"))) {
    IntegerAttr latency;
    if (parser.parseEqual() || parser.parseAttribute(latency))
      return failure();
    result.addAttribute("latency", latency);
  }

  // Parse iter_args assignment list.
  SmallVector<OpAsmParser::Argument> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseAssignmentList(regionArgs, operands))
      return failure();
  }

  // Parse function type from iter_args to results.
  FunctionType type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  // Function result type is the pipeline result type.
  result.addTypes(type.getResults());

  // Resolve iter_args operands.
  for (auto [regionArg, operand, type] :
       llvm::zip(regionArgs, operands, type.getInputs())) {
    regionArg.type = type;
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }

  // Parse stages region.
  Region *stages = result.addRegion();
  if (parser.parseRegion(*stages, regionArgs))
    return failure();

  return success();
}

void LoopSchedulePipelineOp::print(OpAsmPrinter &p) {
  // Print the initiation interval.
  p << " II = " << getII();

  // Print the optional tripCount.
  if (getTripCount())
    p << " trip_count = " << *getTripCount();

  // Print the optional iteration latency.
  if (getLatency())
    p << " latency = " << *getLatency();

  // Print iter_args assignment list.
  p << " iter_args(";
  llvm::interleaveComma(
      llvm::zip(getStages().getArguments(), getInits()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ") : ";

  // Print function type from iter_args to results.
  auto type = FunctionType::get(getContext(), getStages().getArgumentTypes(),
                                getResultTypes());
  p.printType(type);

  // Print stages region.
  p << ' ';
  p.printRegion(getStages(), /*printEntryBlockArgs=*/false);
}

LogicalResult LoopSchedulePipelineOp::verify() {
  // Verify the stages block contains at least one stage and a terminator.
  Block &stagesBlock = getStages().front();

  std::optional<uint64_t> lastStartTime;
  for (Operation &inner : stagesBlock) {
    // Verify the stages block contains only `loopschedule.at` or
    // `loopschedule.terminator` ops.
    if (!isa<LoopScheduleAtOp, LoopScheduleTerminatorOp>(inner))
      return emitOpError("stages may only contain 'loopschedule.at' or "
                         "'loopschedule.terminator' ops, found ")
             << inner;

    // Verify the phase start times are monotonically increasing.
    if (auto phase = dyn_cast<PhaseInterface>(inner)) {
      auto startTime = phase.getStartTime();
      if (!startTime.has_value())
        continue;
      if (!lastStartTime.has_value()) {
        lastStartTime = *startTime;
        continue;
      }

      if (*lastStartTime > *startTime)
        return phase->emitOpError("start time must be after previous start (")
               << *lastStartTime << ')';

      lastStartTime = *startTime;
    }
  }

  // When the iteration latency is declared, it must cover every static-port
  // store's commit (issue stage + store latency) — lowerings size the
  // pipeline's `done` from it. Dynamic stores are excluded: their commit is
  // a runtime event no schedule constant can bound.
  if (auto latency = getLatency()) {
    for (auto stage : getStagesBlock().getOps<LoopScheduleAtOp>()) {
      auto offset = stage.getOffset();
      auto result = stage.walk([&](Operation *inner) {
        if (auto store = dyn_cast<StoreInterface>(inner))
          if (!store.isDynamic() && offset + store.getLatency() > *latency) {
            store->emitOpError("store commits at cycle ")
                << (offset + store.getLatency())
                << ", past the pipeline latency (" << *latency << ')';
            return WalkResult::interrupt();
          }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return failure();
    }
  }

  return success();
}

Value LoopSchedulePipelineOp::getConditionValue() {
  return cast<LoopScheduleTerminatorOp>(getStagesBlock().getTerminator())
      .getCondition();
}

void LoopSchedulePipelineOp::build(OpBuilder &builder, OperationState &state,
                                   TypeRange resultTypes, IntegerAttr ii,
                                   std::optional<IntegerAttr> tripCount,
                                   ValueRange iterArgs) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("II", ii);
  if (tripCount)
    state.addAttribute("tripCount", *tripCount);
  state.addOperands(iterArgs);

  SmallVector<Location, 4> argLocs;
  for (auto arg : iterArgs)
    argLocs.push_back(arg.getLoc());

  Region *stagesRegion = state.addRegion();
  Block &stagesBlock = stagesRegion->emplaceBlock();
  stagesBlock.addArguments(iterArgs.getTypes(), argLocs);
  // Note: no default terminator is inserted; the loop's terminator requires
  // a `condition` operand which only the producer can supply.
}

bool LoopSchedulePipelineOp::canStall() {
  auto mightStallRes = this->walk([&](Operation *op) {
    if (auto load = dyn_cast<LoadInterface>(op)) {
      if (load.isDynamic()) {
        return WalkResult::interrupt();
      }
    }

    if (auto store = dyn_cast<StoreInterface>(op)) {
      if (store.isDynamic()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return mightStallRes.wasInterrupted();
}

//===----------------------------------------------------------------------===//
// LoopScheduleSequentialOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleSequentialOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  // Parse optional trip count.
  if (succeeded(parser.parseOptionalKeyword("trip_count"))) {
    IntegerAttr tripCount;
    if (parser.parseEqual() || parser.parseAttribute(tripCount))
      return failure();
    result.addAttribute("tripCount", tripCount);
  }

  // Parse iter_args assignment list.
  SmallVector<OpAsmParser::Argument> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseAssignmentList(regionArgs, operands))
      return failure();
  }

  // Parse function type from iter_args to results.
  FunctionType type;
  if (parser.parseColon() || parser.parseType(type))
    return failure();

  // Function result type is the stg result type.
  result.addTypes(type.getResults());

  // Resolve iter_args operands.
  for (auto [regionArg, operand, type] :
       llvm::zip(regionArgs, operands, type.getInputs())) {
    regionArg.type = type;
    if (parser.resolveOperand(operand, type, result.operands))
      return failure();
  }

  // Parse schedule region.
  Region *stages = result.addRegion();
  if (parser.parseRegion(*stages, regionArgs))
    return failure();

  return success();
}

void LoopScheduleSequentialOp::print(OpAsmPrinter &p) {
  // Print the optional tripCount.
  if (getTripCount())
    p << " trip_count = " << *getTripCount();

  // Print iter_args assignment list.
  p << " iter_args(";
  llvm::interleaveComma(
      llvm::zip(getSchedule().getArguments(), getInits()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ") : ";

  // Print function type from iter_args to results.
  auto type = FunctionType::get(getContext(), getSchedule().getArgumentTypes(),
                                getResultTypes());
  p.printType(type);

  // Print schedule region.
  p << ' ';
  p.printRegion(getSchedule(), /*printEntryBlockArgs=*/false);
}

LogicalResult LoopScheduleSequentialOp::verify() {
  Block &scheduleBlock = getSchedule().front();

  for (Operation &inner : scheduleBlock) {
    // Verify the schedule block contains only `loopschedule.frame` and
    // `loopschedule.terminator` ops.
    if (!isa<LoopScheduleTerminatorOp, LoopScheduleFrameOp>(inner))
      return emitOpError("schedule may only contain 'loopschedule.frame' or "
                         "'loopschedule.terminator' ops, found ")
             << inner;
  }

  return success();
}

void LoopScheduleSequentialOp::build(OpBuilder &builder, OperationState &state,
                                     TypeRange resultTypes,
                                     std::optional<IntegerAttr> tripCount,
                                     ValueRange iterArgs) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  if (tripCount)
    state.addAttribute("tripCount", *tripCount);
  state.addOperands(iterArgs);

  SmallVector<Location, 4> argLocs;
  for (auto arg : iterArgs)
    argLocs.push_back(arg.getLoc());

  Region *scheduleRegion = state.addRegion();
  Block &scheduleBlock = scheduleRegion->emplaceBlock();
  scheduleBlock.addArguments(iterArgs.getTypes(), argLocs);
  // Note: no default terminator is inserted; the loop's terminator requires
  // a `condition` operand which only the producer can supply.
}

Value LoopScheduleSequentialOp::getConditionValue() {
  return cast<LoopScheduleTerminatorOp>(getScheduleBlock().getTerminator())
      .getCondition();
}

bool LoopScheduleSequentialOp::canStall() {
  auto mightStallRes = this->walk([&](Operation *op) {
    if (auto load = dyn_cast<LoadInterface>(op)) {
      if (load.isDynamic()) {
        return WalkResult::interrupt();
      }
    }

    if (auto store = dyn_cast<StoreInterface>(op)) {
      if (store.isDynamic()) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return mightStallRes.wasInterrupted();
}

namespace {
/// Drops iter-args the loop carries through iterations unchanged
/// (terminator yields the iter-arg block argument at the same position as
/// its init). Pass-through iter-args are loop-invariant: external uses of
/// the corresponding loop result can reference the original init, and
/// internal uses of the iter-arg block argument can reference the init
/// directly (the sequential op is not IsolatedFromAbove).
struct ElideSequentialPassThroughIterArgs
    : public OpRewritePattern<LoopScheduleSequentialOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LoopScheduleSequentialOp op,
                                PatternRewriter &rewriter) const override {
    Block *body = op.getBodyBlock();
    auto term = cast<LoopScheduleTerminatorOp>(body->getTerminator());
    ValueRange termResults = term.getResults();
    unsigned n = termResults.size();
    if (n == 0 || n != op.getBodyArgs().size())
      return failure();

    SmallVector<bool> isPassThrough(n, false);
    SmallVector<unsigned> keep;
    keep.reserve(n);
    bool any = false;
    for (unsigned i = 0; i < n; ++i) {
      auto ba = dyn_cast<BlockArgument>(termResults[i]);
      if (ba && ba.getOwner() == body && ba.getArgNumber() == i) {
        isPassThrough[i] = true;
        any = true;
      } else {
        keep.push_back(i);
      }
    }
    if (!any)
      return failure();

    // Erase any self-update iter_arg_update ops on passthrough block args;
    // they become no-ops once the iter-arg is elided.
    SmallVector<LoopScheduleIterArgUpdateOp> selfUpdates;
    body->walk([&](LoopScheduleIterArgUpdateOp u) {
      if (auto ba = dyn_cast<BlockArgument>(u.getIterArg()))
        if (ba.getOwner() == body && isPassThrough[ba.getArgNumber()])
          selfUpdates.push_back(u);
    });
    for (auto u : selfUpdates)
      rewriter.eraseOp(u);

    SmallVector<Type> newResultTypes;
    SmallVector<Value> newInits;
    newResultTypes.reserve(keep.size());
    newInits.reserve(keep.size());
    for (unsigned i : keep) {
      newResultTypes.push_back(op.getResultTypes()[i]);
      newInits.push_back(op.getInits()[i]);
    }

    std::optional<IntegerAttr> tripCountOpt;
    if (auto tc = op.getTripCountAttr())
      tripCountOpt = tc;
    auto newOp = rewriter.create<LoopScheduleSequentialOp>(
        op.getLoc(), newResultTypes, tripCountOpt, newInits);
    Block *newBody = newOp.getBodyBlock();

    // Rebuild the terminator with passthrough positions dropped. The new
    // op's body is empty; we'll merge the old body into it below.
    SmallVector<Value> newTermResults;
    newTermResults.reserve(keep.size());
    for (unsigned i : keep)
      newTermResults.push_back(termResults[i]);
    rewriter.setInsertionPoint(term);
    rewriter.create<LoopScheduleTerminatorOp>(
        term.getLoc(), term.getCondition(), newTermResults, term.getAwait());
    rewriter.eraseOp(term);

    // Build replacement values for the old body's block args:
    //   passthrough -> corresponding init (loop-invariant, dominates newOp)
    //   kept        -> new body's block argument at the new index
    SmallVector<Value> argReplacements(n);
    unsigned newArgIdx = 0;
    for (unsigned i = 0; i < n; ++i) {
      if (isPassThrough[i])
        argReplacements[i] = op.getInits()[i];
      else
        argReplacements[i] = newBody->getArgument(newArgIdx++);
    }

    rewriter.mergeBlocks(body, newBody, argReplacements);

    // Replace the old op's results: kept positions come from newOp's
    // results, passthrough positions come from the original inits.
    SmallVector<Value> opReplacements(n);
    unsigned newResIdx = 0;
    for (unsigned i = 0; i < n; ++i) {
      if (isPassThrough[i])
        opReplacements[i] = op.getInits()[i];
      else
        opReplacements[i] = newOp.getResult(newResIdx++);
    }
    rewriter.replaceOp(op, opReplacements);
    return success();
  }
};
} // namespace

void LoopScheduleSequentialOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ElideSequentialPassThroughIterArgs>(context);
}

//===----------------------------------------------------------------------===//
// LoopScheduleTerminatorOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleTerminatorOp::verify() {
  // Verify the condition operand is defined by a phase op.
  Value cond = getCondition();
  if (!isa_and_nonnull<PhaseInterface>(cond.getDefiningOp()))
    return emitOpError("'condition' must be defined by a phase op");

  // Verify loop terminates with the same result types as the loop.
  auto opResults = getResults();
  TypeRange terminatorResultTypes = opResults.getTypes();
  TypeRange loopResultTypes = this->getResults().getTypes();
  if (terminatorResultTypes != loopResultTypes)
    return emitOpError("'results' types (")
           << terminatorResultTypes << ") must match loop result types ("
           << loopResultTypes << ")";

  // Verify `results` are defined by a phase, OR are iter-arg block arguments
  // of the enclosing loop (the pass-through case — the iter-arg is carried
  // through iterations without modification and will be canonicalized away
  // if the op has other iter-args; if it's the only iter-arg, it's
  // semantically loop-invariant state the loop returns as-is).
  auto loop = cast<LoopInterface>((*this)->getParentOp());
  Block *bodyBlock = loop.getBodyBlock();
  for (auto result : opResults) {
    if (isa_and_nonnull<PhaseInterface>(result.getDefiningOp()))
      continue;
    if (auto ba = dyn_cast<BlockArgument>(result))
      if (ba.getOwner() == bodyBlock)
        continue;
    return emitOpError("'results' must be defined by a phase op or be an "
                       "iter-arg of the enclosing loop");
  }

  // Verify `await` operands trace back to a `loopschedule.launch` by walking
  // through `loopschedule.yield` forwarders out of frame and at body regions.
  for (Value h : getAwait()) {
    Value cur = h;
    while (cur) {
      Operation *def = cur.getDefiningOp();
      if (!def)
        return emitOpError(
            "'await' operand must be produced by a 'loopschedule.launch'");
      if (isa<LoopScheduleLaunchOp>(def))
        break;
      if (auto frame = dyn_cast<LoopScheduleFrameOp>(def)) {
        auto yield = frame.getBodyYield();
        cur = yield.getResults()[cast<OpResult>(cur).getResultNumber()];
        continue;
      }
      if (auto at = dyn_cast<LoopScheduleAtOp>(def)) {
        auto *yield = at.getBodyBlock().getTerminator();
        cur = yield->getOperand(cast<OpResult>(cur).getResultNumber());
        continue;
      }
      return emitOpError(
          "'await' operand must be produced by a 'loopschedule.launch'");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleLoadOp::verify() {
  if (static_cast<int64_t>(getIndices().size()) != getMemRefType().getRank()) {
    return emitOpError("incorrect number of indices for load, expected ")
           << getMemRefType().getRank() << " but got " << getIndices().size();
  }
  return success();
}

/// Per-dim address widths for a memref load: ceil_log2(dim_size), minimum 1.
static SmallVector<unsigned> addrWidthsForMemref(MemRefType memTy) {
  SmallVector<unsigned> widths;
  widths.reserve(memTy.getRank());
  for (int64_t d : memTy.getShape())
    widths.push_back(d <= 1 ? 1u : (unsigned)llvm::Log2_64_Ceil((uint64_t)d));
  return widths;
}

Value LoopScheduleLoadOp::getMemoryValue() { return getMemRef(); }
SmallVector<unsigned> LoopScheduleLoadOp::getAddrWidths() {
  return addrWidthsForMemref(getMemRefType());
}
unsigned LoopScheduleLoadOp::getReadLatency() { return 1; }
bool LoopScheduleLoadOp::requiresReadEnable() { return false; }

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleStoreOp::verify() {
  if (getNumOperands() != 2 + getMemRefType().getRank())
    return emitOpError("store index operand count not equal to memref rank");

  return success();
}

Value LoopScheduleStoreOp::getMemoryValue() { return getMemRef(); }
Value LoopScheduleStoreOp::getValueToStore() { return getOperand(0); }
SmallVector<unsigned> LoopScheduleStoreOp::getAddrWidths() {
  return addrWidthsForMemref(getMemRefType());
}
unsigned LoopScheduleStoreOp::getWriteLatency() { return 1; }

//===----------------------------------------------------------------------===//
// DependenciesOp
//===----------------------------------------------------------------------===//

// LogicalResult LoopScheduleDependenciesOp::verify() {
//   // auto res = getBody().walk([](Operation *op) {
//   //   if (!isa<LoopScheduleDependsOnOp>(op)) {
//   //     return WalkResult::interrupt();
//   //   }
//   //   return WalkResult::advance();
//   // });

//   // if (res.wasInterrupted()) {
//   //   return emitOpError("dependencies op can only contain dependence ops");
//   // }

//   return success();
// }

//===----------------------------------------------------------------------===//
// DependsOnOp
//===----------------------------------------------------------------------===//

// LogicalResult LoopScheduleDependsOnOp::verify() {
//   // auto res = getBody().walk([](Operation *op) {
//   //   if (!isa<LoopScheduleAccessOp>(op)) {
//   //     return WalkResult::interrupt();
//   //   }
//   //   return WalkResult::advance();
//   // });

//   // if (res.wasInterrupted()) {
//   //   return emitOpError("depends_on op can only contain access ops");
//   // }

//   return success();
// }

//===----------------------------------------------------------------------===//
// AccessOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleAccessOp::parse(OpAsmParser &p,
                                        OperationState &result) {
  StringAttr name;
  IntegerAttr dist;
  if (p.parseAttribute(name))
    return failure();

  result.addAttribute("accessName", name);

  if (succeeded(p.parseOptionalKeyword("dist"))) {
    if (p.parseLess() || p.parseAttribute(dist) || p.parseGreater())
      return failure();
    result.addAttribute("dist", dist);
  }

  if (p.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void LoopScheduleAccessOp::print(::mlir::OpAsmPrinter &p) {
  p << " ";
  p.printString(this->getAccessName());
  if (this->getDist() > 0) {
    p << " dist<" << this->getDist() << ">";
  }
  SmallVector<StringRef> elidedAttrs = {
      this->getAccessNameAttrName().getValue(),
      this->getDistAttrName().getValue()};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleIfOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(1);
  Region *bodyRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
    return failure();
  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the 'then' region.
  if (parser.parseRegion(*bodyRegion))
    return failure();
  LoopScheduleIfOp::ensureTerminator(*bodyRegion, parser.getBuilder(),
                                     result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void LoopScheduleIfOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << " " << getCond();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getBodyRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  p.printOptionalAttrDict((*this)->getAttrs());
}

void LoopScheduleIfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             TypeRange resultTypes, Value cond) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(cond);

  // Build then region.
  OpBuilder::InsertionGuard guard(odsBuilder);
  Region *thenRegion = odsState.addRegion();
  odsBuilder.createBlock(thenRegion);
  if (resultTypes.empty())
    LoopScheduleIfOp::ensureTerminator(*thenRegion, odsBuilder,
                                       odsState.location);
}

//===----------------------------------------------------------------------===//
// LoopScheduleYieldOp (shared terminator)
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleYieldOp::verify() {
  Operation *parent = (*this)->getParentOp();
  TypeRange yielded = getResults().getTypes();

  auto typesEqual = [](TypeRange a, TypeRange b) {
    if (a.size() != b.size())
      return false;
    return llvm::equal(a, b);
  };

  if (auto ifOp = dyn_cast<LoopScheduleIfOp>(parent)) {
    if (!typesEqual(yielded, ifOp.getResultTypes()))
      return emitOpError("yielded types must match parent loopschedule.if "
                         "result types");
    return success();
  }

  if (auto atOp = dyn_cast<LoopScheduleAtOp>(parent)) {
    if (!typesEqual(yielded, atOp.getResultTypes()))
      return emitOpError("yielded types must match parent loopschedule.at "
                         "result types");
    return success();
  }

  if (isa<LoopScheduleLaunchOp>(parent)) {
    // Loose at this slice — the launch surface result is a handle; child
    // yielded types are not plumbed onto the launch op yet.
    return success();
  }

  if (auto frameOp = dyn_cast<LoopScheduleFrameOp>(parent)) {
    Region *region = (*this)->getParentRegion();
    if (region == &frameOp.getAwaitRegion()) {
      // Feeds body region's entry-block args.
      Block &bodyBlock = frameOp.getBodyBlock();
      if (!typesEqual(yielded, bodyBlock.getArgumentTypes()))
        return emitOpError("await-region yield types must match frame body "
                           "block arg types");
      return success();
    }
    if (region == &frameOp.getBodyRegion()) {
      if (!typesEqual(yielded, frameOp.getResultTypes()))
        return emitOpError("body-region yield types must match frame op "
                           "result types");
      return success();
    }
    return emitOpError("unrecognized parent region of loopschedule.frame");
  }

  return emitOpError("unsupported parent op for loopschedule.yield");
}

//===----------------------------------------------------------------------===//
// LoopScheduleFrameOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleFrameOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  // `-> (types)` result list (optional).
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen() ||
        parser.parseTypeList(result.types) ||
        parser.parseRParen())
      return failure();
  }

  Region *awaitRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();

  // `await { ... }` form: one region is the await region; the body is
  // synthesized as a passthrough that yields the await's yielded values
  // (equivalently, the frame's result types) unchanged.
  if (succeeded(parser.parseOptionalKeyword("await"))) {
    if (parser.parseRegion(*awaitRegion, /*arguments=*/{}))
      return failure();
    LoopScheduleFrameOp::ensureTerminator(*awaitRegion, parser.getBuilder(),
                                           result.location);
    OpBuilder builder(parser.getBuilder().getContext());
    Block *bodyBlock = builder.createBlock(bodyRegion);
    SmallVector<Location> argLocs(result.types.size(), result.location);
    bodyBlock->addArguments(result.types, argLocs);
    builder.setInsertionPointToStart(bodyBlock);
    builder.create<LoopScheduleYieldOp>(result.location,
                                        bodyBlock->getArguments());
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    return success();
  }

  // Parse the first region. It may be either the await region (followed by
  // `do` and a body region), or — if `do` is absent — the body region itself,
  // in which case we synthesize an empty await region.
  Region firstRegion;
  if (parser.parseRegion(firstRegion, /*arguments=*/{}))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("do"))) {
    awaitRegion->takeBody(firstRegion);

    // Optional `(%a: T, %b: T, ...)` block-arg list inline with `do`.
    SmallVector<OpAsmParser::Argument> bodyArgs;
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseOptionalRParen())) {
        if (parser.parseArgumentList(bodyArgs, OpAsmParser::Delimiter::None,
                                     /*allowType=*/true) ||
            parser.parseRParen())
          return failure();
      }
    }
    if (parser.parseRegion(*bodyRegion, bodyArgs))
      return failure();
  } else {
    bodyRegion->takeBody(firstRegion);
    OpBuilder builder(parser.getBuilder().getContext());
    builder.createBlock(awaitRegion);
    builder.create<LoopScheduleYieldOp>(result.location);
  }
  LoopScheduleFrameOp::ensureTerminator(*awaitRegion, parser.getBuilder(),
                                         result.location);
  LoopScheduleFrameOp::ensureTerminator(*bodyRegion, parser.getBuilder(),
                                         result.location);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void LoopScheduleFrameOp::print(OpAsmPrinter &p) {
  if (!getResultTypes().empty()) {
    p << " -> (";
    llvm::interleaveComma(getResultTypes(), p);
    p << ")";
  }
  Block &awaitBlock = getAwaitBlock();
  bool awaitIsTrivial = awaitBlock.without_terminator().empty() &&
                        getAwaitYield().getOperands().empty();

  // "await { ... }" form: body is a passthrough of the await yield. That is,
  // body has exactly one block whose args are yielded unchanged and whose
  // types match the frame's result types. Only use this form when the await
  // region is non-trivial (otherwise prefer the existing body-only form).
  auto isPassthroughBody = [this]() {
    Region &body = getBodyRegion();
    if (!body.hasOneBlock())
      return false;
    Block &bb = body.front();
    if (!bb.without_terminator().empty())
      return false;
    auto yield = dyn_cast<LoopScheduleYieldOp>(bb.getTerminator());
    if (!yield)
      return false;
    if (yield.getOperands().size() != bb.getNumArguments())
      return false;
    for (auto [arg, val] : llvm::zip(bb.getArguments(), yield.getOperands()))
      if (val != arg)
        return false;
    return true;
  };
  if (!awaitIsTrivial && isPassthroughBody()) {
    p << " await ";
    p.printRegion(getAwaitRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!hasTrivialYield(awaitBlock));
    p.printOptionalAttrDict((*this)->getAttrs());
    return;
  }

  p << ' ';
  if (!awaitIsTrivial) {
    p.printRegion(getAwaitRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!hasTrivialYield(awaitBlock));
    p << " do";
    Block &bodyBlock = getBodyBlock();
    if (bodyBlock.getNumArguments() > 0) {
      p << " (";
      llvm::interleaveComma(bodyBlock.getArguments(), p, [&](BlockArgument arg) {
        p.printRegionArgument(arg);
      });
      p << ")";
    }
    p << ' ';
  }
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!hasTrivialYield(getBodyBlock()));
  p.printOptionalAttrDict((*this)->getAttrs());
}

LoopScheduleYieldOp LoopScheduleFrameOp::getAwaitYield() {
  return cast<LoopScheduleYieldOp>(getAwaitBlock().getTerminator());
}

LoopScheduleYieldOp LoopScheduleFrameOp::getBodyYield() {
  return cast<LoopScheduleYieldOp>(getBodyBlock().getTerminator());
}

LogicalResult LoopScheduleFrameOp::verify() {
  // Await region: only `loopschedule.await` ops (plus the yield terminator).
  // Body region: only `loopschedule.at` / `loopschedule.launch` ops (plus
  // the yield terminator).
  for (Operation &op : getAwaitBlock().without_terminator()) {
    if (!isa<LoopScheduleAwaitOp>(op))
      return emitOpError("await region may contain only loopschedule.await "
                         "ops, found: ")
             << op.getName();
  }
  std::optional<uint64_t> lastOffset;
  for (Operation &op : getBodyBlock().without_terminator()) {
    if (!isa<LoopScheduleAtOp>(op))
      return emitOpError(
                 "body region may contain only loopschedule.at ops, found: ")
             << op.getName();
    uint64_t offset = cast<LoopScheduleAtOp>(op).getOffset();
    if (lastOffset.has_value() && offset < *lastOffset)
      return op.emitOpError("offset must be >= previous at's offset (")
             << *lastOffset << ")";
    lastOffset = offset;
  }

  // Terminator presence is guaranteed by SizedRegion + the yield ops' parent
  // trait, but double-check here so we can give a clean error.
  if (getAwaitBlock().empty() ||
      !isa<LoopScheduleYieldOp>(getAwaitBlock().getTerminator()))
    return emitOpError("await region must be terminated by loopschedule.yield");
  if (getBodyBlock().empty() ||
      !isa<LoopScheduleYieldOp>(getBodyBlock().getTerminator()))
    return emitOpError("body region must be terminated by loopschedule.yield");

  return success();
}

//===----------------------------------------------------------------------===//
// LoopScheduleAtOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleAtOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  IntegerAttr offset;
  if (parser.parseAttribute(offset, parser.getBuilder().getIntegerType(64),
                            "offset", result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalArrow())) {
    // `-> (T, U, ...)` for multi-result; `-> T` for a single result.
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseTypeList(result.types) || parser.parseRParen())
        return failure();
    } else {
      Type t;
      if (parser.parseType(t))
        return failure();
      result.types.push_back(t);
    }
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}))
    return failure();

  LoopScheduleAtOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void LoopScheduleAtOp::print(OpAsmPrinter &p) {
  p << ' ' << getOffset();
  bool printBlockTerminators = false;
  auto rts = getResultTypes();
  if (!rts.empty()) {
    p << " -> ";
    if (rts.size() == 1) {
      p << rts.front();
    } else {
      p << "(";
      llvm::interleaveComma(rts, p);
      p << ")";
    }
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                printBlockTerminators);
  p.printOptionalAttrDict((*this)->getAttrs(), {"offset"});
}

LoopScheduleYieldOp LoopScheduleAtOp::getYieldOp() {
  return cast<LoopScheduleYieldOp>(getBodyBlock().getTerminator());
}

void LoopScheduleAtOp::build(OpBuilder &builder, OperationState &state,
                             TypeRange resultTypes, IntegerAttr offset) {
  OpBuilder::InsertionGuard g(builder);
  state.addTypes(resultTypes);
  state.addAttribute("offset", offset);
  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  LoopScheduleYieldOp::create(builder, builder.getUnknownLoc(), ValueRange());
}

unsigned LoopScheduleAtOp::getStageNumber() {
  unsigned number = 0;
  Block *parentBlock = (*this)->getBlock();
  for (Operation &op : *parentBlock) {
    if (&op == getOperation())
      return number;
    if (isa<LoopScheduleAtOp>(op))
      ++number;
  }
  return number;
}

std::string LoopScheduleAtOp::getRegisterNamePrefix() {
  if (isa<LoopSchedulePipelineOp>((*this)->getParentOp()))
    return "stage_" + std::to_string(getStageNumber());
  return "frame_at_" + std::to_string(getStageNumber());
}

LogicalResult LoopScheduleAtOp::verify() {
  Operation *parent = (*this)->getParentOp();
  if (auto frame = dyn_cast<LoopScheduleFrameOp>(parent)) {
    if ((*this)->getParentRegion() != &frame.getBodyRegion())
      return emitOpError(
          "loopschedule.at must appear in a frame's body region");
    return success();
  }
  if (isa<LoopSchedulePipelineOp, LoopScheduleFuncPipelineOp>(parent))
    return success();
  return emitOpError("loopschedule.at must be inside a loopschedule.frame, "
                     "loopschedule.pipeline, or loopschedule.func_pipeline");
}

//===----------------------------------------------------------------------===//
// LoopScheduleLaunchOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleLaunchOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  // `: !loopschedule.handle`
  Type handleTy;
  if (parser.parseColon() || parser.parseType(handleTy))
    return failure();
  result.types.push_back(handleTy);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}))
    return failure();

  LoopScheduleLaunchOp::ensureTerminator(*body, parser.getBuilder(),
                                         result.location);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void LoopScheduleLaunchOp::print(OpAsmPrinter &p) {
  p << " : ";
  p.printType(getHandle().getType());
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/!hasTrivialYield(getBodyBlock()));
  p.printOptionalAttrDict((*this)->getAttrs());
}

LoopScheduleYieldOp LoopScheduleLaunchOp::getYieldOp() {
  return cast<LoopScheduleYieldOp>(getBodyBlock().getTerminator());
}

LogicalResult LoopScheduleLaunchOp::verify() {
  // Body: exactly one non-terminator op, plus the trailing
  // `loopschedule.yield`. That single payload op can have its own nested
  // regions / inner ops — the constraint is one op at the *top* level of
  // the launch body. This keeps each handle tied to a single underlying
  // done signal, which downstream lowerings (e.g. the FSM backend's
  // stall logic) can read without having to join multiple done wires.
  auto &block = getBodyBlock();
  unsigned nonTerminatorCount = 0;
  for (auto &op : block) {
    if (isa<LoopScheduleYieldOp>(op))
      continue;
    ++nonTerminatorCount;
  }
  if (nonTerminatorCount != 1)
    return emitOpError(
        "body must contain exactly one non-terminator op (found ")
           << nonTerminatorCount << ")";

  // The handle must flow — via at-yield + frame-yield forwarding — to
  // exactly one terminal consumer:
  //   * `loopschedule.await` or a terminator's await list (frame-level
  //     form), OR
  //   * `loopschedule.expect` in a later at-stage of the same pipeline
  //     (pipeline-level form).
  SmallVector<Value, 4> worklist{getHandle()};
  llvm::SmallPtrSet<Value, 4> seen;
  unsigned terminalCount = 0;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (isa<LoopScheduleAwaitOp>(user)) {
        ++terminalCount;
      } else if (isa<LoopScheduleExpectOp>(user)) {
        ++terminalCount;
      } else if (auto term = dyn_cast<LoopScheduleTerminatorOp>(user)) {
        if (llvm::is_contained(term.getAwait(), use.get()))
          ++terminalCount;
        else
          return emitOpError("handle used in non-await operand of terminator");
      } else if (auto yield = dyn_cast<LoopScheduleYieldOp>(user)) {
        Operation *yieldParent = yield->getParentOp();
        if (auto at = dyn_cast<LoopScheduleAtOp>(yieldParent)) {
          worklist.push_back(at.getResult(use.getOperandNumber()));
        } else if (auto frame = dyn_cast<LoopScheduleFrameOp>(yieldParent)) {
          if (yield->getParentRegion() != &frame.getBodyRegion())
            return emitOpError("handle yielded outside a frame body");
          worklist.push_back(frame.getResult(use.getOperandNumber()));
        } else {
          return emitOpError("handle yielded outside an at or frame body");
        }
      } else {
        return emitOpError("handle has illegal use: ") << *user;
      }
    }
  }
  if (terminalCount == 0)
    return emitOpError("handle is never awaited");
  if (terminalCount > 1)
    return emitOpError("handle is awaited more than once");
  return success();
}

//===----------------------------------------------------------------------===//
// LoopScheduleExpectOp
//===----------------------------------------------------------------------===//

LoopScheduleLaunchOp LoopScheduleExpectOp::getLaunchOp() {
  // Trace the handle back through at-yield / frame-yield forwarding until
  // we hit the defining launch, or give up.
  Value v = getHandle();
  while (v) {
    if (auto launch = v.getDefiningOp<LoopScheduleLaunchOp>())
      return launch;
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (blockArg)
      return {};
    Operation *def = v.getDefiningOp();
    if (!def)
      return {};
    // Thread back through at / frame results: find the yield producing
    // the result at v's index.
    if (auto at = dyn_cast<LoopScheduleAtOp>(def)) {
      unsigned resultIdx = cast<OpResult>(v).getResultNumber();
      auto yield = at.getYieldOp();
      if (resultIdx >= yield->getNumOperands())
        return {};
      v = yield->getOperand(resultIdx);
      continue;
    }
    if (auto frame = dyn_cast<LoopScheduleFrameOp>(def)) {
      unsigned resultIdx = cast<OpResult>(v).getResultNumber();
      auto yieldOp = frame.getBodyYield();
      if (!yieldOp || resultIdx >= yieldOp->getNumOperands())
        return {};
      v = yieldOp->getOperand(resultIdx);
      continue;
    }
    return {};
  }
  return {};
}

LogicalResult LoopScheduleExpectOp::verify() {
  // The handle must originate from a `loopschedule.launch` in the same
  // `loopschedule.pipeline`, at an earlier stage.
  auto launch = getLaunchOp();
  if (!launch)
    return emitOpError("handle must originate from a `loopschedule.launch`");

  auto expectPipeline =
      getOperation()->getParentOfType<LoopSchedulePipelineOp>();
  auto launchPipeline =
      launch->getParentOfType<LoopSchedulePipelineOp>();
  if (!expectPipeline || expectPipeline != launchPipeline)
    return emitOpError(
        "expect and its launch must live in the same `loopschedule.pipeline`");

  auto launchAt = launch->getParentOfType<LoopScheduleAtOp>();
  auto expectAt = getOperation()->getParentOfType<LoopScheduleAtOp>();
  if (!launchAt || !expectAt)
    return emitOpError("expect and its launch must each be inside an "
                       "`loopschedule.at`");
  if (launchAt.getOffset() >= expectAt.getOffset())
    return emitOpError(
        "expect's `at` offset (")
           << expectAt.getOffset()
           << ") must be strictly greater than the launch's `at` offset ("
           << launchAt.getOffset() << ")";

  // Result types must match the launch's body-level yield operand types
  // one-for-one.
  auto launchYield = launch.getYieldOp();
  if (launchYield->getNumOperands() != getNumResults())
    return emitOpError("has ")
           << getNumResults() << " result(s), launch yields "
           << launchYield->getNumOperands();
  for (auto [i, expectTy, yieldOpnd] : llvm::enumerate(
           getResultTypes(), launchYield->getOperandTypes())) {
    if (expectTy != yieldOpnd)
      return emitOpError("result #")
             << (unsigned)i << " type " << expectTy
             << " does not match launch yield operand type " << yieldOpnd;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoopScheduleAwaitOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleAwaitOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> handles;
  if (parser.parseOperandList(handles))
    return failure();

  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseTypeList(result.types) || parser.parseRParen())
        return failure();
    } else {
      Type t;
      if (parser.parseType(t))
        return failure();
      result.types.push_back(t);
    }
  }

  if (parser.resolveOperands(
          handles, HandleType::get(parser.getContext()),
          result.operands))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void LoopScheduleAwaitOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getHandles());
  auto rts = getResultTypes();
  if (!rts.empty()) {
    p << " -> ";
    if (rts.size() == 1) {
      p << rts.front();
    } else {
      p << "(";
      llvm::interleaveComma(rts, p);
      p << ")";
    }
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult LoopScheduleIterArgUpdateOp::verify() {
  Operation *op = (*this)->getParentOp();
  while (op && !isa<LoopInterface>(op))
    op = op->getParentOp();
  if (!op)
    return emitOpError("must be nested inside a loopschedule.sequential or "
                       "loopschedule.pipeline op");
  auto loop = cast<LoopInterface>(op);
  auto bbArg = dyn_cast<BlockArgument>(getIterArg());
  if (!bbArg || bbArg.getOwner() != loop.getBodyBlock())
    return emitOpError("'iterArg' must be an iter-arg block argument of the "
                       "enclosing loopschedule.sequential or "
                       "loopschedule.pipeline op");
  return success();
}

LogicalResult LoopScheduleAwaitOp::verify() {
  auto frame = cast<LoopScheduleFrameOp>((*this)->getParentOp());
  if ((*this)->getParentRegion() != &frame.getAwaitRegion())
    return emitOpError(
        "loopschedule.await must appear in a frame's await region");
  return success();
}

#include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.cpp.inc"
#include "circt/Dialect/LoopSchedule/LoopScheduleLoweringInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"

//===----------------------------------------------------------------------===//
// iter_arg_update helpers
//===----------------------------------------------------------------------===//

LoopScheduleIterArgUpdateOp
circt::loopschedule::getIterArgUpdate(LoopInterface loop,
                                      BlockArgument iterArgBlockArg) {
  LoopScheduleIterArgUpdateOp found;
  loop.getBodyBlock()->walk([&](LoopScheduleIterArgUpdateOp u) {
    if (u.getIterArg() == iterArgBlockArg) {
      found = u;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

Value circt::loopschedule::getIterArgNewValue(LoopInterface loop,
                                              BlockArgument iterArgBlockArg) {
  if (auto u = getIterArgUpdate(loop, iterArgBlockArg))
    return u.getValue();
  return {};
}

SmallVector<LoopScheduleIterArgUpdateOp>
circt::loopschedule::getIterArgUpdatesInOrder(LoopInterface loop) {
  auto bodyArgs = loop.getBodyArgs();
  SmallVector<LoopScheduleIterArgUpdateOp> result(bodyArgs.size(), {});
  loop.getBodyBlock()->walk([&](LoopScheduleIterArgUpdateOp u) {
    auto ba = dyn_cast<BlockArgument>(u.getIterArg());
    if (!ba || ba.getOwner() != loop.getBodyBlock())
      return;
    unsigned idx = ba.getArgNumber();
    if (idx < result.size())
      result[idx] = u;
  });
  return result;
}

Value circt::loopschedule::getIterArgPhaseResult(
    LoopScheduleIterArgUpdateOp u) {
  Value inside = u.getValue();
  Operation *parent = u->getParentOp();
  while (parent && !isa<LoopScheduleAtOp>(parent))
    parent = parent->getParentOp();
  if (!parent)
    return inside;
  auto at = cast<LoopScheduleAtOp>(parent);
  Operation *term = at.getYieldOp();
  for (auto it : llvm::enumerate(term->getOperands()))
    if (it.value() == inside)
      return parent->getResult(it.index());
  return inside;
}

//===----------------------------------------------------------------------===//
// LoopScheduleFuncSequentialOp / LoopScheduleFuncPipelineOp
//===----------------------------------------------------------------------===//

namespace {
// Shared parser/printer/builder helpers for the two func-level container
// ops. They differ only in whether an `II` attribute is parsed/printed.
template <typename OpT>
ParseResult parseFuncLikeOp(OpAsmParser &parser, OperationState &result,
                             bool hasII) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag, std::string &) {
        return builder.getFunctionType(argTypes, results);
      };

  if (hasII) {
    IntegerAttr ii;
    if (parser.parseKeyword("ii") || parser.parseEqual() ||
        parser.parseAttribute(ii, parser.getBuilder().getIntegerType(64), "II",
                              result.attributes))
      return failure();
  }

  // Optional block-control protocol clause: `control = <kind>` (a bare
  // keyword, e.g. ap_ctrl_hs or none).
  if (succeeded(parser.parseOptionalKeyword("control"))) {
    StringRef kind;
    if (parser.parseEqual() || parser.parseKeyword(&kind))
      return failure();
    result.addAttribute(OpT::getControlInterfaceAttrName(result.name),
                        parser.getBuilder().getStringAttr(kind));
  }

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      OpT::getFunctionTypeAttrName(result.name), buildFuncType,
      OpT::getArgAttrsAttrName(result.name),
      OpT::getResAttrsAttrName(result.name));
}

template <typename OpT>
void printFuncLikeOp(OpT op, OpAsmPrinter &p, bool hasII) {
  if (hasII) {
    p << " ii = " << op->template getAttrOfType<IntegerAttr>("II").getInt();
  }
  if (auto ctrl = op.getControlInterface())
    p << " control = " << *ctrl;
  // Replicate function_interface_impl::printFunctionOp but with `II` (and
  // the control clause) added to the elided attribute list so they don't
  // get printed twice.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  StringRef sym = op.getSymName();
  p << ' ';
  if (auto visibility = op->template getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(sym);
  function_interface_impl::printFunctionSignature(
      p, op, op.getArgumentTypes(), /*isVariadic=*/false, op.getResultTypes());
  SmallVector<StringRef> elided{
      visibilityAttrName,
      op.getFunctionTypeAttrName().getValue(),
      op.getArgAttrsAttrName().getValue(),
      op.getResAttrsAttrName().getValue(),
      SymbolTable::getSymbolAttrName(),
      op.getControlInterfaceAttrName().getValue(),
  };
  if (hasII)
    elided.push_back("II");
  function_interface_impl::printFunctionAttributes(p, op, elided);
  Region &body = op->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

template <typename OpT>
LogicalResult verifyFuncLikeBody(OpT op) {
  if (auto ctrl = op.getControlInterface())
    if (*ctrl != "ap_ctrl_hs" && *ctrl != "none")
      return op.emitOpError("control interface must be 'ap_ctrl_hs' or "
                            "'none', got '")
             << *ctrl << "'";
  // External (declaration-only) ops have an empty body — nothing to check.
  if (op.isExternal())
    return success();
  Block &body = op.getBody().front();
  if (body.empty() || !isa<LoopScheduleReturnOp>(body.back()))
    return op.emitOpError(
        "body must be terminated with `loopschedule.return`");
  // Argument types must match function signature.
  auto fnType = op.getFunctionType();
  if (body.getNumArguments() != fnType.getNumInputs())
    return op.emitOpError("body block argument count must match function "
                          "signature input count");
  for (auto [i, t] : llvm::enumerate(fnType.getInputs())) {
    if (body.getArgument(i).getType() != t)
      return op.emitOpError("body block argument #")
             << i << " type mismatch with function signature";
  }
  return success();
}
} // namespace

ParseResult LoopScheduleFuncSequentialOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseFuncLikeOp<LoopScheduleFuncSequentialOp>(parser, result,
                                                        /*hasII=*/false);
}
void LoopScheduleFuncSequentialOp::print(OpAsmPrinter &p) {
  printFuncLikeOp(*this, p, /*hasII=*/false);
}
LogicalResult LoopScheduleFuncSequentialOp::verify() {
  return verifyFuncLikeBody(*this);
}
void LoopScheduleFuncSequentialOp::build(OpBuilder &builder,
                                          OperationState &state, StringRef name,
                                          FunctionType type,
                                          ArrayRef<NamedAttribute> attrs,
                                          ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
  if (!argAttrs.empty()) {
    call_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, /*resultAttrs=*/{},
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
  }
}

ParseResult LoopScheduleFuncPipelineOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseFuncLikeOp<LoopScheduleFuncPipelineOp>(parser, result,
                                                      /*hasII=*/true);
}
void LoopScheduleFuncPipelineOp::print(OpAsmPrinter &p) {
  printFuncLikeOp(*this, p, /*hasII=*/true);
}
LogicalResult LoopScheduleFuncPipelineOp::verify() {
  if (getII() < 1)
    return emitOpError("II must be >= 1");
  return verifyFuncLikeBody(*this);
}
void LoopScheduleFuncPipelineOp::build(OpBuilder &builder,
                                        OperationState &state, StringRef name,
                                        FunctionType type, IntegerAttr ii,
                                        ArrayRef<NamedAttribute> attrs,
                                        ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttribute("II", ii);
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();
  if (!argAttrs.empty()) {
    call_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, /*resultAttrs=*/{},
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
  }
}

FunctionType LoopScheduleCallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperands().getTypes(),
                           getResultTypes());
}

LogicalResult
LoopScheduleCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (!isa_and_nonnull<LoopScheduleLaunchOp>((*this)->getParentOp()))
    return emitOpError(
        "must be directly nested in a loopschedule.launch "
        "(dynamic-latency op: caller stalls on callee done)");

  auto calleeAttr = getCalleeAttr();
  Operation *callee =
      symbolTable.lookupNearestSymbolFrom(*this, calleeAttr);
  if (!callee)
    return emitOpError("'") << calleeAttr.getValue()
                            << "' does not reference a valid symbol";

  FunctionType calleeType;
  if (auto seq = dyn_cast<LoopScheduleFuncSequentialOp>(callee))
    calleeType = seq.getFunctionType();
  else if (auto pipe = dyn_cast<LoopScheduleFuncPipelineOp>(callee))
    calleeType = pipe.getFunctionType();
  else
    return emitOpError("'") << calleeAttr.getValue()
                            << "' must reference a loopschedule.func_sequential "
                               "or loopschedule.func_pipeline";

  if (calleeType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee: expected ")
           << calleeType.getNumInputs() << " got " << getNumOperands();
  for (auto [i, t] : llvm::enumerate(calleeType.getInputs())) {
    if (getOperand(i).getType() != t)
      return emitOpError("operand #")
             << i << " type mismatch: expected " << t << " got "
             << getOperand(i).getType();
  }

  if (calleeType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee: expected ")
           << calleeType.getNumResults() << " got " << getNumResults();
  for (auto [i, t] : llvm::enumerate(calleeType.getResults())) {
    if (getResult(i).getType() != t)
      return emitOpError("result #")
             << i << " type mismatch: expected " << t << " got "
             << getResult(i).getType();
  }

  return success();
}

LogicalResult LoopScheduleReturnOp::verify() {
  Operation *parent = (*this)->getParentOp();
  TypeRange parentResults;
  if (auto seq = dyn_cast<LoopScheduleFuncSequentialOp>(parent))
    parentResults = seq.getFunctionType().getResults();
  else if (auto pipe = dyn_cast<LoopScheduleFuncPipelineOp>(parent))
    parentResults = pipe.getFunctionType().getResults();
  else
    return emitOpError("must be inside a loopschedule func-level container");

  if (getOperands().size() != parentResults.size())
    return emitOpError("operand count must match parent result count");
  for (auto [i, t] : llvm::enumerate(parentResults)) {
    if (getOperand(i).getType() != t)
      return emitOpError("operand #")
             << i << " type mismatch with parent result type";
  }
  return success();
}

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.cpp.inc"

void LoopScheduleDialect::initialize() {
  registerAttributes();
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"
      >();
}
