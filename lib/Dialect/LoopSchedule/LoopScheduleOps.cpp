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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

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

  // Verify `results` are defined by a phase.
  for (auto result : opResults)
    if (!isa_and_nonnull<PhaseInterface>(result.getDefiningOp()))
      return emitOpError("'results' must be defined by a phase op");

  // Verify `await` operands trace back to a `loopschedule.launch` by walking
  // through `loopschedule.yield` forwarders out of frame body regions.
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

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleStoreOp::verify() {
  if (getNumOperands() != 2 + getMemRefType().getRank())
    return emitOpError("store index operand count not equal to memref rank");

  return success();
}

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

  // Parse the first region. It may be either the await region (followed by
  // `do` and a body region), or — if `do` is absent — the body region itself,
  // in which case we synthesize an empty await region.
  Region *awaitRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();

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
  p << ' ';
  // Elide the await region if it has a single empty-yield terminator.
  Block &awaitBlock = getAwaitBlock();
  bool awaitIsTrivial = awaitBlock.without_terminator().empty() &&
                        getAwaitYield().getOperands().empty();
  if (!awaitIsTrivial) {
    p.printRegion(getAwaitRegion(), /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
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
                /*printBlockTerminators=*/true);
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
  for (Operation &op : getBodyBlock().without_terminator()) {
    if (!isa<LoopScheduleAtOp, LoopScheduleLaunchOp>(op))
      return emitOpError("body region may contain only loopschedule.at and "
                         "loopschedule.launch ops, found: ")
             << op.getName();
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
  if (isa<LoopSchedulePipelineOp>(parent))
    return success();
  return emitOpError("loopschedule.at must be inside a loopschedule.frame or "
                     "loopschedule.pipeline");
}

//===----------------------------------------------------------------------===//
// LoopScheduleLaunchOp
//===----------------------------------------------------------------------===//

ParseResult LoopScheduleLaunchOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  // `at N`
  if (parser.parseKeyword("at"))
    return failure();
  IntegerAttr offset;
  if (parser.parseAttribute(offset, parser.getBuilder().getIntegerType(64),
                            "offset", result.attributes))
    return failure();

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
  p << " at " << getOffset() << " : ";
  p.printType(getHandle().getType());
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict((*this)->getAttrs(), {"offset"});
}

LoopScheduleYieldOp LoopScheduleLaunchOp::getYieldOp() {
  return cast<LoopScheduleYieldOp>(getBodyBlock().getTerminator());
}

LogicalResult LoopScheduleLaunchOp::verify() {
  auto frame = cast<LoopScheduleFrameOp>((*this)->getParentOp());
  if ((*this)->getParentRegion() != &frame.getBodyRegion())
    return emitOpError(
        "loopschedule.launch must appear in a frame's body region");

  // Walk forward through forwarding uses (body-region yields of an enclosing
  // frame) and count terminal consumers (await ops, or terminator await-list
  // operands). Exactly one terminal consumer is required.
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
      } else if (auto term = dyn_cast<LoopScheduleTerminatorOp>(user)) {
        if (llvm::is_contained(term.getAwait(), use.get()))
          ++terminalCount;
        else
          return emitOpError("handle used in non-await operand of terminator");
      } else if (auto yield = dyn_cast<LoopScheduleYieldOp>(user)) {
        auto parentFrame =
            dyn_cast<LoopScheduleFrameOp>(yield->getParentOp());
        if (!parentFrame ||
            yield->getParentRegion() != &parentFrame.getBodyRegion())
          return emitOpError("handle yielded outside a frame body");
        worklist.push_back(
            parentFrame.getResult(use.getOperandNumber()));
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

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.cpp.inc"

void LoopScheduleDialect::initialize() {
  registerAttributes();
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"
      >();
}
