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

  // Verify the loop's condition value is produced by the first phase.
  auto firstPhaseRange = stagesBlock->getOps<PhaseInterface>();
  if (firstPhaseRange.empty())
    return loop.emitOpError("body must contain at least one phase");
  Operation *firstPhase = (*firstPhaseRange.begin()).getOperation();
  Value condValue = loop.getConditionValue();
  if (!condValue)
    return loop.emitOpError("missing condition value on terminator");
  if (!condValue.getType().isInteger(1))
    return loop.emitOpError("loop condition must be i1, found ")
           << condValue.getType();
  if (condValue.getDefiningOp() != firstPhase)
    return loop.emitOpError(
        "loop condition must be produced by the first phase of the body");

  // Verify iter_args are produced by the first phase that uses it
  // and is only used before new value is produced
  for (auto it : llvm::enumerate(loop.getTerminatorIterArgs())) {
    auto val = it.value();
    auto i = it.index();
    if (!isa<PhaseInterface>(val.getDefiningOp()))
      return loop.emitOpError("New iter_args must be produced by a phase");
    auto definingPhase = val.getDefiningOp<PhaseInterface>();
    SmallVector<PhaseInterface> validPhases;
    auto phases = loop.getBodyBlock()->getOps<PhaseInterface>();
    llvm::copy_if(phases, std::back_inserter(validPhases),
                  [&](PhaseInterface phase) {
                    return phase == definingPhase ||
                           phase->isBeforeInBlock(definingPhase);
                  });
    if (i >= loop.getBodyArgs().size())
      return loop.emitOpError(
          "mismatched number of iter_args between block and terminator");
    auto bodyArg = loop.getBodyArgs()[i];
    for (auto &use : bodyArg.getUses()) {
      auto *user = use.getOwner();
      bool inValidPhase = false;
      for (auto phase : validPhases) {
        if (phase->isAncestor(user))
          inValidPhase = true;
      }
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
    // Verify the stages block contains only `loopschedule.pipeline.stage` and
    // `loopschedule.terminator` ops.
    if (!isa<LoopSchedulePipelineStageOp, LoopScheduleTerminatorOp>(inner))
      return emitOpError(
                 "stages may only contain 'loopschedule.pipeline.stage' or "
                 "'loopschedule.terminator' ops, found ")
             << inner;

    // Verify the stage start times are monotonically increasing.
    if (auto stage = dyn_cast<LoopSchedulePipelineStageOp>(inner)) {
      if (!lastStartTime.has_value()) {
        lastStartTime = stage.getStart();
        continue;
      }

      if (lastStartTime > stage.getStart())
        return stage.emitOpError("'start' must be after previous 'start' (")
               << lastStartTime.value() << ')';

      lastStartTime = stage.getStart();
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

uint64_t LoopSchedulePipelineOp::getBodyLatency() {
  auto stages = this->getStagesBlock().getOps<LoopSchedulePipelineStageOp>();
  uint64_t bodyLatency = 0;
  for (auto stage : stages) {
    if (stage.getEnd() > bodyLatency)
      bodyLatency = stage.getEnd();
  }
  return bodyLatency;
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
// PipelineStageOp
//===----------------------------------------------------------------------===//

std::optional<LoopSchedulePipelineStageOp>
getStageAfter(LoopSchedulePipelineStageOp stage, uint64_t cycles) {
  auto startTime = stage.getStart();
  auto desiredTime = startTime + cycles;

  auto *op = stage->getNextNode();

  while (op != nullptr) {
    if (auto newStage = dyn_cast<LoopSchedulePipelineStageOp>(op)) {
      if (newStage.getStart() == desiredTime)
        return newStage;
    }
    op = op->getNextNode();
  }

  return std::nullopt;
}

LogicalResult LoopSchedulePipelineStageOp::verify() {
  // auto stage = (*this);
  // auto *term = stage.getBodyBlock().getTerminator();

  // // Verify results produced by pipelined ops are only used when ready
  // for (auto res : stage.getResults()) {
  //   auto num = res.getResultNumber();
  //   auto &termOperand = term->getOpOperand(num);
  //   auto *op = termOperand.get().getDefiningOp();
  //   if (op == nullptr)
  //     continue;
  //   if (!isa<memref::LoadOp, arith::MulIOp>(op))
  //     continue;
  //   uint64_t cycles = 0;
  //   if (isa<memref::LoadOp>(op)) {
  //     cycles = 1;
  //   } else if (isa<arith::MulIOp>(op)) {
  //     cycles = 4;
  //   }
  //   auto correctStep = getStageAfter(stage, cycles);
  //   if (!correctStep.has_value())
  //     continue;
  //   if (res.isUsedOutsideOfBlock(&correctStep->getBodyBlock()))
  //     return emitOpError(
  //         "pipelined ops can only be used the cycle results are ready");
  // }

  return success();
}

void LoopSchedulePipelineStageOp::build(OpBuilder &builder,
                                        OperationState &state,
                                        TypeRange resultTypes,
                                        IntegerAttr start, IntegerAttr end) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("start", start);
  state.addAttribute("end", end);

  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  LoopScheduleRegisterOp::create(builder, builder.getUnknownLoc(),
                                 ValueRange());
}

unsigned LoopSchedulePipelineStageOp::getStageNumber() {
  unsigned number = 0;
  auto *op = getOperation();
  auto parent = op->getParentOfType<LoopSchedulePipelineOp>();
  Operation *stage = &parent.getStagesBlock().front();
  while (stage != op && stage->getNextNode()) {
    ++number;
    stage = stage->getNextNode();
  }
  return number;
}

LoopScheduleRegisterOp LoopSchedulePipelineStageOp::getRegisterOp() {
  return cast<LoopScheduleRegisterOp>(this->getBodyBlock().getTerminator());
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
    // Verify the schedule block contains only `loopschedule.step` and
    // `loopschedule.terminator` ops.
    if (!isa<LoopScheduleStepOp, LoopScheduleTerminatorOp>(inner))
      return emitOpError("stages may only contain 'stg.step' or "
                         "'stg.terminator' ops, found ")
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
// LoopScheduleStepOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleStepOp::verify() {
  // Verify results produced by sequential op are only used in next step or
  // terminator
  auto step = *this;
  auto *term = step.getBodyBlock().getTerminator();
  auto *next = step->getNextNode();
  if (auto nextStep = dyn_cast<LoopScheduleStepOp>(next)) {
    for (auto res : step.getResults()) {
      auto num = res.getResultNumber();
      auto &termOperand = term->getOpOperand(num);
      if (!isa_and_nonnull<memref::LoadOp>(termOperand.get().getDefiningOp()))
        continue;

      for (auto *user : res.getUsers()) {
        auto *ancestor = nextStep.getBodyBlock().findAncestorOpInBlock(*user);
        if (ancestor == nullptr)
          return emitOpError("load results can only be used in next step (must "
                             "be reregistered if used later in schedule)");
      }
    }
  }

  // Verify that result types match register types
  auto regOp = step.getRegisterOp();
  auto regOpTypes = regOp.getOperandTypes();
  auto stepResTypes = step.getResultTypes();

  for (auto p : llvm::zip(regOpTypes, stepResTypes)) {
    if (std::get<0>(p) != std::get<1>(p)) {
      return emitOpError("step op results do not match register op types");
    }
  }

  return success();
}

void LoopScheduleStepOp::build(OpBuilder &builder, OperationState &state,
                               TypeRange resultTypes) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);

  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  builder.create<LoopScheduleRegisterOp>(builder.getUnknownLoc(), ValueRange());
}

unsigned LoopScheduleStepOp::getStepNumber() {
  unsigned number = 0;
  auto *op = getOperation();
  Operation *step;
  if (auto parent = op->getParentOfType<LoopScheduleSequentialOp>(); parent)
    step = &parent.getScheduleBlock().front();
  else if (auto parent = op->getParentOfType<func::FuncOp>(); parent)
    step = &parent.getBody().front().front();
  else {
    op->emitOpError("not inside a function or LoopScheduleSequentialOp");
    return -1;
  }

  while (step != op && step->getNextNode()) {
    ++number;
    step = step->getNextNode();
  }
  return number;
}

LoopScheduleRegisterOp LoopScheduleStepOp::getRegisterOp() {
  return cast<LoopScheduleRegisterOp>(this->getBodyBlock().getTerminator());
}

//===----------------------------------------------------------------------===//
// LoopScheduleRegisterOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleRegisterOp::verify() {
  // Verify the parent phase terminates with the same types as its result types.
  TypeRange registerTypes = getOperandTypes();
  TypeRange resultTypes;
  if (auto stage = (*this)->getParentOfType<LoopSchedulePipelineStageOp>())
    resultTypes = stage.getResultTypes();
  else if (auto step = (*this)->getParentOfType<LoopScheduleStepOp>())
    resultTypes = step.getResultTypes();
  else
    return emitOpError("must be inside a 'loopschedule.pipeline.stage' or "
                       "'loopschedule.step'");

  if (registerTypes != resultTypes)
    return emitOpError("operand types (")
           << registerTypes << ") must match result types (" << resultTypes
           << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// LoopScheduleTerminatorOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleTerminatorOp::verify() {
  // Verify the condition operand is defined by a phase op.
  Value cond = getCondition();
  if (cond.getDefiningOp<LoopSchedulePipelineStageOp>() == nullptr &&
      cond.getDefiningOp<LoopScheduleStepOp>() == nullptr)
    return emitOpError("'condition' must be defined by a "
                       "'loopschedule.pipeline.stage' or 'loopschedule.step'");

  // Verify loop terminates with the same `iter_args` types as the pipeline.
  auto iterArgs = getIterArgs();
  TypeRange terminatorArgTypes = iterArgs.getTypes();
  TypeRange loopArgTypes = this->getIterArgs().getTypes();
  if (terminatorArgTypes != loopArgTypes)
    return emitOpError("'iter_args' types (")
           << terminatorArgTypes << ") must match pipeline 'iter_args' types ("
           << loopArgTypes << ")";

  // Verify `iter_args` are defined by a pipeline stage or step.
  for (auto iterArg : iterArgs)
    if (iterArg.getDefiningOp<LoopSchedulePipelineStageOp>() == nullptr &&
        iterArg.getDefiningOp<LoopScheduleStepOp>() == nullptr)
      return emitOpError(
          "'iter_args' must be defined by a 'loopschedule.pipeline.stage' or "
          "'loopschedule.step'");

  // Verify loop terminates with the same result types as the loop.
  auto opResults = getResults();
  TypeRange terminatorResultTypes = opResults.getTypes();
  TypeRange loopResultTypes = this->getResults().getTypes();
  if (terminatorResultTypes != loopResultTypes)
    return emitOpError("'results' types (")
           << terminatorResultTypes << ") must match loop result types ("
           << loopResultTypes << ")";

  // Verify `results` are defined by a pipeline stage or step.
  for (auto result : opResults)
    if (result.getDefiningOp<LoopSchedulePipelineStageOp>() == nullptr &&
        result.getDefiningOp<LoopScheduleStepOp>() == nullptr)
      return emitOpError(
          "'results' must be defined by a 'loopschedule.pipeline.stage' or "
          "'loopschedule.step'");

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

#include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.cpp.inc"

void LoopScheduleDialect::initialize() {
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"
      >();
}
