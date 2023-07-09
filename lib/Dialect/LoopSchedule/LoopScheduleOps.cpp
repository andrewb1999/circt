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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;
using namespace circt;
using namespace circt::loopschedule;

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

  // Parse condition region.
  Region *condition = result.addRegion();
  if (parser.parseRegion(*condition, regionArgs))
    return failure();

  // Parse stages region.
  if (parser.parseKeyword("do"))
    return failure();
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
      llvm::zip(getStages().getArguments(), getIterArgs()), p,
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
  if (stagesBlock.getOperations().size() < 2)
    return emitOpError("stages must contain at least one stage");

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

      if (lastStartTime >= stage.getStart())
        return stage.emitOpError("'start' must be after previous 'start' (")
               << lastStartTime.value() << ')';

      lastStartTime = stage.getStart();
    }
  }

  // If no trip count is set, termintor must have condition
  if (!getTripCount().has_value()) {
    if (!getConditionValue().has_value())
      return emitOpError(
          "pipeline terminator must have conditon if trip count is not set");
  }

  return success();
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
  builder.setInsertionPointToEnd(&stagesBlock);
  builder.create<LoopScheduleTerminatorOp>(builder.getUnknownLoc(),
                                           ValueRange(), ValueRange());
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

//===----------------------------------------------------------------------===//
// PipelineStageOp
//===----------------------------------------------------------------------===//

LogicalResult LoopSchedulePipelineStageOp::verify() { 
  // if (getStart() == 0)
  //   return success();

  // WalkResult res = this->walk([&](Operation *op) {
  //   auto operands = op->getOpOperands();
  //   for (auto &operand : operands) {
  //     Value v = operand.get();
  //     if (auto blockArg = dyn_cast<BlockArgument>(v)) {
  //       auto *definingOp = blockArg.getOwner()->getParentOp();
  //       if (isa<LoopSchedulePipelineOp>(definingOp)) {
  //         return WalkResult::interrupt();
  //       }
  //     }
  //   }
  //   return WalkResult::advance();
  // });
  // if (res.wasInterrupted()) {
  //   emitOpError("Pipeline iter_args can only be accessed in first cycle");
  //   return failure();
  // }
  return success(); 
}

void LoopSchedulePipelineStageOp::build(OpBuilder &builder,
                                        OperationState &state,
                                        TypeRange resultTypes,
                                        IntegerAttr start,
                                        IntegerAttr end) {
  OpBuilder::InsertionGuard g(builder);

  state.addTypes(resultTypes);
  state.addAttribute("start", start);
  state.addAttribute("end", end);

  Region *region = state.addRegion();
  Block &block = region->emplaceBlock();
  builder.setInsertionPointToEnd(&block);
  builder.create<LoopScheduleRegisterOp>(builder.getUnknownLoc(), ValueRange());
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

  // Parse condition region.
  Region *condition = result.addRegion();
  if (parser.parseRegion(*condition, regionArgs))
    return failure();

  // Parse stages region.
  if (parser.parseKeyword("do"))
    return failure();
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
      llvm::zip(getSchedule().getArguments(), getIterArgs()), p,
      [&](auto it) { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ") : ";

  // Print function type from iter_args to results.
  auto type = FunctionType::get(getContext(), getSchedule().getArgumentTypes(),
                                getResultTypes());
  p.printType(type);

  // Print stages region.
  p << ' ';
  p.printRegion(getSchedule(), /*printEntryBlockArgs=*/false);
}

LogicalResult LoopScheduleSequentialOp::verify() {
  // Verify the stages block contains at least one stage and a terminator.
  Block &scheduleBlock = getSchedule().front();
  if (scheduleBlock.getOperations().size() < 2)
    return emitOpError("stages must contain at least one stage");

  for (Operation &inner : scheduleBlock) {
    // Verify the stages block contains only `stg.step` and
    // `stg.terminator` ops.
    if (!isa<LoopScheduleStepOp, LoopScheduleTerminatorOp>(inner))
      return emitOpError("stages may only contain 'stg.step' or "
                         "'stg.terminator' ops, found ")
             << inner;
  }

  // If no trip count is set, terminator must have condition
  if (!getTripCount().has_value()) {
    auto term =
        cast<LoopScheduleTerminatorOp>(getScheduleBlock().getTerminator());
    if (!term.getCondition().has_value())
      return emitOpError(
          "pipeline terminator must have conditon if trip count is not set");
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
  builder.setInsertionPointToEnd(&scheduleBlock);
  builder.create<LoopScheduleTerminatorOp>(builder.getUnknownLoc(), ValueRange(),
                                       ValueRange());
}

//===----------------------------------------------------------------------===//
// STGStepOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleStepOp::verify() {
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
    op->emitOpError("STGStepOp not inside a function or STGWhileOp");
    return -1;
  }

  while (step != op && step->getNextNode()) {
    ++number;
    step = step->getNextNode();
  }
  return number;
}

//===----------------------------------------------------------------------===//
// LoopScheduleRegisterOp
//===----------------------------------------------------------------------===//

LogicalResult LoopScheduleRegisterOp::verify() {
  LoopSchedulePipelineStageOp stage =
      (*this)->getParentOfType<LoopSchedulePipelineStageOp>();

  // If this doesn't terminate a stage, it is terminating the condition.
  if (stage == nullptr)
    return success();

  // Verify stage terminates with the same types as the result types.
  TypeRange registerTypes = getOperandTypes();
  TypeRange resultTypes = stage.getResultTypes();
  if (registerTypes != resultTypes)
    return emitOpError("operand types (")
           << registerTypes << ") must match result types (" << resultTypes
           << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// LoopScheduleTerminatorOp
//===----------------------------------------------------------------------===//

void LoopScheduleTerminatorOp::build(OpBuilder &builder, OperationState &state,
                                     ValueRange iterArgs, ValueRange results) {
  OpBuilder::InsertionGuard g(builder);

  state.addOperands(iterArgs);
  state.addOperands(results);
  state.addAttribute(getOperandSegmentSizesAttrName(state.name), builder.getDenseI32ArrayAttr({(0), static_cast<int32_t>(iterArgs.size()), static_cast<int32_t>(results.size())}));
}

LogicalResult LoopScheduleTerminatorOp::verify() {
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
          "'iter_args' must be defined by a 'loopschedule.pipeline.stage' or 'loopschedule.step'");

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
          "'results' must be defined by a 'loopschedule.pipeline.stage' or 'loopschedule.step'");

  return success();
}

#include "circt/Dialect/LoopSchedule/LoopScheduleInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"

void LoopScheduleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/LoopSchedule/LoopSchedule.cpp.inc"
      >();
}

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.cpp.inc"
