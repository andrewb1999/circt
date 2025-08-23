//===- LoopScheduleInterfaces.td - LoopSchedule Interfaces -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"

// Op interface for scheduled loops.
def LoopOpInterface : OpInterface<"LoopInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for LoopSchedule scheduled loops.
  }];

  let methods = [
    InterfaceMethod<
      "Get the arguments passed to the body by the loop.",
      "Block::BlockArgListType",
      "getBodyArgs">,
    InterfaceMethod<
      "Get the initial values for iter args.",
      "Operation::operand_range",
      "getInits">,
    InterfaceMethod<
      "Get a pointer to the body block.",
      "Block*",
      "getBodyBlock">,
    InterfaceMethod<
      "Get a pointer to the condition block.",
      "Block*",
      "getConditionBlock">,
    InterfaceMethod<
      "Get the SSA value corresponding to the loop condition.",
      "Value",
      "getConditionValue">,
    InterfaceMethod<
      "",
      "Operation::operand_range",
      "getTerminatorIterArgs">,
    InterfaceMethod<
      "Get the number of times a loop will run, if known statically.",
      "std::optional<uint64_t>",
      "getBound">,
    InterfaceMethod<
      "Returns if the loop is pipelined and false otherwise",
      "bool",
      "isPipelined">,
    InterfaceMethod<
      "Returns if the loop has inputs that might stall",
      "bool",
      "canStall">
  ];

  let verify = [{ return verifyLoop(op); }];
}

// Op interface for ops that describes a phase of the schedule.
def PhaseOpInterface : OpInterface<"PhaseInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for LoopSchedule ops that describes
    a phase of the schedule.
  }];

  let methods = [
    InterfaceMethod<
      "",
      "std::string",
      "getRegisterNamePrefix">,
    InterfaceMethod<
      "",
      "Block &",
      "getBodyBlock">,
    InterfaceMethod<
      "",
      "bool",
      "isStatic">,
    InterfaceMethod<
      "",
      "std::optional<uint64_t>",
      "getStartTime">
  ];
}

// Op interface for allocations.
def AllocOpInterface : OpInterface<"AllocInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for alloc ops to be lowered.
  }];
}

// Op interface for loads.
def LoadOpInterface : OpInterface<"LoadInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for schedulable loads.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the latency of the load.",
      "unsigned int",
      "getLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the incoming delay of the load.",
      "float",
      "getIncomingDelay",
      (ins)>,
    InterfaceMethod<
      "This returns the outgoing delay of the load.",
      "float",
      "getOutgoingDelay",
      (ins)>,
    InterfaceMethod<
      "This returns true if the load is dynamic.",
      "bool",
      "isDynamic",
      (ins)>,
    InterfaceMethod<
      "This returns the limit of the load.",
      "std::optional<unsigned int>",
      "getLimit",
      (ins)>,
    InterfaceMethod<
      "This returns the value being loaded from.",
      "Value",
      "getMemoryValue",
      (ins)>,
    InterfaceMethod<
      "This returns the indices for the load.",
      "OperandRange",
      "getIndices",
      (ins)>,
    InterfaceMethod<
      "This returns the mutable indices for the load.",
      "MutableOperandRange",
      "getIndicesMutable",
      (ins)>,
    InterfaceMethod<
      "This returns the minimum bitwidth of the ith dimension of the memory.",
      "int64_t",
      "getDimBitwidth",
      (ins "int64_t":$idx)>,
    InterfaceMethod<
      "This returns the result.",
      "Value",
      "getResult",
      (ins)>,
    InterfaceMethod<
      "This returns true if a dependence could exist with otherOp.",
      "bool",
      "hasDependence",
      (ins "Operation *":$otherOp)>,
    InterfaceMethod<
      "This returns the unique identifier of the associated memory",
      "std::string",
      "getUniqueId",
      (ins)>
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for stores.
def StoreOpInterface : OpInterface<"StoreInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for schedulable stores.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the latency of the store.",
      "unsigned int",
      "getLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the incoming delay of the store.",
      "float",
      "getIncomingDelay",
      (ins)>,
    InterfaceMethod<
      "This returns true if the store is dynamic.",
      "bool",
      "isDynamic",
      (ins)>,
    InterfaceMethod<
      "This returns the limit of the store.",
      "std::optional<unsigned int>",
      "getLimit",
      (ins)>,
    InterfaceMethod<
      "This returns the value being stored to.",
      "Value",
      "getMemoryValue",
      (ins)>,
    InterfaceMethod<
      "This returns the indices for the load.",
      "OperandRange",
      "getIndices",
      (ins)>,
    InterfaceMethod<
      "This returns the mutable indices for the load.",
      "MutableOperandRange",
      "getIndicesMutable",
      (ins)>,
    InterfaceMethod<
      "This returns the minimum bitwidth of the ith dimension of the memory.",
      "int64_t",
      "getDimBitwidth",
      (ins "int64_t":$idx)>,
    InterfaceMethod<
      "This returns true if a dependence could exist with otherOp.",
      "bool",
      "hasDependence",
      (ins "Operation *":$otherOp)>,
    InterfaceMethod<
      "This returns true if a control dependence exists with otherOp.",
      "bool",
      "hasControlDependence",
      (ins "Operation *":$otherOp)>,
    InterfaceMethod<
      "This returns the unique identifier of the associated memory",
      "std::string",
      "getUniqueId",
      (ins)>
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for allocations.
def SchedulableOpInterface : OpInterface<"SchedulableInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for alloc ops to be lowered.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the number of cycles it takes for an op to complete.",
      "uint32_t",
      "getOpLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the number of this op that can be scheduled at the same time.",
      "std::optional<uint32_t>",
      "getOpLimit",
      (ins)>,
    InterfaceMethod<
      "This returns the incoming delay of the op.",
      "float",
      "getIncomingDelay",
      (ins)>,
    InterfaceMethod<
      "This returns the outgoing delay of the op.",
      "float",
      "getOutgoingDelay",
      (ins)>,
    InterfaceMethod<
      "This returns a unique identifier for this op",
      "std::string",
      "getUniqueId",
      (ins)>,
    InterfaceMethod<
      "This returns true if this is an init op (like an alloc, constant, etc)",
      "bool",
      "isInitOp",
      (ins)>,
  ];
}

// Op interface for AffineOps.
def SchedulableAffineOpInterface : OpInterface<"SchedulableAffineInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    This is an op interface for schedulable loads.
  }];

  let methods = [
    InterfaceMethod<
      "Creates a new non-affine op equivalent to the original Op",
      "Operation*",
      "createNonAffineOp",
      (ins "mlir::OpBuilder":$builder, "mlir::ValueRange":$indices)>,
    InterfaceMethod<
      "This returns the value being stored to.",
      "Value",
      "getMemoryValue",
      (ins)>,
    InterfaceMethod<
      "",
      "bool",
      "hasIntraIterationDeps",
      (ins)>,
    InterfaceMethod<
      "This returns the result.",
      "bool",
      "hasDependence",
      (ins "Operation *":$otherOp)>,
  ];
}
