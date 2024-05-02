//===- CalyxInterfaces.td - Calyx Interfaces -------------*- tablegen -*---===//
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

#ifndef CALYX_LOWERING_INTERFACES
#define CALYX_LOWERING_INTERFACES

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Op interface for allocations.
def AllocOpLoweringInterface : OpInterface<"AllocLoweringInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for alloc ops to be lowered.
  }];

  let methods = [
    InterfaceMethod<
      "Insert memory into Calyx component.",
      "void",
      "insertMemory",
      (ins "OpBuilder&":$rewriter, "calyx::ComponentLoweringStateInterface&":$state)>
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for loads.
def LoadOpLoweringInterface : OpInterface<"LoadLoweringInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for schedulable loads.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the value of the associated memory",
      "mlir::Value",
      "getMemoryValue",
      (ins)>,
    InterfaceMethod<
      "This returns the value of the associated memory",
      "mlir::Operation::operand_range",
      "getIndices",
      (ins)>,
    InterfaceMethod<
      "This returns the value of the associated memory",
      "mlir::Value",
      "getResult",
      (ins)>,
    InterfaceMethod<
      "This returns the latency of the store.",
      "std::optional<unsigned int>",
      "getLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the value of the associated memory",
      "LogicalResult",
      "connectToMemInterface",
      (ins "mlir::OpBuilder&":$rewriter, "calyx::GroupInterface":$group,
           "const calyx::ComponentOp&":$component, 
           "calyx::ComponentLoweringStateInterface&":$state,
           "std::optional<Block *>&":$blockSched)>
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for stores.
def StoreOpLoweringInterface : OpInterface<"StoreLoweringInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for schedulable stores.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the value of the associated memory",
      "mlir::Value",
      "getMemoryValue",
      (ins)>,
    InterfaceMethod<
      "This returns the value of the associated memory",
      "mlir::Operation::operand_range",
      "getIndices",
      (ins)>,
    InterfaceMethod<
      "This returns the latency of the store.",
      "std::optional<unsigned int>",
      "getLatency",
      (ins)>,
    InterfaceMethod<
      "This returns the value of the associated memory",
      "LogicalResult",
      "connectToMemInterface",
      (ins "mlir::OpBuilder&":$rewriter, "calyx::GroupInterface":$group,
           "const calyx::ComponentOp&":$component, 
           "calyx::ComponentLoweringStateInterface&":$state,
           "std::optional<Block *>&":$blockSched)>
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

// Op interface for allocations.
def MemoryLikeTypeInterface : TypeInterface<"MemoryLikeTypeInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is a type interrface for memory types that can be lowered to Calyx.
  }];

  let methods = [
    InterfaceMethod<
      "Insert memory for this type into the component.",
      "void",
      "insertMemory",
      (ins "OpBuilder&":$rewriter, "calyx::ComponentLoweringStateInterface&":$state)>
  ];
  /* let verify = [{ return verifyGroupInterface(op); }]; */
}

#endif // CALYX_LOWERING_INTERFACES
