//===- LoopScheduleLoweringInterfaces.td - HW-lowering ifaces -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dialect-neutral op interfaces that let the LoopScheduleToFSM conversion
// dispatch generically on load/store/memory-instance ops without depending
// on any downstream dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LOOPSCHEDULE_LOWERING_INTERFACES
#define LOOPSCHEDULE_LOWERING_INTERFACES

include "mlir/IR/OpBase.td"

// Op interface for ops that read one element through a memory-like handle
// (memref argument, AMC port, etc.). Descriptive only — the conversion
// pass owns the HW-emission logic; the op provides metadata.
def HWLoadLoweringInterface : OpInterface<"HWLoadLoweringInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    Exposes enough metadata for the LoopSchedule→HW/FSM lowering to drive
    a memory port: the memory handle SSA value, the per-dim index
    operands, the result SSA value, per-dim address widths, read latency,
    and whether an explicit read-enable signal is required.
  }];

  let methods = [
    InterfaceMethod<
      "Return the SSA value identifying the memory this load reads from.",
      "mlir::Value", "getMemoryValue", (ins)>,
    InterfaceMethod<
      "Return the per-dim index operands for this load.",
      "mlir::OperandRange", "getIndices", (ins)>,
    InterfaceMethod<
      "Return the result SSA value (where the loaded element lands).",
      "mlir::Value", "getResult", (ins)>,
    InterfaceMethod<
      "Return the per-dim address bit-widths the memory port expects.",
      "::llvm::SmallVector<unsigned>", "getAddrWidths", (ins)>,
    InterfaceMethod<
      "Return the read latency, in cycles.",
      "unsigned", "getReadLatency", (ins)>,
    InterfaceMethod<
      "Return whether this op needs an explicit read-enable signal "
      "driven. Memref-style loads are always-on (false); AMC-style ports "
      "need per-cycle gating (true).",
      "bool", "requiresReadEnable", (ins)>,
  ];
}

// Op interface for ops that write one element through a memory-like
// handle. Descriptive only.
def HWStoreLoweringInterface : OpInterface<"HWStoreLoweringInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    Exposes enough metadata for the LoopSchedule→HW/FSM lowering to drive
    a memory port's write signals.
  }];

  let methods = [
    InterfaceMethod<
      "Return the SSA value identifying the memory this store targets.",
      "mlir::Value", "getMemoryValue", (ins)>,
    InterfaceMethod<
      "Return the per-dim index operands for this store.",
      "mlir::OperandRange", "getIndices", (ins)>,
    InterfaceMethod<
      "Return the SSA value being stored.",
      "mlir::Value", "getValueToStore", (ins)>,
    InterfaceMethod<
      "Return the per-dim address bit-widths the memory port expects.",
      "::llvm::SmallVector<unsigned>", "getAddrWidths", (ins)>,
    InterfaceMethod<
      "Return the write latency, in cycles.",
      "unsigned", "getWriteLatency", (ins)>,
  ];
}

// Op interface for ops producing memory-port-like SSA values at func-op
// scope (e.g. amc.instance). Connect-style: the op owns its HW
// instantiation and port registration. The conversion pass hands it a
// state object it can use to create an hw.instance and register each
// port's signal bundle.
//
// The state type is forward-declared in LoopScheduleOps.h; consumers
// (including interface implementers) include
// "circt/Dialect/LoopSchedule/HWMemoryLoweringState.h" in their .cpp.
def HWMemoryInstanceLoweringInterface
    : OpInterface<"HWMemoryInstanceLoweringInterface"> {
  let cppNamespace = "::circt::loopschedule";

  let description = [{
    Invoked by the LoopSchedule→HW/FSM lowering once per memory instance
    at func-op scope. The implementing op must:

      * Create any hw.instance / seq primitives needed to materialize the
        backing memory module in the caller's hw.module.
      * Register each SSA value it produces (a port handle) with the
        state's port map, filling in the concrete per-port HW signals
        (addresses, read-enable, write-data, write-enable, and the
        memory-driven read-data) as backedges or concrete values.

    The conversion pass then resolves each port's consumer-driven
    backedges after load/store lowering has populated them via the
    HWLoad/StoreLoweringInterface descriptive methods.
  }];

  let methods = [
    InterfaceMethod<
      "Materialize this memory instance in HW and register its port "
      "signals on the lowering state.",
      "::llvm::LogicalResult", "lowerToHW",
      (ins "::mlir::OpBuilder &":$builder,
           "::circt::loopschedule::HWMemoryLoweringState &":$state)>,
  ];
}

#endif // LOOPSCHEDULE_LOWERING_INTERFACES
