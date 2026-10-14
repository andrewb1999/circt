//===- OpLibInterfaces.td - OpLib interfaces ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OPLIB_OPLIBINTERFACES_TD
#define CIRCT_DIALECT_OPLIB_OPLIBINTERFACES_TD

include "mlir/IR/OpBase.td"

def LibraryEntryOpInterface : OpInterface<"LibraryEntryOpInterface"> {
  let cppNamespace = "::circt::oplib";
  let description = [{
    An entry of an `oplib.library`: something ops can be implemented by,
    with the timing and resource facts the scheduler and binder need.
    An op names the entry it uses with a `loopschedule.operator` reference
    to the entry's symbol.

    `oplib.operator` is the entry for a compute operator: it matches ops by
    pattern (its `oplib.target`) and builds hardware (its `oplib.hw_match`).
    Other dialects add their own kinds of entry: AMC's `amc.port_timing`
    times a memory port, whose accesses it is assigned to explicitly and
    whose hardware the memory builds.
  }];

  let methods = [
    InterfaceMethod<"The symbol ops reference the entry by.",
      "::llvm::StringRef", "getEntryName", (ins), [{}], [{
        return ::mlir::SymbolTable::getSymbolName($_op).getValue();
      }]>,
    InterfaceMethod<"Cycles from an op's start to its result.",
      "unsigned", "getEntryLatency">,
    InterfaceMethod<[{
        Delay (ns) from an op's operands to its first register, or to its
        result if it has none; none if unknown.
      }], "std::optional<double>", "getIncomingDelay">,
    InterfaceMethod<[{
        Delay (ns) from an op's last register to its result; none if
        unknown.
      }], "std::optional<double>", "getOutgoingDelay">,
    InterfaceMethod<[{
        How many of the ops referencing the entry may be in flight at once
        (an operator's physical instances); none for no cap. The scheduler
        pools every op referencing the entry under the limit.
      }], "std::optional<unsigned>", "getEntryLimit">,
    InterfaceMethod<"Whether `op` may use the entry.",
      "bool", "matches", (ins "::mlir::Operation *":$op)>,
    InterfaceMethod<[{
        The name of the ops the entry matches by pattern, which the library
        indexes it under; none for an entry ops are assigned to explicitly.
      }], "std::optional<::llvm::StringRef>", "getMatchedOperationName", (ins),
      [{}], [{ return std::nullopt; }]>,
    InterfaceMethod<[{
        The `oplib.hw_match` describing the hardware to build for an op
        using the entry; null when the entry builds none.
      }], "::mlir::Operation *", "getHwMatch", (ins), [{}],
      [{ return nullptr; }]>,
  ];
}

#endif // CIRCT_DIALECT_OPLIB_OPLIBINTERFACES_TD
