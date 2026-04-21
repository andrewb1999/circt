//===- HWMemoryLoweringState.h - State for HW memory lowering ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared state passed between the LoopSchedule→HW/FSM conversion pass and
// the implementers of `HWMemoryInstanceLoweringInterface::lowerToHW`. The
// state captures the hw.module chassis (clk, rst, BackedgeBuilder) and a
// port map that the op populates and the pass consumes.
//
// This header is intentionally kept out of the core LoopSchedule dialect
// include chain (LoopScheduleOps.h forward-declares the class). Only the
// conversion target and interface implementers should include it.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LOOPSCHEDULE_HWMEMORYLOWERINGSTATE_H
#define CIRCT_DIALECT_LOOPSCHEDULE_HWMEMORYLOWERINGSTATE_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace loopschedule {

/// A backedge on a port-signal slot that the conversion pass will resolve
/// once load/store drives have been merged into the unified port map.
///
/// `HWMemoryInstanceLoweringInterface::lowerToHW` implementations create
/// these when they need to emit an hw.instance before the consumer-driven
/// signals are known. `kind`/`addrIdx` identifies which slot of the port's
/// `HWPortSignals` the backedge corresponds to, so the pass can look up
/// the final driver in the merged port map and call `be.setValue(...)`.
struct PortBackedge {
  enum class Kind : uint8_t { Addr, RdEn, WrData, WrEn };
  mlir::Value portValue;
  Kind kind;
  unsigned addrIdx = 0;
  circt::Backedge be;
};

/// HW signal bundle for one memory port at its point of use.
///
/// Directionality is interpreted relative to the consumer (the
/// load/store op):
///   - `addrs`, `rdEn`, `wrData`, `wrEn` are driven by the consumer.
///   - `rdData` is driven by the memory.
/// A null `rdEn` / `wrEn` / `wrData` means "tie to a safe default"
/// (constant 1 for rdEn, constant 0 otherwise). A null `rdData` means
/// the port is write-only.
struct HWPortSignals {
  llvm::SmallVector<mlir::Value> addrs;
  mlir::Value rdEn;
  mlir::Value rdData;
  mlir::Value wrData;
  mlir::Value wrEn;
  unsigned latency = 0;
};

/// State passed to `HWMemoryInstanceLoweringInterface::lowerToHW`.
///
/// Ownership: the conversion pass constructs this before invoking any
/// instance op's `lowerToHW`, invokes each op once, then reads
/// `portMap` when dispatching on `HWLoad/StoreLoweringInterface` for
/// load/store ops inside loop bodies.
class HWMemoryLoweringState {
public:
  HWMemoryLoweringState(mlir::Value clk, mlir::Value rst,
                        circt::BackedgeBuilder &bb,
                        mlir::SymbolTable &symTab)
      : clk(clk), rst(rst), bb(bb), symTab(symTab) {}

  /// Clock / reset values of the hw.module being built. The op should
  /// thread these into any hw.instance or seq.* primitive it creates.
  mlir::Value clk;
  mlir::Value rst;

  /// Backedge builder for input signals (addrs, rdEn, wrData, wrEn) that
  /// the op surfaces on its hw.instance. The pass resolves these from
  /// accumulated load/store drives after all bodies have been lowered.
  circt::BackedgeBuilder &bb;

  /// Module-scope symbol table for resolving @memoryName references.
  mlir::SymbolTable &symTab;

  /// Register the signals for a port SSA value (i.e. one of the op's
  /// results). Must be called for every SSA value the op produces.
  void registerPort(mlir::Value portValue, HWPortSignals signals) {
    portMap[portValue] = std::move(signals);
  }

  /// Look up the signals registered for `portValue`. Returns nullptr if
  /// no entry was registered.
  HWPortSignals *lookupPort(mlir::Value portValue) {
    auto it = portMap.find(portValue);
    return it == portMap.end() ? nullptr : &it->second;
  }

  /// The full port map. The conversion pass consumes this after all
  /// instance ops have run.
  llvm::DenseMap<mlir::Value, HWPortSignals> portMap;

  /// Back-edges the conversion pass must resolve once all load/store
  /// drives have been populated. Populated by interface implementers
  /// during `lowerToHW`.
  llvm::SmallVector<PortBackedge> pendingBackedges;
};

} // namespace loopschedule
} // namespace circt

#endif // CIRCT_DIALECT_LOOPSCHEDULE_HWMEMORYLOWERINGSTATE_H
