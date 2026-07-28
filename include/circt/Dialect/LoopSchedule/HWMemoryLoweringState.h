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
#include <optional>
#include <string>

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
  /// Memory-driven "this port's outstanding request has completed
  /// this cycle" signal. For a port with a fixed-latency passthrough,
  /// this is `rd_en | wr_en` registered `latency` cycles; for an
  /// arbitrated port it's gated by the arbiter grant so that contention
  /// surfaces as a delayed done. Null when the memory doesn't expose a
  /// done signal (e.g. memref-backed local memories), in which case the
  /// FSM treats it as tied-high (no stall from this port).
  mlir::Value done;
  /// Separate write-completion pulse for read+write DYNAMIC ports (e.g. a
  /// `dyn rw` AXI face): when non-null, `done` carries READ completions
  /// only and `wrDone` carries WRITE completions (the B response). The FSM
  /// then attributes load expects against `done` and store expects against
  /// `wrDone` — each direction completes in issue order within itself,
  /// which cross-direction interleaving (pipeline fill) does not
  /// guarantee. Null for single-direction ports, where `done` carries the
  /// port's only completion stream.
  mlir::Value wrDone;
  /// Memory-driven same-cycle acceptance level for DYNAMIC ports: a request
  /// presented this cycle will be taken iff `ready` is high. Must be
  /// valid-independent (no combinational path from the request inputs).
  /// The FSM stalls posted (fire-and-forget) issue on it so a one-cycle
  /// enable pulse can never be dropped by a busy memory. Null when the
  /// port has no acceptance backpressure (static ports, memref-backed
  /// memories) — treated as tied high.
  mlir::Value ready;
  /// Memory-driven write-drain / read-drain levels for POSTED dynamic ports
  /// (an AXI master face): `wrIdle` is high iff no write is outstanding,
  /// `rdIdle` iff no read is outstanding. A per-access RAW fence gates a
  /// load's `rdEn` on `wrIdle` (wait for prior writes to commit); a WAR fence
  /// gates a store's `wrEn` on `rdIdle`. Null when the memory exposes no drain
  /// (non-posted / non-AXI ports) — treated as tied high (fence is a no-op).
  mlir::Value rdIdle;
  mlir::Value wrIdle;
  unsigned latency = 0;
};

/// Top-level hardware-interface wiring for one external memory (an
/// `!amc.memory_ref` function arg). The kernel `hw.module` exposes the
/// memory's interface (BRAM port wires now, AXI later) at its boundary; the
/// `expand_ref` adapter instance bridges the internal port protocol to it.
///
/// The interface implementer (AMC-side, which has HW available) fills this in
/// `lowerToHW`: it produces the `hw::PortInfo` for the boundary and, for each
/// kernel INPUT port (e.g. BRAM read data), an `inputBackedge` it uses as the
/// adapter instance operand; for each kernel OUTPUT port (addr/en/we/din), the
/// adapter instance result `Value`.
///
/// The conversion pass stays AMC-agnostic: after `lowerToHW`, it appends
/// `inputPorts`/`outputPorts` to the kernel `hw.module`, resolves each
/// `inputBackedges[i]` to the new input block argument, and drops each
/// `outputValues[i]` into the module's `hw.output`.
struct BramBoundary {
  llvm::SmallVector<circt::hw::PortInfo> inputPorts;
  llvm::SmallVector<circt::Backedge> inputBackedges;
  llvm::SmallVector<circt::hw::PortInfo> outputPorts;
  llvm::SmallVector<mlir::Value> outputValues;

  /// Set when the boundary belongs to an external AXI memory: the metadata a
  /// testbench generator needs to size/init a behavioral slave for the
  /// bundle. Accumulated into an `amc.axi_bundles` attribute on the kernel
  /// hw.module by the conversion pass.
  struct AxiMeta {
    std::string bundle;
    uint64_t depth;
    unsigned elemWidth;
    /// Element count of each ARGUMENT carried on this bundle, in the order
    /// their runtime base registers are laid out (which is the order
    /// amc-insert-axi-lite-control assigns s_axilite offsets in). One entry is
    /// a dedicated bundle; several are a SHARED bundle, whose arguments are
    /// placed independently by the host, so a testbench must give each of them
    /// its own base rather than assume one packed allocation.
    llvm::SmallVector<uint64_t, 1> argElems;
  };
  std::optional<AxiMeta> axiMeta;
};

/// One wire of an interface-tier bundle (`!amc.axi` bus or `!amc.burst` face)
/// threaded between a producer and a consumer of that SSA value across two
/// `HWMemoryInstanceLoweringInterface` ops (e.g. an `expand_ref` boundary
/// splice producing a bus that an `amc.instance` of the master consumes).
///
/// `name` matches the `hw.module` port name on both faces (producer and
/// consumer use mirror-image port lists with identical names).
///   - Producer-driven wire (a response the design receives): `value` holds the
///     producer's concrete signal (e.g. a boundary input backedge resolved to a
///     block arg); the consumer reads it as an instance operand. `outputSlot`
///     is -1.
///   - Consumer-driven wire (a request the design drives): `outputSlot` is the
///     index into the producer's `BramBoundary::outputValues` the consumer must
///     fill with its matching instance result (found via `boundaryKey`).
/// Because the conversion pass lowers instances in SSA (def-before-use) order,
/// the producer runs first, so `value` wires are concrete and the output slots
/// exist by the time the consumer runs.
struct InterfaceWire {
  std::string name;
  mlir::Value value;      // set iff producer-driven (consumer reads it)
  int outputSlot = -1;    // >=0 iff consumer-driven (fills a boundary output)
};

/// The set of wires for one interface-tier SSA value, keyed in `interfaceBundles`
/// by that value (an `!amc.axi` or `!amc.burst` result of the producing op).
/// `boundaryKey` is the key (an `!amc.memory_ref`) under which the producer
/// registered its `BramBoundary`, so a consumer can fill the reserved output
/// slots with concrete instance results.
struct InterfaceBundle {
  llvm::SmallVector<InterfaceWire> wires;
  mlir::Value boundaryKey;
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
                        circt::BackedgeBuilder &bb, mlir::SymbolTable &symTab,
                        llvm::DenseMap<mlir::Value, BramBoundary> &bramBoundaries)
      : clk(clk), rst(rst), bb(bb), symTab(symTab),
        bramBoundaries(bramBoundaries) {}

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

  /// Per-external-memory boundary wiring, keyed by the `!amc.memory_ref`
  /// function argument. Filled by the `expand_ref` adapter's `lowerToHW`; the
  /// conversion pass appends the ports to the kernel module and resolves them.
  llvm::DenseMap<mlir::Value, BramBoundary> &bramBoundaries;

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

  /// Interface-tier bundles (`!amc.axi` / `!amc.burst`) threaded between a
  /// producer and consumer instance op, keyed by the producing SSA value.
  /// Registered by the producer's `lowerToHW`, read by the consumer's.
  llvm::DenseMap<mlir::Value, InterfaceBundle> interfaceBundles;

  /// Register an interface bundle for a producer's result value.
  void registerInterface(mlir::Value ifaceValue, InterfaceBundle bundle) {
    interfaceBundles[ifaceValue] = std::move(bundle);
  }

  /// Look up the interface bundle registered for `ifaceValue` (the SSA value a
  /// consumer op takes as an operand). Null if none was registered.
  InterfaceBundle *lookupInterface(mlir::Value ifaceValue) {
    auto it = interfaceBundles.find(ifaceValue);
    return it == interfaceBundles.end() ? nullptr : &it->second;
  }
};

} // namespace loopschedule
} // namespace circt

#endif // CIRCT_DIALECT_LOOPSCHEDULE_HWMEMORYLOWERINGSTATE_H
