//===- SCFToLoopScheduleLowering.h - SCF to LoopSchedule, shared -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the two SCF-to-LoopSchedule passes share: building each region's
// scheduling problem, the simplex's schedule of it, the timing of what the
// FSM lowering puts in front of a region's ops (LoweringDelays), and
// lowering a schedule into LoopSchedule ops. The passes differ only in how
// they schedule a region: `convert-scf-to-loopschedule` with the simplex
// (SCFToLoopScheduleSimplex.cpp), `convert-scf-to-loopschedule-cpsat` with a
// CP-SAT model of each whole function (SCFToLoopScheduleCPSAT.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LIB_CONVERSION_SCFTOLOOPSCHEDULE_LOWERING_H
#define CIRCT_LIB_CONVERSION_SCFTOLOOPSCHEDULE_LOWERING_H

#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/AnalysisManager.h"
#include <optional>

namespace circt {
namespace scftoloopschedule {

/// The options both passes take.
struct LoweringConfig {
  /// The combinational budget per cycle, in ns.
  float cycleTimeNs = 1.0f;
  /// Stamp each op's `loopschedule.chain_slack`.
  bool recordChainSlack = false;
  /// The shared-operand mux's delay for 2, 3, ... users (ns), plain and
  /// with the first user's NOR apart, at 32 and past 32 bits.
  SmallVector<float> shareMuxDelays, shareMuxNorDelays, shareMuxDelays64,
      shareMuxNorDelays64;
};

/// On the copy the CP-SAT scheduler describes its regions on, a loop's
/// region index, carried onto the op it lowers to: its parent's region
/// finds the loops it launches by it.
inline constexpr llvm::StringLiteral kCPSATRegionAttr =
    "loopschedule.cpsat_region";

struct ScheduleStrategy;

/// What the FSM lowering puts in front of a region's ops, beyond the ops
/// before each in its region, which a schedule has to time: the delay model
/// both schedulers take (the simplex's terms in its problem's earliest starts
/// in a cycle, the CP-SAT model's in its constraints).
struct InputDelays {
  /// One LUT level (a hold's bypass, a select decided in the cycle, the
  /// expect FIFO's bypass): the two-user shared-operand mux's delay.
  float gate = 0.0f;
  struct Terms {
    /// Ops of enclosing regions whose results this op reads through their
    /// hold's bypass (`ready ? instance : capture`): a function-level frame's
    /// operator instance or static port, which this op starts a LUT past the
    /// output delay of the operator it takes (none of latency 0). A loop
    /// frame hands what it launches `_latched` registers, and a dynamic
    /// access's result is its capture register: neither has a bypass.
    SmallVector<Operation *, 2> held;
    /// The pipeline's dynamic loads this op reads: in a load's expect stage
    /// through the expect FIFO's bypass (`empty ? response : head`), one
    /// input more on a shared instance's mux.
    SmallVector<Operation *, 1> expectReads;
  };
  DenseMap<Operation *, Terms> ops;
  /// A pipeline's continuation: a stage-0 user of a shared instance selects
  /// with the stage-0 enable, decided in the cycle, so its mux starts a LUT
  /// after the condition is in.
  Operation *condition = nullptr;
};

/// Lowers a module's functions from SCF to LoopSchedule; a subclass
/// schedules each region.
class SCFToLoopScheduleLowering {
public:
  SCFToLoopScheduleLowering(LoweringConfig config, mlir::AnalysisManager am)
      : config(std::move(config)), analysisManager(am) {}
  virtual ~SCFToLoopScheduleLowering() = default;

  /// Lower every function of `moduleOp`.
  LogicalResult run(mlir::ModuleOp moduleOp);

protected:
  /// Before any function is lowered (the CP-SAT pass solves its functions
  /// here).
  virtual LogicalResult prepare(mlir::ModuleOp moduleOp) { return success(); }
  /// As a function's lowering starts: drop what the last one left.
  virtual void beginFunction() {}
  /// Schedule a pipelined loop's (`pipelinedLoop`) or function's problem,
  /// checked, at II `ii` if one is asked for; `owner` is the loop or the
  /// function, `condValue` the loop's continuation.
  /// `delays` is what the lowering puts in front of the region's ops.
  virtual LogicalResult scheduleModulo(scheduling::ChainingModuloProblem &problem,
                                       Operation *anchor, Operation *owner,
                                       mlir::scf::WhileOp pipelinedLoop,
                                       std::optional<int32_t> ii,
                                       Value condValue,
                                       const InputDelays &delays) = 0;
  /// Schedule a sequential loop's or function's problem, checked; `owner`
  /// is the loop or the function.
  virtual LogicalResult
  scheduleSequential(scheduling::ChainingSharedOperatorsProblem &problem,
                     Operation *anchor, Operation *owner,
                     const InputDelays &delays) = 0;

  /// The simplex's schedules, verified, with `delays`: the simplex pass's,
  /// and the CP-SAT pass's start.
  LogicalResult simplexModulo(scheduling::ChainingModuloProblem &problem,
                              Operation *anchor, std::optional<int32_t> ii,
                              const InputDelays &delays);
  LogicalResult
  simplexSequential(scheduling::ChainingSharedOperatorsProblem &problem,
                    Operation *anchor, const InputDelays &delays);

  /// The delay model's terms for the ops of `problem`'s region (a pipeline's
  /// continuation `condValue`).
  InputDelays describeInputDelays(scheduling::Problem &problem, bool pipelined,
                                  Value condValue);

  LoweringConfig config;
  mlir::AnalysisManager analysisManager;
  /// The module being lowered.
  mlir::ModuleOp module;
  std::optional<analysis::LoopScheduleDependenceAnalysis> dependenceAnalysis;
  std::optional<analysis::OperatorLibraryAnalysis> operatorLibraryAnalysis;
  loopschedule::PredicateUse predicateUse;
  loopschedule::PredicateMap predicateMap;
  /// Collect the operators allocation offers each op besides its own
  /// (`loopschedule.candidates`) into `candidates`, the op's own first.
  bool collectCandidates = false;
  DenseMap<Operation *, SmallVector<scheduling::Problem::OperatorType, 4>>
      candidates;

private:
  /// The terms the simplex takes as given: a held result's bypass.
  void applyHeldDelays(scheduling::ChainingProblem &problem,
                       const InputDelays &delays);
  /// Raise `op`'s earliest start in a cycle to `at` (capped at the cycle less
  /// its operator's incoming delay); whether it rose.
  bool raiseEarliestStart(scheduling::ChainingProblem &problem, Operation *op,
                          float at);
  /// Whether `op` takes a shared instance: an operator with a limit
  /// (allocation's shared variants), not a memory port.
  bool takesSharedInstance(Operation *op);

  LogicalResult runOnFunc(mlir::func::FuncOp funcOp);
  LogicalResult
  populateOperatorTypes(Operation *op, Region &body,
                        scheduling::ChainingSharedOperatorsProblem &problem,
                        bool sequentialRegion);
  LogicalResult
  solveChainingModuloProblem(mlir::scf::WhileOp &loop,
                             scheduling::ChainingModuloProblem &problem,
                             float cycleTime, Value condValue);
  LogicalResult solveChainingSharedOperatorsProblem(
      Region &region, scheduling::ChainingSharedOperatorsProblem &problem,
      float cycleTime);
  LogicalResult createLoopSchedulePipeline(mlir::scf::WhileOp &loop,
                                           scheduling::CyclicProblem &problem,
                                           Value condValue);
  LogicalResult createLoopScheduleSequential(mlir::scf::WhileOp &loop,
                                             scheduling::Problem &problem,
                                             Value condValue);
  LogicalResult createFuncLoopSchedule(mlir::func::FuncOp &funcOp,
                                       scheduling::Problem &problem);

  /// Pipelined function-level lowering: produces a
  /// `loopschedule.func_pipeline` op directly. Day-1 restriction: no nested
  /// loops in the function body.
  LogicalResult
  createFuncLoopSchedulePipeline(mlir::func::FuncOp funcOp,
                                 scheduling::ChainingModuloProblem &problem);

  /// Merged driver shared by `createLoopScheduleSequential` and
  /// `createFuncLoopSchedule`. The variant differences are encapsulated in
  /// `strategy`.
  LogicalResult lowerSchedule(ScheduleStrategy &strategy,
                              scheduling::Problem &problem);

  /// Walk the per-stage `startGroups` and compute the per-value pipe
  /// lifetimes (`pipeTimes`) plus the set of values that must be
  /// registered out of each stage so consumers in later stages can read
  /// them (`registerValues`). Shared between the loop-pipeline and
  /// func-pipeline drivers; the optional `funcReturnSentinel` lets the
  /// func variant treat `loopschedule.return` as an end-of-pipeline
  /// consumer without needing a `problem.getStartTime` entry for it.
  /// `startTimes` should be the sorted set of stage start times to walk.
  void computePipelineStageRegisters(
      ArrayRef<unsigned> startTimes,
      DenseMap<unsigned, SmallVector<Operation *>> &startGroups,
      scheduling::CyclicProblem &problem, Operation *funcReturnSentinel,
      SmallVectorImpl<SmallVector<Value>> &registerValues,
      DenseMap<Value, std::pair<unsigned, unsigned>> &pipeTimes);

  /// Emit one `loopschedule.at offset(startTime)` stage at the builder's
  /// current insertion point. Clones each scheduled op into the stage,
  /// honouring `predicateMap` for if-guarded ops; updates the per-stage
  /// `valueMap` so later stages read the new values; populates the
  /// stage's terminator with the `registerValuesAtStage` operands and
  /// re-maps each registered value into its consumer-stage's value map.
  /// Returns the created stage so loop-pipeline callers can record its
  /// stage results (e.g. for the condition-result lookup).
  ///
  /// Shared between `createLoopSchedulePipeline` and
  /// `createFuncLoopSchedulePipeline`; the per-driver insertion-point
  /// difference is handled by the caller setting `builder` before the
  /// call.
  loopschedule::LoopScheduleAtOp emitOnePipelineStage(
      mlir::ImplicitLocOpBuilder &builder, unsigned startTime,
      ArrayRef<Operation *> group, ArrayRef<Value> registerValuesAtStage,
      ArrayRef<Type> registerTypesAtStage,
      MutableArrayRef<mlir::IRMapping> stageValueMaps,
      scheduling::CyclicProblem &problem, mlir::DominanceInfo &dom);
};

/// Does `op` represent a dynamic-latency memory access?
bool isDynamicLatencyOp(Operation &op);
/// An op whose effect is visible outside the loop (a store, a stream push).
bool isObservableEffect(Operation &op);
/// A stream pop a pipelined `loop` may drain past its last iteration.
bool isDrainableStreamPop(Operation *op, mlir::scf::WhileOp loop);

/// The chain slack (ns) of each operator-backed operation of a solved
/// problem: the cycle time less the latest end of a combinational chain
/// through it, as the schedule places it. A combinational operator's chain
/// continues through the consumers that start in its cycle (an edge to a
/// later iteration, `distance` > 0, goes through a register); a pipelined
/// operator's ends at its first register, its incoming delay into the cycle.
/// Whatever is put on the operation's inputs (a shared operator's operand
/// mux) delays that whole chain, so it fits within the slack.
llvm::DenseMap<Operation *, float>
chainSlacks(scheduling::ChainingProblem &problem, float cycleTime,
            llvm::function_ref<unsigned(scheduling::Problem::Dependence)>
                distance);

} // namespace scftoloopschedule
} // namespace circt

#endif // CIRCT_LIB_CONVERSION_SCFTOLOOPSCHEDULE_LOWERING_H
