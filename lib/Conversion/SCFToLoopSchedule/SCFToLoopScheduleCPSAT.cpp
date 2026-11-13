//===- SCFToLoopScheduleCPSAT.cpp - SCF to LoopSchedule in CP-SAT --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// convert-scf-to-loopschedule-cpsat: each function's regions scheduled in
// one CP-SAT model (CPSATChaining). The pass describes the regions on a copy
// of the module first, each lowered on the simplex's schedule (the shared
// lowering's), which is where each function's search starts; then solves
// each function; then lowers the module on the solutions.
//
//===----------------------------------------------------------------------===//

#include "SCFToLoopScheduleLowering.h"
#include "circt/Scheduling/Utilities.h"
#include "circt/Conversion/SCFToLoopSchedule.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/LoopScheduleDependenceAnalysis.h"
#include "circt/Analysis/NameAnalysis.h"
#include "circt/Analysis/OperatorLibraryAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Dialect/LoopSchedule/Utils.h"
#include "circt/Dialect/SSP/SSPInterfaces.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/CPSATChaining.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <math.h>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <utility>

#define DEBUG_TYPE "scf-to-loopschedule"

namespace circt {
#define GEN_PASS_DEF_SCFTOLOOPSCHEDULECPSAT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::scf;
using namespace mlir::func;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::loopschedule;
using namespace circt::scftoloopschedule;

namespace {

/// What the CP-SAT scheduler knows of a region's ops beyond the problem.
struct CPSATRegionInput {
  /// The operator types each operation may take, its linked one first; one
  /// not in the map keeps its linked operator type.
  DenseMap<Operation *, SmallVector<Problem::OperatorType, 4>> candidates;
  DenseMap<Problem::OperatorType, CPSATOperatorInfo> operatorInfo;
  /// CPSATFunctionModel::Op's `alone`, `aloneMultiCycle`, `carriedReaders`.
  DenseSet<Operation *> alone;
  DenseSet<Operation *> aloneMultiCycle;
  DenseMap<Operation *, SmallVector<Operation *, 2>> carriedReaders;
};

/// Schedules each region in its function's CP-SAT model: on the copy
/// (`snapshotSink` set), describes the regions and finds the start of each
/// function's search; for real, solves the functions (prepare) and takes
/// each region's solution.
class CPSATLowering : public SCFToLoopScheduleLowering {
public:
  CPSATLowering(LoweringConfig config, AnalysisManager am,
                MLIRContext *context,
                const SCFToLoopScheduleCPSATOptions &options,
                llvm::StringMap<CPSATFunctionModel> *snapshotSink)
      : SCFToLoopScheduleLowering(std::move(config), am), context(context),
        options(options), snapshotSink(snapshotSink),
        cpsatTimeLimit(options.cpsatTimeLimit),
        cpsatRegionTimeLimit(options.cpsatRegionTimeLimit),
        cpsatRegionObjective(options.cpsatRegionObjective),
        cpsatCycleSlack(options.cpsatCycleSlack),
        cpsatDefaultTripCount(options.cpsatDefaultTripCount),
        cpsatWorkers(options.cpsatWorkers),
        cpsatParameters(options.cpsatParameters),
        cpsatArea(options.cpsatArea), cpsatDeviceLuts(options.cpsatDeviceLuts),
        cpsatDeviceDsps(options.cpsatDeviceDsps),
        cpsatDeviceFFs(options.cpsatDeviceFFs),
        cpsatMuxInputLuts(options.cpsatMuxInputLuts) {
    collectCandidates = true;
  }

protected:
  LogicalResult prepare(ModuleOp moduleOp) override;
  void beginFunction() override {
    cpsatOpAt.clear();
    cpsatHeldPending.clear();
    regionIndex = 0;
  }
  LogicalResult scheduleModulo(ChainingModuloProblem &problem,
                               Operation *anchor, Operation *owner,
                               scf::WhileOp pipelinedLoop,
                               std::optional<int32_t> ii, Value condValue,
                               const InputDelays &delays) override;
  LogicalResult scheduleSequential(ChainingSharedOperatorsProblem &problem,
                                   Operation *anchor, Operation *owner,
                                   const InputDelays &delays) override;

private:
  /// What the CP-SAT scheduler knows of a region: its candidates,
  /// operators' areas and sharing, carried results.
  template <typename ProblemT>
  CPSATRegionInput buildCPSATInput(ProblemT &problem, Operation *anchor,
                                   scf::WhileOp pipelinedLoop);
  /// The CP-SAT scheduler's step for a region, in place of the simplex: on
  /// the copy (`cpsat-snapshot`), describe the region and take the
  /// simplex's schedule for it (\p seeded: \p problem holds one) as the
  /// function's solve's start, or, where the model does not take it,
  /// schedule the region alone in CP-SAT; for real (`cpsat`), take the
  /// function's solution for it. `owner` is the region's loop, or its
  /// function.
  template <typename ProblemT>
  LogicalResult scheduleRegionCPSAT(ProblemT &problem, Operation *anchor,
                                    Operation *owner,
                                    scf::WhileOp pipelinedLoop,
                                    float cycleTime, bool seeded,
                                    const InputDelays &delays,
                                    Value condValue = Value());
  /// On the copy, the simplex's schedule of a region (\p schedule, which
  /// solves and verifies \p problem), quietly: false where it has none.
  bool seedSimplex(llvm::function_ref<LogicalResult()> schedule) {
    if (!snapshotSink)
      return false;
    ScopedDiagnosticHandler quiet(context,
                                  [](Diagnostic &) { return success(); });
    return succeeded(schedule());
  }
  /// Describe the region for the whole-function model (a pipelined loop's
  /// continuation by \p condValue), with the delay model's \p delays.
  template <typename ProblemT>
  CPSATFunctionModel::Region &
  captureRegion(ProblemT &problem, Operation *anchor, Operation *owner,
                scf::WhileOp pipelinedLoop, Value condValue,
                const InputDelays &delays, CPSATFunctionModel &model);
  /// Set the region's schedule, operators and (\p bind) instances from
  /// \p sol, which solved \p region of \p model.
  template <typename ProblemT>
  LogicalResult applyRegionSolution(ProblemT &problem, Operation *anchor,
                                    const CPSATFunctionModel &model,
                                    const CPSATFunctionModel::Region &region,
                                    const CPSATFunctionSolution::Region &sol,
                                    bool bind);
  /// The solver's settings, from the pass's options.
  CPSATChainingConfig cpsatConfig(float cycleTime);
  /// Describe every region on a copy of the module (each scheduled alone in
  /// CP-SAT and lowered as usual) and solve each function's at once.
  LogicalResult solveFunctionsCPSAT(ModuleOp moduleOp);

  MLIRContext *context;
  /// The pass's options, for its run on the copy.
  SCFToLoopScheduleCPSATOptions options;
  /// Where the run on the copy describes its functions' regions.
  llvm::StringMap<CPSATFunctionModel> *snapshotSink;
  double cpsatTimeLimit, cpsatRegionTimeLimit;
  std::string cpsatRegionObjective;
  double cpsatCycleSlack;
  int cpsatDefaultTripCount, cpsatWorkers;
  std::string cpsatParameters, cpsatArea;
  double cpsatDeviceLuts, cpsatDeviceDsps, cpsatDeviceFFs, cpsatMuxInputLuts;
  /// The whole-function model's descriptions and solutions, by function,
  /// and the index of the next region scheduled in this function.
  llvm::StringMap<CPSATFunctionModel> functionModels;
  /// The ids of the values the ops of each function's model read: a
  /// constant's by its value and a function argument's common to the
  /// function, any other value's to its region.
  struct CPSATValueIds {
    DenseMap<Attribute, unsigned> constants;
    DenseMap<Value, unsigned> arguments;
    unsigned next = 0;
  };
  DenseMap<const CPSATFunctionModel *, CPSATValueIds> cpsatValueIds;
  /// Where each described op is in its function's model (region, op), and
  /// the ops reading a result of an enclosing region's op through its
  /// hold's bypass, until that region is described (consumer region and
  /// op, producer).
  DenseMap<Operation *, std::pair<unsigned, unsigned>> cpsatOpAt;
  SmallVector<std::tuple<unsigned, unsigned, Operation *>> cpsatHeldPending;
  llvm::StringMap<CPSATFunctionSolution> functionSolutions;
  unsigned regionIndex = 0;
  /// The mux tables, as the solver's config refers to them.
  SmallVector<float> muxTables[4];
};

struct SCFToLoopScheduleCPSATPass
    : public circt::impl::SCFToLoopScheduleCPSATBase<
          SCFToLoopScheduleCPSATPass> {
  using SCFToLoopScheduleCPSATBase::SCFToLoopScheduleCPSATBase;
  SCFToLoopScheduleCPSATPass(const SCFToLoopScheduleCPSATOptions &options,
                             llvm::StringMap<CPSATFunctionModel> *sink)
      : SCFToLoopScheduleCPSATBase(options), snapshotSink(sink) {}
  void runOnOperation() override;
  /// The pass's options, as given.
  SCFToLoopScheduleCPSATOptions getOptions();

  /// The run on a copy: where it describes its functions' regions.
  llvm::StringMap<CPSATFunctionModel> *snapshotSink = nullptr;
};

} // namespace

/// The width of a value's type in bits; 0 for one without one.
static unsigned bitWidthOf(Type type) {
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  return 0;
}

#ifdef SCHEDULING_OR_TOOLS
template <typename ProblemT>
CPSATRegionInput CPSATLowering::buildCPSATInput(
    ProblemT &problem, Operation *anchor, scf::WhileOp pipelinedLoop) {
  CPSATRegionInput input;
  // The area of an operator, by its library entry's measurement: its LUTs
  // and DSPs as shares of the device, in LUTs.
  auto entryArea = [&](StringRef name) -> std::optional<double> {
    if (!operatorLibraryAnalysis->hasOperator(name))
      return std::nullopt;
    oplib::HwMatchOp hwMatch = operatorLibraryAnalysis->getHwMatchOp(name);
    if (!hwMatch)
      return std::nullopt;
    Operation *entry = hwMatch->getParentOp();
    double lut = 0, dsp = 0;
    if (auto a = entry->getAttrOfType<IntegerAttr>("amc.area_lut"))
      lut = a.getInt();
    if (auto a = entry->getAttrOfType<IntegerAttr>("amc.area_dsp"))
      dsp = a.getInt();
    return lut + dsp * cpsatDeviceLuts / std::max(1.0, double(cpsatDeviceDsps));
  };
  // An operator whose delays exceed the cycle (clamped to it in the
  // problem): one allocation lowered for a recurrence at an II it took as
  // given, past the clock, which this scheduler keeps.
  auto overClock = [&](StringRef name) {
    return operatorLibraryAnalysis->getOperatorIncomingDelay(name).value_or(
               0.0f) > this->config.cycleTimeNs ||
           operatorLibraryAnalysis->getOperatorOutgoingDelay(name).value_or(
               0.0f) > this->config.cycleTimeNs;
  };
  // Each op's candidates, but those costing over half again its cheapest
  // that fits the clock (a Full_Usage core where a DSP58 primitive does):
  // they are taken only where nothing cheaper closes the schedule, and each
  // multiplies the instances the search weighs; and, where one fits, those
  // past the clock. Allocation's own choice stays (the model takes it only
  // if it fits, or none does).
  for (auto *op : problem.getOperations()) {
    auto it = candidates.find(op);
    if (it == candidates.end())
      continue;
    bool anyFits = llvm::any_of(it->second, [&](Problem::OperatorType opr) {
      return !overClock(opr.getValue());
    });
    std::optional<double> cheapest;
    for (auto opr : it->second)
      if (!anyFits || !overClock(opr.getValue()))
        if (auto a = entryArea(opr.getValue()))
          cheapest = cheapest ? std::min(*cheapest, *a) : *a;
    SmallVector<Problem::OperatorType, 4> kept;
    for (auto [i, opr] : llvm::enumerate(it->second)) {
      auto a = entryArea(opr.getValue());
      if (i > 0 && anyFits && overClock(opr.getValue()))
        continue;
      if (i == 0 || !a || !cheapest || *a <= 1.5 * *cheapest)
        kept.push_back(opr);
    }
    if (kept.size() > 1)
      input.candidates[op] = std::move(kept);
  }

  // What each operator costs, and whether its instances may be shared: an
  // extern instance that sees no issue cycles (no `valid`) and declares no
  // limit (a resource-constrained one is the simplex flow's), as binding
  // decides.
  for (auto opr : problem.getOperatorTypes()) {
    StringRef name = opr.getValue();
    if (!operatorLibraryAnalysis->hasOperator(name))
      continue;
    oplib::HwMatchOp hwMatch = operatorLibraryAnalysis->getHwMatchOp(name);
    if (!hwMatch)
      continue;
    Operation *entry = hwMatch->getParentOp();
    auto number = [&](StringRef attr) -> double {
      if (auto i = entry->getAttrOfType<IntegerAttr>(attr))
        return double(i.getInt());
      if (auto f = entry->getAttrOfType<FloatAttr>(attr))
        return f.getValueAsDouble();
      return 0.0;
    };
    CPSATOperatorInfo info;
    info.lut = number("amc.area_lut");
    info.dsp = number("amc.area_dsp");
    info.ff = number("amc.area_ff");
    bool instance = llvm::any_of(*hwMatch.getBodyBlock(), [](Operation &op) {
      return isa<oplib::HwInstanceOp>(op);
    });
    bool issueCycles =
        llvm::any_of(hwMatch.getArgRoles(), [](Attribute role) {
          return cast<StringAttr>(role).getValue() == "valid";
        });
    info.shareable = instance && !issueCycles &&
                     !operatorLibraryAnalysis->getOperatorLimit(name);
    info.overClock = overClock(name);
    info.width = info.operands = 0; // from its ops, below
    input.operatorInfo[opr] = info;
  }
  // An instance's operand mux: its operands' count and widest.
  for (auto *op : problem.getOperations()) {
    SmallVector<Problem::OperatorType, 4> oprs;
    if (auto it = candidates.find(op); it != candidates.end())
      oprs = it->second;
    else if (auto linked = problem.getLinkedOperatorType(op))
      oprs.push_back(*linked);
    unsigned widest = 0;
    for (Value operand : op->getOperands())
      widest = std::max(widest, bitWidthOf(operand.getType()));
    for (auto opr : oprs) {
      auto it = input.operatorInfo.find(opr);
      if (it != input.operatorInfo.end() && it->second.shareable) {
        it->second.width = std::max(it->second.width, widest);
        it->second.operands =
            std::max(it->second.operands, op->getNumOperands());
      }
    }
    // An op with memory effects is never shared (binding leaves it alone).
    if (!isMemoryEffectFree(op))
      input.alone.insert(op);
  }

  // A pipeline's carried results: an op whose result, through bitcasts, is
  // the next iteration's value of an iter arg; read there by that iter
  // arg's users. One passed along to another iter arg, or read after the
  // loop, has no ready cycle to be read in and stays alone.
  if (pipelinedLoop) {
    Block *body = pipelinedLoop.getAfterBody();
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    auto cond =
        cast<scf::ConditionOp>(pipelinedLoop.getBeforeBody()->getTerminator());
    auto inProblem = [&](Operation *op) -> Operation * {
      for (; op; op = op->getParentOp())
        if (problem.hasOperation(op))
          return op;
      return nullptr;
    };
    for (auto [k, next] : llvm::enumerate(yield.getOperands())) {
      Operation *producer = next.getDefiningOp();
      while (producer && isa<arith::BitcastOp>(producer))
        producer = producer->getOperand(0).getDefiningOp();
      if (!producer || !problem.hasOperation(producer))
        continue;
      BlockArgument iterArg = body->getArgument(k);
      bool late = false;
      SmallVector<Operation *, 2> readers;
      for (Operation *user : iterArg.getUsers()) {
        if (user == yield.getOperation()) {
          late = true;
          continue;
        }
        if (Operation *reader = inProblem(user))
          readers.push_back(reader);
      }
      // Read after the loop: the condition forwards the iter arg to a used
      // loop result.
      for (auto [j, forwarded] : llvm::enumerate(cond.getArgs()))
        if (forwarded == pipelinedLoop.getBeforeArguments()[k] &&
            !pipelinedLoop.getResult(j).use_empty())
          late = true;
      if (late)
        input.aloneMultiCycle.insert(producer);
      else
        input.carriedReaders[producer].append(readers);
    }
  }

  return input;
}

template <typename ProblemT>
CPSATFunctionModel::Region &CPSATLowering::captureRegion(
    ProblemT &problem, Operation *anchor, Operation *owner,
    scf::WhileOp pipelinedLoop, Value condValue, const InputDelays &delays,
    CPSATFunctionModel &model) {
  CPSATRegionInput input = buildCPSATInput(problem, anchor, pipelinedLoop);

  // The function's operators, by index.
  auto oprIndex = [&](Problem::OperatorType opr) -> unsigned {
    for (auto [k, o] : llvm::enumerate(model.oprs))
      if (o.name == opr.getValue())
        return k;
    CPSATFunctionModel::Operator o;
    o.name = opr.getValue().str();
    o.latency = *problem.getLatency(opr);
    o.inc = *problem.getIncomingDelay(opr);
    o.out = *problem.getOutgoingDelay(opr);
    o.info = input.operatorInfo.lookup(opr);
    model.oprs.push_back(o);
    return model.oprs.size() - 1;
  };

  CPSATFunctionModel::Region region;
  DenseMap<Operation *, unsigned> index;
  for (auto [i, op] : llvm::enumerate(problem.getOperations()))
    index[op] = i;
  // The delay model's terms (InputDelays): the held results' producers
  // (recorded, the model taking the delay of the operator each is given) and
  // their candidates' slowest (for a solve without them); a pipeline op's
  // dynamic loads, read in a load's expect stage through its FIFO's bypass
  // (one input more on a shared instance's mux).
  unsigned self = model.regions.size();
  auto termsOf = [&](Operation *op, unsigned i, CPSATFunctionModel::Op &o) {
    auto it = delays.ops.find(op);
    if (it == delays.ops.end())
      return;
    for (Operation *def : it->second.held) {
      SmallVector<StringRef, 4> names;
      if (auto offered =
              def->getAttrOfType<ArrayAttr>("loopschedule.candidates"))
        for (auto sym : offered.getAsRange<FlatSymbolRefAttr>())
          names.push_back(sym.getValue());
      if (auto own = def->getAttrOfType<SymbolRefAttr>("loopschedule.operator"))
        names.push_back(own.getLeafReference());
      for (StringRef name : names)
        if (operatorLibraryAnalysis->hasOperator(name) &&
            operatorLibraryAnalysis->getOperatorLatency(name) > 0)
          o.heldWorst = std::max(
              o.heldWorst,
              operatorLibraryAnalysis->getOperatorOutgoingDelay(name)
                      .value_or(0.0f) +
                  delays.gate);
      cpsatHeldPending.push_back({self, unsigned(i), def});
    }
    for (Operation *load : it->second.expectReads)
      o.expectReads.push_back(index[load]);
  };
  for (auto [i, op] : llvm::enumerate(problem.getOperations())) {
    cpsatOpAt[op] = {self, unsigned(i)};
    CPSATFunctionModel::Op o;
    o.name = op->getName().getStringRef().str();
    auto linked = *problem.getLinkedOperatorType(op);
    o.oprs.push_back(oprIndex(linked));
    if (auto it = input.candidates.find(op); it != input.candidates.end())
      for (auto opr : it->second)
        if (!(opr == linked))
          o.oprs.push_back(oprIndex(opr));
    o.alone = input.alone.contains(op);
    o.aloneMultiCycle = input.aloneMultiCycle.contains(op);
    if (auto it = input.carriedReaders.find(op);
        it != input.carriedReaders.end())
      for (Operation *reader : it->second)
        if (auto r = index.find(reader); r != index.end())
          o.carriedReaders.push_back(r->second);
    for (Value result : op->getResults())
      o.width += bitWidthOf(result.getType());
    o.dynamic = isDynamicLatencyOp(*op);
    termsOf(op, i, o);
    region.ops.push_back(std::move(o));
  }
  // The values on the ops' operands: their users of one shared instance
  // put one input on its mux.
  CPSATValueIds &ids = cpsatValueIds[&model];
  DenseMap<Value, unsigned> local;
  for (auto [i, op] : llvm::enumerate(problem.getOperations()))
    for (Value operand : op->getOperands()) {
      auto idIn = [&](auto &map, auto key) {
        auto [it, added] = map.try_emplace(key, ids.next);
        if (added)
          ++ids.next;
        return it->second;
      };
      Attribute constant;
      auto arg = dyn_cast<BlockArgument>(operand);
      unsigned id;
      if (matchPattern(operand, m_Constant(&constant)))
        id = idIn(ids.constants, constant);
      else if (arg && isa<FunctionOpInterface>(arg.getOwner()->getParentOp()))
        id = idIn(ids.arguments, operand);
      else
        id = idIn(local, operand);
      region.ops[i].inputs.push_back(id);
    }
  // The child loops its ops launch: lowered already (inner regions first),
  // each carrying the index of the region it was described as.
  for (auto [i, launcher] : llvm::enumerate(problem.getOperations())) {
    Operation *op = launcher;
    op->walk<WalkOrder::PreOrder>([&](Operation *nested) {
      auto child = nested->getAttrOfType<IntegerAttr>(kCPSATRegionAttr);
      if (!child)
        return WalkResult::advance();
      if (unsigned c = child.getInt(); c < self) {
        model.regions[c].parent = self;
        model.regions[c].launch = i;
        // The operators of the child and the loops inside it (described
        // already, each linked to its parent).
        llvm::SmallSetVector<unsigned, 8> oprs;
        for (auto [q, inner] : llvm::enumerate(model.regions)) {
          // Up the parents (one may be this region, not described yet).
          std::optional<unsigned> up = q;
          while (up && *up != c && *up < self)
            up = model.regions[*up].parent;
          if (up && *up == c)
            for (auto &innerOp : inner.ops)
              oprs.insert(innerOp.oprs.begin(), innerOp.oprs.end());
        }
        SmallVector<unsigned> sorted(oprs.begin(), oprs.end());
        llvm::sort(sorted);
        region.ops[i].childOprs.append(sorted.begin(), sorted.end());
      }
      return WalkResult::skip(); // the loops inside it are its own
    });
  }
  if (isa<scf::WhileOp>(owner))
    owner->setAttr(kCPSATRegionAttr,
                   IntegerAttr::get(IntegerType::get(owner->getContext(), 64),
                                    self));
  for (Operation *op : problem.getOperations())
    for (auto dep : problem.getDependences(op)) {
      auto src = index.find(dep.getSource());
      if (src == index.end())
        continue;
      CPSATFunctionModel::Dependence d;
      d.src = src->second;
      d.dst = index[op];
      d.auxiliary = dep.isAuxiliary();
      if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>)
        d.distance = problem.getDistance(dep).value_or(0);
      if (d.auxiliary) {
        unsigned depLat = *problem.getDependenceLatency(dep);
        unsigned linkedLat =
            *problem.getLatency(*problem.getLinkedOperatorType(dep.getSource()));
        // Past the source's latency, a bound of its own (whichever
        // operator the source takes); short of it, the edge's latency.
        if (depLat > linkedLat)
          d.minLatency = depLat;
        else if (depLat != linkedLat)
          d.latency = depLat;
      }
      region.deps.push_back(d);
    }
  llvm::MapVector<Problem::ResourceType, unsigned> rsrcIndex;
  for (Operation *op : problem.getOperations())
    if (auto rsrcs = problem.getLinkedResourceTypes(op))
      for (auto rsrc : *rsrcs) {
        auto limit = problem.getLimit(rsrc);
        if (!limit)
          continue;
        auto [it, inserted] =
            rsrcIndex.try_emplace(rsrc, region.resources.size());
        if (inserted)
          region.resources.push_back({*limit, {}});
        region.resources[it->second].ops.push_back(index[op]);
      }
  region.anchor = index[anchor];
  if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>) {
    // `hls.pipeline = N` (on the loop or the function) asks for II N; a
    // unit one leaves the II to the solver, from the resources' bound up.
    auto requested = owner->getAttrOfType<IntegerAttr>("hls.pipeline");
    region.ii = requested ? unsigned(requested.getInt()) : 1;
    region.searchII = !requested;
  }
  // A pipelined loop's continuation, by createLoopSchedulePipeline's rules.
  if (Operation *condOp = condValue ? condValue.getDefiningOp() : nullptr;
      pipelinedLoop && condOp && index.count(condOp)) {
    region.cond = index[condOp];
    bool dynamic = false;
    pipelinedLoop.getAfter().walk([&](Operation *op) {
      dynamic |= isDynamicLatencyOp(*op) &&
                 !isDrainableStreamPop(op, pipelinedLoop);
    });
    region.condFirst =
        dynamic || !llvm::all_of(pipelinedLoop.getResults(),
                                 [](Value r) { return r.use_empty(); });
    pipelinedLoop.getAfter().walk([&](Operation *op) {
      if (isObservableEffect(*op) && index.count(op) &&
          !isDrainableStreamPop(op, pipelinedLoop))
        region.effects.push_back(index[op]);
    });
  }
  // How often the region runs: its loop's iterations (a pipeline's per
  // launch), times those of the loops around it.
  auto tripOf = [&](Operation *loop) -> int64_t {
    if (auto trip = loop->getAttrOfType<IntegerAttr>("loopschedule.trip_count"))
      return std::max<int64_t>(1, trip.getInt());
    return std::max<int64_t>(1, cpsatDefaultTripCount);
  };
  int64_t outer = 1;
  for (Operation *p = owner->getParentOp(); p; p = p->getParentOp())
    if (isa<scf::WhileOp>(p))
      outer *= tripOf(p);
  if (pipelinedLoop) {
    region.runs = outer;
    region.tripCount = tripOf(owner);
  } else if (isa<scf::WhileOp>(owner)) {
    region.runs = outer * tripOf(owner);
  } else {
    region.runs = 1;
  }
  model.regions.push_back(std::move(region));
  // The held results of this region's ops, read in the regions inside it.
  llvm::erase_if(cpsatHeldPending, [&](auto &pending) {
    auto [r, i, def] = pending;
    auto at = cpsatOpAt.find(def);
    if (at == cpsatOpAt.end() || at->second.first != self)
      return false;
    model.regions[r].ops[i].held.push_back(at->second);
    return true;
  });
  // Described last, the function leaves any producer not found in a region
  // (outside every problem): its reader takes the slowest candidates'.
  if (isa<FuncOp>(owner)) {
    for (auto [r, i, def] : cpsatHeldPending)
      model.regions[r].ops[i].held.clear();
    cpsatHeldPending.clear();
  }
  return model.regions.back();
}

template <typename ProblemT>
LogicalResult CPSATLowering::applyRegionSolution(
    ProblemT &problem, Operation *anchor, const CPSATFunctionModel &model,
    const CPSATFunctionModel::Region &region,
    const CPSATFunctionSolution::Region &sol, bool bind) {
  auto cannot = [&](StringRef why) -> LogicalResult {
    return anchor->emitError("the CP-SAT schedule cannot be applied to this "
                             "region (")
           << why << ")";
  };
  auto &ops = problem.getOperations();
  if (ops.size() != region.ops.size() || sol.start.size() != ops.size())
    return cannot("its operations differ from the description");
  auto *ctx = anchor->getContext();
  SmallVector<Problem::OperatorType> chosen;
  for (auto [i, op] : llvm::enumerate(ops)) {
    if (op->getName().getStringRef() != region.ops[i].name)
      return cannot("its operations differ from the description");
    // An op with no choice keeps its own operator: one named per run (a
    // memory port's, by its memory reference) may be named otherwise on the
    // copy the model was described on.
    if (region.ops[i].oprs.size() == 1) {
      chosen.push_back(*problem.getLinkedOperatorType(op));
      continue;
    }
    auto opr = Problem::OperatorType::get(ctx, model.oprs[sol.opr[i]].name);
    if (!problem.hasOperatorType(opr))
      return cannot("the operator it chose for " +
                    op->getName().getStringRef().str() + ", @" +
                    model.oprs[sol.opr[i]].name + ", is not offered here");
    chosen.push_back(opr);
  }
  if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>) {
    if (!sol.ii)
      return cannot("it has no II for a pipeline");
    problem.setInitiationInterval(sol.ii);
  }
  for (auto [i, op] : llvm::enumerate(ops)) {
    problem.setStartTime(op, sol.start[i]);
    problem.setLinkedOperatorType(op, chosen[i]);
    problem.setStartTimeInCycle(op, sol.inCycle[i]);
  }
  // A pinned region's chains run past the clock, which the model's offsets
  // do not describe: its in-cycle times are the simplex's, from its starts.
  if (region.pinned && failed(computeStartTimesInCycle(problem)))
    return cannot("its in-cycle times cannot be computed");
  if (failed(problem.verify()))
    return cannot("it does not verify");
  auto i64 = IntegerType::get(ctx, 64);
  for (auto [i, op] : llvm::enumerate(ops)) {
    if (candidates.count(op))
      op->setAttr("loopschedule.operator",
                  SymbolRefAttr::get(chosen[i].getAttr()));
    if (bind && sol.instance[i])
      op->setAttr("loopschedule.binding",
                  IntegerAttr::get(i64, *sol.instance[i]));
  }
  return success();
}

CPSATChainingConfig CPSATLowering::cpsatConfig(float cycleTime) {
  CPSATChainingConfig config;
  config.cycleTime = cycleTime;
  config.timeLimitS = cpsatTimeLimit;
  config.numWorkers = cpsatWorkers;
  config.cycleSlack = cpsatCycleSlack;
  config.lutCapacity = cpsatDeviceLuts;
  config.dspCapacity = cpsatDeviceDsps;
  config.ffCapacity = cpsatDeviceFFs;
  config.bottleneck = cpsatArea == "bottleneck";
  config.solverParameters = cpsatParameters;
  auto table = [&](unsigned i, auto &option) {
    muxTables[i].assign(option.begin(), option.end());
    return ArrayRef<float>(muxTables[i]);
  };
  config.muxDelays = table(0, this->config.shareMuxDelays);
  config.muxNorDelays = table(1, this->config.shareMuxNorDelays);
  config.muxDelays64 = table(2, this->config.shareMuxDelays64);
  config.muxNorDelays64 = table(3, this->config.shareMuxNorDelays64);
  config.muxInputLuts = cpsatMuxInputLuts;
  return config;
}

LogicalResult CPSATLowering::solveFunctionsCPSAT(ModuleOp moduleOp) {
  // Describe every region on a copy, each scheduled alone in CP-SAT and
  // lowered as usual. Its diagnostics are the real run's too (which repeats
  // them), but for the first error: why the copy failed.
  OwningOpRef<ModuleOp> copy = moduleOp.clone();
  SCFToLoopScheduleCPSATOptions opts = options;
  opts.recordChainSlack = false;
  auto pass =
      std::make_unique<SCFToLoopScheduleCPSATPass>(opts, &functionModels);
  PassManager pm(context);
  pm.addPass(std::move(pass));
  LogicalResult described = success();
  std::string firstError;
  {
    ScopedDiagnosticHandler capture(context, [&](Diagnostic &diag) {
      if (diag.getSeverity() == DiagnosticSeverity::Error &&
          firstError.empty()) {
        llvm::raw_string_ostream os(firstError);
        os << diag.getLocation() << ": " << diag;
      }
      return success();
    });
    described = pm.run(*copy);
  }
  if (failed(described))
    return moduleOp.emitError("the CP-SAT scheduler could not describe the "
                              "regions (scheduling a copy of the module "
                              "failed")
           << (firstError.empty() ? std::string() : ": " + firstError) << ")";

  CPSATChainingConfig config = cpsatConfig(this->config.cycleTimeNs);
  for (auto &entry : functionModels) {
    CPSATFunctionSolution solution;
    std::string why;
    if (failed(scheduleCPSATFunction(entry.second, config, solution, why)))
      return moduleOp.emitError("the CP-SAT scheduler found no schedule for "
                                "@")
             << entry.first() << " (" << why << ")";
    LLVM_DEBUG(llvm::dbgs()
               << "cpsat: @" << entry.first() << " cycles " << solution.cycles
               << ", " << solution.luts << " LUTs, " << solution.dsps
               << " DSPs, " << solution.ffs << " FFs"
               << (solution.optimal ? " optimal" : "") << "\n");
    functionSolutions[entry.first()] = std::move(solution);
  }
  return success();
}
#endif

template <typename ProblemT>
LogicalResult CPSATLowering::scheduleRegionCPSAT(
    ProblemT &problem, Operation *anchor, Operation *owner,
    scf::WhileOp pipelinedLoop, float cycleTime, bool seeded,
    const InputDelays &delays, Value condValue) {
#ifndef SCHEDULING_OR_TOOLS
  return anchor->emitError("the CP-SAT scheduler needs CIRCT built with "
                           "OR-Tools");
#else
  unsigned r = regionIndex++;
  auto func = anchor->getParentOfType<FuncOp>();
  if (!func)
    return anchor->emitError("the CP-SAT scheduler has no function for this "
                             "region");
  if (snapshotSink) {
    // Describe the region, and find the schedule the function's solve
    // starts from: the simplex's (seeded), where the model takes it, or the
    // region's alone in CP-SAT, for its cycles. The copy is lowered on it,
    // and the regions around it described on what it lowers to (a child
    // loop is a launch of a fixed latency to its parent, whatever its
    // schedule).
    CPSATFunctionModel &model = (*snapshotSink)[func.getSymName()];
    CPSATFunctionModel::Region &region =
        captureRegion(problem, anchor, owner, pipelinedLoop, condValue,
                      delays, model);
    CPSATChainingConfig config = cpsatConfig(cycleTime);
    config.cyclesOnly = cpsatRegionObjective != "area";
    config.timeLimitS = std::min<double>(cpsatTimeLimit, cpsatRegionTimeLimit);
    auto alone = [&] {
      CPSATFunctionModel one;
      one.oprs = model.oprs;
      one.regions.push_back(region);
      return one;
    };
    if (seeded) {
      // Each op on its own operator (allocation's, the first of its
      // candidates) at the simplex's start. The model may not take it: the
      // simplex keeps a recurrence's chain whole past the clock below an II
      // it was given, takes an operator past the clock where one fits, and
      // knows nothing of a child's launch beside its parent's dynamic
      // access.
      for (auto [i, op] : llvm::enumerate(problem.getOperations())) {
        region.ops[i].hint = *problem.getStartTime(op);
        region.ops[i].hintChoice = 0;
      }
      if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>)
        region.hintII = *problem.getInitiationInterval();
      CPSATChainingConfig check = config;
      check.fixHints = true;
      CPSATFunctionSolution solution;
      std::string why;
      if (succeeded(scheduleCPSATFunction(alone(), check, solution, why)))
        return success(); // the copy is lowered on the simplex's schedule
      LLVM_DEBUG(llvm::dbgs() << "cpsat: region " << r
                              << ": the model does not take the simplex's "
                                 "schedule ("
                              << why << "); solving it alone\n");
    }
    CPSATFunctionSolution solution;
    std::string why;
    if (failed(scheduleCPSATFunction(alone(), config, solution, why))) {
      if (!seeded)
        return anchor->emitError("the CP-SAT scheduler found no schedule for "
                                 "this region (")
               << why << ")";
      // A requested II is a strict constraint: where it closes only past
      // the clock, the II takes precedence, as the simplex takes it. The
      // region keeps the simplex's schedule, pinned in the function's
      // model, with no clock on it.
      if (region.ii > 0 && !region.searchII) {
        LLVM_DEBUG(llvm::dbgs()
                   << "cpsat: region " << r << ": no schedule at its requested "
                   << "II " << region.ii << " within the clock (" << why
                   << "); pinned to the simplex's, past the clock\n");
        region.pinned = true;
        // What a shared instance's mux may add in front of each op: its
        // chain's slack, less a pipeline's late select at cycle 0 (the
        // stage-0 enable, a LUT after the condition).
        auto distance = [&](Problem::Dependence dep) -> unsigned {
          if constexpr (std::is_base_of_v<CyclicProblem, ProblemT>)
            return problem.getDistance(dep).value_or(0);
          return 0;
        };
        auto slacks = chainSlacks(problem, cycleTime, distance);
        auto ops = problem.getOperations();
        std::optional<float> condIn;
        if (region.cond) {
          Operation *cond = ops[*region.cond];
          if (auto opr = problem.getLinkedOperatorType(cond))
            condIn = (*problem.getLatency(*opr) == 0
                          ? problem.getStartTimeInCycle(cond).value_or(0.0f)
                          : 0.0f) +
                     *problem.getOutgoingDelay(*opr);
        }
        for (auto [i, op] : llvm::enumerate(ops)) {
          auto slack = slacks.find(op);
          if (slack == slacks.end())
            continue;
          float budget = slack->second;
          if (condIn && i != *region.cond && *problem.getStartTime(op) == 0)
            budget -= std::max(0.0f, *condIn + delays.gate -
                                         *problem.getStartTimeInCycle(op));
          region.ops[i].pinnedMuxBudget = budget;
        }
        return success();
      }
      // The copy keeps the simplex's schedule, and the function's solve
      // starts the region from nothing.
      LLVM_DEBUG(llvm::dbgs() << "cpsat: region " << r << ": no schedule ("
                              << why << "); the function's solve starts it "
                                        "from nothing\n");
      for (auto &op : region.ops)
        op.hint.reset();
      region.hintII.reset();
      return success();
    }
    auto &sol = solution.regions.front();
    for (auto [i, op] : llvm::enumerate(region.ops)) {
      op.hint = sol.start[i];
      op.hintChoice = llvm::find(op.oprs, sol.opr[i]) - op.oprs.begin();
    }
    if (sol.ii)
      region.hintII = sol.ii;
    return applyRegionSolution(problem, anchor, model, region, sol,
                               /*bind=*/false);
  }
  {
    auto modelIt = functionModels.find(func.getSymName());
    auto solIt = functionSolutions.find(func.getSymName());
    if (modelIt == functionModels.end() || solIt == functionSolutions.end())
      return anchor->emitError("the CP-SAT scheduler has no solution for this "
                               "region's function");
    if (r >= modelIt->second.regions.size() ||
        r >= solIt->second.regions.size())
      return anchor->emitError("the CP-SAT scheduler described fewer regions "
                               "of this function");
    auto &sol = solIt->second.regions[r];
    if (failed(applyRegionSolution(problem, anchor, modelIt->second,
                                   modelIt->second.regions[r], sol,
                                   /*bind=*/true)))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "cpsat: region " << r << " length "
                            << sol.length << (sol.ii ? " II " : "")
                            << (sol.ii ? std::to_string(sol.ii) : "") << "\n");
    return success();
  }
#endif
}

LogicalResult CPSATLowering::prepare(ModuleOp moduleOp) {
  if (snapshotSink)
    return success();
#ifdef SCHEDULING_OR_TOOLS
  return solveFunctionsCPSAT(moduleOp);
#else
  return moduleOp.emitError("convert-scf-to-loopschedule-cpsat needs CIRCT "
                            "built with OR-Tools");
#endif
}

LogicalResult CPSATLowering::scheduleModulo(ChainingModuloProblem &problem,
                                            Operation *anchor, Operation *owner,
                                            scf::WhileOp pipelinedLoop,
                                            std::optional<int32_t> ii,
                                            Value condValue,
                                            const InputDelays &delays) {
  // On the copy, the simplex's schedule first: the CP-SAT search's start.
  bool seeded = seedSimplex([&]() -> LogicalResult {
    if (failed(simplexModulo(problem, anchor, ii, delays)) ||
        (ii && problem.getInitiationInterval() != ii))
      return failure();
    return success();
  });
  return scheduleRegionCPSAT(problem, anchor, owner, pipelinedLoop,
                             this->config.cycleTimeNs, seeded, delays,
                             condValue);
}

LogicalResult
CPSATLowering::scheduleSequential(ChainingSharedOperatorsProblem &problem,
                                  Operation *anchor, Operation *owner,
                                  const InputDelays &delays) {
  // On the copy, the simplex's schedule first: the CP-SAT search's start.
  bool seeded = seedSimplex([&]() -> LogicalResult {
    return simplexSequential(problem, anchor, delays);
  });
  return scheduleRegionCPSAT(problem, anchor, owner, scf::WhileOp(),
                             this->config.cycleTimeNs, seeded, delays);
}

SCFToLoopScheduleCPSATOptions SCFToLoopScheduleCPSATPass::getOptions() {
  SCFToLoopScheduleCPSATOptions opts;
  opts.cycleTimeNs = cycleTimeNs;
  opts.recordChainSlack = recordChainSlack;
  auto list = [](auto &option) {
    return SmallVector<float>(option.begin(), option.end());
  };
  opts.shareMuxDelays = list(shareMuxDelays);
  opts.shareMuxNorDelays = list(shareMuxNorDelays);
  opts.shareMuxDelays64 = list(shareMuxDelays64);
  opts.shareMuxNorDelays64 = list(shareMuxNorDelays64);
  opts.cpsatTimeLimit = cpsatTimeLimit;
  opts.cpsatRegionTimeLimit = cpsatRegionTimeLimit;
  opts.cpsatRegionObjective = cpsatRegionObjective;
  opts.cpsatCycleSlack = cpsatCycleSlack;
  opts.cpsatDefaultTripCount = cpsatDefaultTripCount;
  opts.cpsatWorkers = cpsatWorkers;
  opts.cpsatParameters = cpsatParameters;
  opts.cpsatArea = cpsatArea;
  opts.cpsatDeviceLuts = cpsatDeviceLuts;
  opts.cpsatDeviceDsps = cpsatDeviceDsps;
  opts.cpsatDeviceFFs = cpsatDeviceFFs;
  opts.cpsatMuxInputLuts = cpsatMuxInputLuts;
  return opts;
}

void SCFToLoopScheduleCPSATPass::runOnOperation() {
  SCFToLoopScheduleCPSATOptions opts = getOptions();
  LoweringConfig config;
  config.cycleTimeNs = opts.cycleTimeNs;
  config.recordChainSlack = opts.recordChainSlack;
  config.shareMuxDelays.assign(opts.shareMuxDelays.begin(),
                               opts.shareMuxDelays.end());
  config.shareMuxNorDelays.assign(opts.shareMuxNorDelays.begin(),
                                  opts.shareMuxNorDelays.end());
  config.shareMuxDelays64.assign(opts.shareMuxDelays64.begin(),
                                 opts.shareMuxDelays64.end());
  config.shareMuxNorDelays64.assign(opts.shareMuxNorDelays64.begin(),
                                    opts.shareMuxNorDelays64.end());
  CPSATLowering lowering(std::move(config), getAnalysisManager(),
                         &getContext(), opts, snapshotSink);
  if (failed(lowering.run(getOperation())))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::createSCFToLoopScheduleCPSATPass(
    const SCFToLoopScheduleCPSATOptions &options) {
  return std::make_unique<SCFToLoopScheduleCPSATPass>(options);
}
