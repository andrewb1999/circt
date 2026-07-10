//===- SimplexSchedulers.cpp - Linear programming-based schedulers --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of linear programming-based schedulers with a built-in simplex
// solver.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Utilities.h"

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "simplex-schedulers"

using namespace circt;
using namespace circt::scheduling;

using llvm::dbgs;
using llvm::format;

namespace {

/// This class provides a framework to model certain scheduling problems as
/// lexico-parametric linear programs (LP), which are then solved with an
/// extended version of the dual simplex algorithm.
///
/// The approach is described in:
///  [1] B. D. de Dinechin, "Simplex Scheduling: More than Lifetime-Sensitive
///      Instruction Scheduling", PRISM 1994.22, 1994.
///  [2] B. D. de Dinechin, "Fast Modulo Scheduling Under the Simplex Scheduling
///      Framework", PRISM 1995.01, 1995.
///
/// Resource-free scheduling problems (called "central problems" in the papers)
/// have an *integer* linear programming formulation with a totally unimodular
/// constraint matrix. Such ILPs can however be solved optimally in polynomial
/// time with a (non-integer) LP solver (such as the simplex algorithm), as the
/// LP solution is guaranteed to be integer. Note that this is the same idea as
/// used by SDC-based schedulers.
class SimplexSchedulerBase {
protected:
  /// The objective is to minimize the start time of this operation.
  Operation *lastOp;

  /// S is part of a mechanism to assign fixed values to the LP variables.
  int parameterS;

  /// T represents the initiation interval (II). Its minimally-feasible value is
  /// computed by the algorithm.
  int parameterT;

  /// The simplex tableau is the algorithm's main data structure.
  /// The dashed parts always contain the zero respectively the identity matrix,
  /// and therefore are not stored explicitly.
  ///
  ///                        ◀───nColumns────▶
  ///           nParameters────┐
  ///                        ◀─┴─▶
  ///                       ┌─────┬───────────┬ ─ ─ ─ ─ ┐
  ///                      ▲│. . .│. . ... . .│    0        ▲
  ///           nObjectives││. . .│. . ... . .│         │   │
  ///                      ▼│. . .│. . ... . .│             │
  ///                       ├─────┼───────────┼ ─ ─ ─ ─ ┤   │
  ///  firstConstraintRow > │. . .│. . ... . .│1            │nRows
  ///                       │. . .│. . ... . .│  1      │   │
  ///                       │. . .│. . ... . .│    1        │
  ///                       │. . .│. . ... . .│      1  │   │
  ///                       │. . .│. . ... . .│        1    ▼
  ///                       └─────┴───────────┴ ─ ─ ─ ─ ┘
  ///       parameter1Column ^
  ///         parameterSColumn ^
  ///           parameterTColumn ^
  ///  firstNonBasicVariableColumn ^
  ///                              ─────────── ──────────
  ///                       nonBasicVariables   basicVariables
  SmallVector<SmallVector<int>> tableau;

  /// During the pivot operation, one column in the elided part of the tableau
  /// is modified; this vector temporarily catches the changes.
  SmallVector<int> implicitBasicVariableColumnVector;

  /// The linear program models the operations' start times as variables, which
  /// we identify here as 0, ..., |ops|-1.
  /// Additionally, for each dependence (precisely, the inequality modeling the
  /// precedence constraint), a slack variable is required; these are identified
  /// as |ops|, ..., |ops|+|deps|-1.
  ///
  /// This vector stores the numeric IDs of non-basic variables. A variable's
  /// index *i* in this vector corresponds to the tableau *column*
  /// `firstNonBasicVariableColumn`+*i*.
  SmallVector<unsigned> nonBasicVariables;

  /// This vector store the numeric IDs of basic variables. A variable's index
  /// *i* in this vector corresponds to the tableau *row*
  /// `firstConstraintRow`+*i*.
  SmallVector<unsigned> basicVariables;

  /// Used to conveniently retrieve an operation's start time variable. The
  /// alternative would be to find the op's index in the problem's list of
  /// operations.
  DenseMap<Operation *, unsigned> startTimeVariables;

  /// This vector keeps track of the current locations (i.e. row or column) of
  /// a start time variable in the tableau. We encode column numbers as positive
  /// integers, and row numbers as negative integers. We do not track the slack
  /// variables.
  SmallVector<int> startTimeLocations;

  /// Non-basic variables can be "frozen" to a specific value, which prevents
  /// them from being pivoted into basis again.
  DenseMap<unsigned, unsigned> frozenVariables;

  /// Whether `solveTableau` may unilaterally increase `parameterT` (the II)
  /// to restore feasibility. Schedulers whose driver owns the II — and whose
  /// modulo reservation table is keyed on it — must disable this: a mid-
  /// placement II change would silently invalidate every occupied congruence
  /// slot. With the flag off, such rows report plain infeasibility, which
  /// the backtracking placement handles (and the driver raises the II
  /// between attempts).
  bool allowIIIncreaseDuringSolve = true;

  /// Number of rows in the tableau = |obj| + |deps|.
  unsigned nRows;
  /// Number of explicitly stored columns in the tableau = |params| + |ops|.
  unsigned nColumns;

  // Number of objective rows.
  unsigned nObjectives;
  /// All other rows encode linear constraints.
  unsigned &firstConstraintRow = nObjectives;

  // Number of parameters (fixed for now).
  static constexpr unsigned nParameters = 3;
  /// The first column corresponds to the always-one "parameter" in u = (1,S,T).
  static constexpr unsigned parameter1Column = 0;
  /// The second column corresponds to the variable-freezing parameter S.
  static constexpr unsigned parameterSColumn = 1;
  /// The third column corresponds to the parameter T, i.e. the current II.
  static constexpr unsigned parameterTColumn = 2;
  /// All other (explicitly stored) columns represent non-basic variables.
  static constexpr unsigned firstNonBasicVariableColumn = nParameters;

  /// Allow subclasses to collect additional constraints that are not part of
  /// the input problem, but should be modeled in the linear problem.
  SmallVector<Problem::Dependence> additionalConstraints;

  virtual Problem &getProblem() = 0;
  virtual LogicalResult checkLastOp();
  virtual bool fillObjectiveRow(SmallVector<int> &row, unsigned obj);
  virtual void fillConstraintRow(SmallVector<int> &row,
                                 Problem::Problem::Dependence dep);
  virtual void fillAdditionalConstraintRow(SmallVector<int> &row,
                                           Problem::Problem::Dependence dep);
  void buildTableau();
  /// Discard the tableau and all variable bookkeeping (including frozen
  /// variables) so `buildTableau` can be called again. `parameterT` (the II)
  /// is preserved. Backtracking schedulers use this to "unfreeze" pinned
  /// operations: rather than surgically undoing an old equality in a tableau
  /// that has pivoted arbitrarily since, they rebuild from scratch and
  /// re-apply the surviving pins.
  void resetTableau();

  int getParametricConstant(unsigned row);
  SmallVector<int> getObjectiveVector(unsigned column);
  std::optional<unsigned> findDualPivotRow();
  std::optional<unsigned> findDualPivotColumn(unsigned pivotRow,
                                              bool allowPositive = false);
  std::optional<unsigned> findPrimalPivotColumn();
  std::optional<unsigned> findPrimalPivotRow(unsigned pivotColumn);
  void multiplyRow(unsigned row, int factor);
  void addMultipleOfRow(unsigned sourceRow, int factor, unsigned targetRow);
  void pivot(unsigned pivotRow, unsigned pivotColumn);
  LogicalResult solveTableau();
  LogicalResult restoreDualFeasibility();
  bool isInBasis(unsigned startTimeVariable);
  unsigned freeze(unsigned startTimeVariable, unsigned timeStep);
  void translate(unsigned column, int factor1, int factorS, int factorT);
  LogicalResult scheduleAt(unsigned startTimeVariable, unsigned timeStep);
  void moveBy(unsigned startTimeVariable, unsigned amount);
  unsigned getStartTime(unsigned startTimeVariable);

  void dumpTableau();

public:
  explicit SimplexSchedulerBase(Operation *lastOp) : lastOp(lastOp) {}
  virtual ~SimplexSchedulerBase() = default;
  virtual LogicalResult schedule() = 0;
};

/// This class solves the basic, acyclic `Problem`.
class SimplexScheduler : public SimplexSchedulerBase {
private:
  Problem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SimplexScheduler(Problem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}

  LogicalResult schedule() override;
};

/// This class solves the resource-free `CyclicProblem`.  The optimal initiation
/// interval (II) is determined as a side product of solving the parametric
/// problem, and corresponds to the "RecMII" (= recurrence-constrained minimum
/// II) usually considered as one component in the lower II bound used by modulo
/// schedulers.
class CyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  CyclicProblem &prob;

protected:
  Problem &getProblem() override { return prob; }
  void fillConstraintRow(SmallVector<int> &row,
                         Problem::Problem::Dependence dep) override;

public:
  CyclicSimplexScheduler(CyclicProblem &prob, Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves acyclic, resource-constrained `SharedOperatorsProblem` with
// a simplified version of the iterative heuristic presented in [2].
class SharedOperatorsSimplexScheduler : public SimplexSchedulerBase {
private:
  SharedOperatorsProblem &prob;

protected:
  Problem &getProblem() override { return prob; }

public:
  SharedOperatorsSimplexScheduler(SharedOperatorsProblem &prob,
                                  Operation *lastOp)
      : SimplexSchedulerBase(lastOp), prob(prob) {}
  LogicalResult schedule() override;
};

// This class solves the `ModuloProblem` using the iterative heuristic presented
// in [2].
class ModuloSimplexScheduler : public CyclicSimplexScheduler {
private:
  struct MRT {
    ModuloSimplexScheduler &sched;

    using TableType = SmallDenseMap<unsigned, DenseSet<Operation *>>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::ResourceType, TableType> tables;
    SmallDenseMap<Problem::ResourceType, ReverseTableType> reverseTables;

    explicit MRT(ModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(Operation *op, unsigned timeStep);
    void release(Operation *op);
  };

  ModuloProblem &prob;
  SmallVector<unsigned> asapTimes, alapTimes;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;

protected:
  Problem &getProblem() override { return prob; }
  LogicalResult checkLastOp() override;
  enum { OBJ_LATENCY = 0, OBJ_AXAP /* i.e. either ASAP or ALAP */ };
  bool fillObjectiveRow(SmallVector<int> &row, unsigned obj) override;
  void updateMargins();
  LogicalResult scheduleOperation(Operation *n);
  unsigned computeResMinII();

public:
  ModuloSimplexScheduler(ModuloProblem &prob, Operation *lastOp)
      : CyclicSimplexScheduler(prob, lastOp), prob(prob), mrt(*this) {}
  LogicalResult schedule() override;
};

// This class solves the `ChainingProblem` by relying on pre-computed
// chain-breaking constraints.
class ChainingSimplexScheduler : public SimplexSchedulerBase {
private:
  ChainingProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingSimplexScheduler(ChainingProblem &prob, Operation *lastOp,
                           float cycleTime)
      : SimplexSchedulerBase(lastOp), prob(prob), cycleTime(cycleTime) {}
  LogicalResult schedule() override;
};

class ChainingSharedOperatorsSimplexScheduler : public SimplexSchedulerBase {
private:
  ChainingSharedOperatorsProblem &prob;
  float cycleTime;

protected:
  Problem &getProblem() override { return prob; }
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingSharedOperatorsSimplexScheduler(ChainingSharedOperatorsProblem &prob,
                                          Operation *lastOp, float cycleTime)
      : SimplexSchedulerBase(lastOp), prob(prob), cycleTime(cycleTime) {}
  LogicalResult schedule() override;
};

// This class solves the resource-free `ChainingCyclicProblem` by relying on
// pre-computed chain-breaking constraints. The optimal initiation interval (II)
// is determined as a side product of solving the parametric problem, and
// corresponds to the "RecMII" (= recurrence-constrained minimum II) usually
// considered as one component in the lower II bound used by modulo schedulers.
class ChainingCyclicSimplexScheduler : public SimplexSchedulerBase {
private:
  ChainingCyclicProblem &prob;
  float cycleTime;
  std::optional<int32_t> ii;

protected:
  Problem &getProblem() override { return prob; }
  void fillConstraintRow(SmallVector<int> &row,
                         Problem::Dependence dep) override;
  void fillAdditionalConstraintRow(SmallVector<int> &row,
                                   Problem::Dependence dep) override;

public:
  ChainingCyclicSimplexScheduler(ChainingCyclicProblem &prob, Operation *lastOp,
                                 float cycleTime,
                                 std::optional<int32_t> ii = std::nullopt)
      : SimplexSchedulerBase(lastOp), prob(prob), cycleTime(cycleTime), ii(ii) {
  }
  LogicalResult schedule() override;
};

class ChainingModuloSimplexScheduler : public ChainingCyclicSimplexScheduler {
private:
  struct MRT {
    ChainingModuloSimplexScheduler &sched;

    using TableType = SmallDenseMap<unsigned, DenseSet<Operation *>>;
    using ReverseTableType = SmallDenseMap<Operation *, unsigned>;
    SmallDenseMap<Problem::ResourceType, TableType> tables;
    SmallDenseMap<Problem::ResourceType, ReverseTableType> reverseTables;

    explicit MRT(ChainingModuloSimplexScheduler &sched) : sched(sched) {}
    LogicalResult enter(Operation *op, unsigned timeStep);
    void release(Operation *op);
    /// Collect the current occupants blocking `op` at `timeStep`'s
    /// congruence slot: for each of `op`'s resource types whose cell is at
    /// its limit, every op in that cell (Rau §3.4: "all operations are
    /// unscheduled which conflict").
    void gatherConflicts(Operation *op, unsigned timeStep,
                         SmallVectorImpl<Operation *> &conflicts);
  };

  ChainingModuloProblem &prob;
  SmallVector<Operation *> unscheduled, scheduled;
  MRT mrt;
  float cycleTime;
  std::optional<int32_t> ii;

  /// Backtracking state (Rau, "Iterative Modulo Scheduling", MICRO-27 1994;
  /// Canis et al., "Modulo SDC Scheduling", FPL 2014). `pinnedOps` holds the
  /// committed (op -> start time) equalities currently baked into the
  /// tableau; unpinning removes entries and rebuilds the tableau from the
  /// survivors. `prevSched` implements the forward-progress rule: a
  /// re-placed op never returns to its previous slot, so two ops cannot
  /// displace each other endlessly.
  DenseMap<Operation *, unsigned> pinnedOps;
  DenseMap<Operation *, unsigned> prevSched;

  /// Rau's budget: maximum number of (re-)placements per II attempt, as a
  /// multiple of the operation count. Both Rau and Canis found ~6 to be a
  /// good trade-off empirically.
  static constexpr unsigned budgetRatio = 6;

protected:
  Problem &getProblem() override { return prob; }
  LogicalResult checkLastOp() override;
  enum { OBJ_LATENCY = 0, OBJ_AXAP /* i.e. either ASAP or ALAP */ };
  bool fillObjectiveRow(SmallVector<int> &row, unsigned obj) override;
  LogicalResult scheduleOperation(Operation *n);
  unsigned computeResMinII();

  /// Rau's height-based priority (MICRO-27 1994, Fig. 5): the longest
  /// dependence path toward the schedule's end, with inter-iteration edges
  /// discounted by II*distance. Static per II attempt — unlike the previous
  /// least-slack heuristic it needs no per-pick tableau re-solves, and it
  /// places producers of long (e.g. port-serialized) chains before their
  /// dependent consumers, keeping eviction (and budget consumption) rare.
  /// This ordering also composes with dependence chains generated for
  /// dynamic-latency launches, which matters once function-frame dyn
  /// accesses are wrapped in launches (see benchmarks/DEBUG_NOTES.md).
  DenseMap<Operation *, int> heightPriority;
  /// Deterministic tie-break: the op's index in the problem's op list.
  DenseMap<Operation *, unsigned> stableIndex;
  void computeHeightPriorities();
  /// Unpin `op`: drop its equality, release its MRT slot, and requeue it.
  /// The tableau is NOT rebuilt here — batch unpins, then rebuildWithPins().
  void unpin(Operation *op);
  /// Rebuild the tableau from scratch and re-apply all surviving pins.
  LogicalResult rebuildWithPins();

public:
  /// Added on top of the resource-derived minimum II for this attempt; the
  /// driver bumps it when an attempt fails (budget exhausted or genuinely
  /// infeasible), mirroring Rau's ModuloSchedule outer loop.
  unsigned iiIncrement = 0;

  ChainingModuloSimplexScheduler(ChainingModuloProblem &prob, Operation *lastOp,
                                 float cycleTime,
                                 std::optional<int32_t> ii = std::nullopt)
      : ChainingCyclicSimplexScheduler(prob, lastOp, cycleTime, ii), prob(prob),
        mrt(*this), cycleTime(cycleTime), ii(ii) {
    // The driver owns the II (fresh attempt per candidate II); the solver
    // must not change it under the MRT's feet.
    allowIIIncreaseDuringSolve = false;
  }
  LogicalResult schedule() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// SimplexSchedulerBase
//===----------------------------------------------------------------------===//

LogicalResult SimplexSchedulerBase::checkLastOp() {
  auto &prob = getProblem();
  if (!prob.hasOperation(lastOp))
    return prob.getContainingOp()->emitError(
        "problem does not include last operation");
  return success();
}

bool SimplexSchedulerBase::fillObjectiveRow(SmallVector<int> &row,
                                            unsigned obj) {
  assert(obj == 0);
  // Minimize start time of user-specified last operation.
  row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
  return false;
}

void SimplexSchedulerBase::fillConstraintRow(SmallVector<int> &row,
                                             Problem::Dependence dep) {
  auto &prob = getProblem();
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  row[parameter1Column] = -latency; // note the negation
  if (src != dst) { // note that these coefficients just zero out in self-arcs.
    row[startTimeLocations[startTimeVariables[src]]] = 1;
    row[startTimeLocations[startTimeVariables[dst]]] = -1;
  }
}

void SimplexSchedulerBase::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  // Handling is subclass-specific, so do nothing by default.
  (void)row;
  (void)dep;
}

void SimplexSchedulerBase::buildTableau() {
  auto &prob = getProblem();

  // The initial tableau is constructed so that operations' start time variables
  // are out of basis, whereas all slack variables are in basis. We will number
  // them accordingly.
  unsigned var = 0;

  // Assign column and variable numbers to the operations' start times.
  for (auto *op : prob.getOperations()) {
    nonBasicVariables.push_back(var);
    startTimeVariables[op] = var;
    startTimeLocations.push_back(firstNonBasicVariableColumn + var);
    ++var;
  }

  // one column for each parameter (1,S,T), and for all operations
  nColumns = nParameters + nonBasicVariables.size();

  // Helper to grow both the tableau and the implicit column vector.
  auto addRow = [&]() -> SmallVector<int> & {
    implicitBasicVariableColumnVector.push_back(0);
    return tableau.emplace_back(nColumns, 0);
  };

  // Set up the objective rows.
  nObjectives = 0;
  bool hasMoreObjectives;
  do {
    auto &objRowVec = addRow();
    hasMoreObjectives = fillObjectiveRow(objRowVec, nObjectives);
    ++nObjectives;
  } while (hasMoreObjectives);

  // Now set up rows/constraints for the dependences.
  for (auto *op : prob.getOperations()) {
    for (auto &dep : prob.getDependences(op)) {
      auto &consRowVec = addRow();
      fillConstraintRow(consRowVec, dep);
      basicVariables.push_back(var);
      ++var;
    }
  }
  for (auto &dep : additionalConstraints) {
    auto &consRowVec = addRow();
    fillAdditionalConstraintRow(consRowVec, dep);
    basicVariables.push_back(var);
    ++var;
  }

  // one row per objective + one row per dependence
  nRows = tableau.size();
}

void SimplexSchedulerBase::resetTableau() {
  tableau.clear();
  implicitBasicVariableColumnVector.clear();
  nonBasicVariables.clear();
  basicVariables.clear();
  startTimeVariables.clear();
  startTimeLocations.clear();
  frozenVariables.clear();
  parameterS = 0;
  nRows = 0;
  nColumns = 0;
  nObjectives = 0;
}

int SimplexSchedulerBase::getParametricConstant(unsigned row) {
  auto &rowVec = tableau[row];
  // Compute the dot-product ~B[row] * u between the constant matrix and the
  // parameter vector.
  return rowVec[parameter1Column] + rowVec[parameterSColumn] * parameterS +
         rowVec[parameterTColumn] * parameterT;
}

SmallVector<int> SimplexSchedulerBase::getObjectiveVector(unsigned column) {
  SmallVector<int> objVec;
  // Extract the column vector C^T[column] from the cost matrix.
  for (unsigned obj = 0; obj < nObjectives; ++obj)
    objVec.push_back(tableau[obj][column]);
  return objVec;
}

std::optional<unsigned> SimplexSchedulerBase::findDualPivotRow() {
  // Find the first row in which the parametric constant is negative.
  for (unsigned row = firstConstraintRow; row < nRows; ++row)
    if (getParametricConstant(row) < 0)
      return row;

  return std::nullopt;
}

std::optional<unsigned>
SimplexSchedulerBase::findDualPivotColumn(unsigned pivotRow,
                                          bool allowPositive) {
  SmallVector<int> maxQuot(nObjectives, std::numeric_limits<int>::min());
  std::optional<unsigned> pivotCol;

  // Look for non-zero entries in the constraint matrix (~A part of the
  // tableau). If multiple candidates exist, take the one corresponding to the
  // lexicographical maximum (over the objective rows) of the quotients:
  //   tableau[<objective row>][col] / pivotCand
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;

    int pivotCand = tableau[pivotRow][col];
    // Only negative candidates bring us closer to the optimal solution.
    // However, when freezing variables to a certain value, we accept that the
    // value of the objective function degrades.
    if (pivotCand < 0 || (allowPositive && pivotCand > 0)) {
      // The constraint matrix has only {-1, 0, 1} entries by construction.
      assert(pivotCand * pivotCand == 1);

      SmallVector<int> quot;
      for (unsigned obj = 0; obj < nObjectives; ++obj)
        quot.push_back(tableau[obj][col] / pivotCand);

      if (std::lexicographical_compare(maxQuot.begin(), maxQuot.end(),
                                       quot.begin(), quot.end())) {
        maxQuot = quot;
        pivotCol = col;
      }
    }
  }

  return pivotCol;
}

std::optional<unsigned> SimplexSchedulerBase::findPrimalPivotColumn() {
  // Find the first lexico-negative column in the cost matrix.
  SmallVector<int> zeroVec(nObjectives, 0);
  for (unsigned col = firstNonBasicVariableColumn; col < nColumns; ++col) {
    if (frozenVariables.count(
            nonBasicVariables[col - firstNonBasicVariableColumn]))
      continue;

    SmallVector<int> objVec = getObjectiveVector(col);
    if (std::lexicographical_compare(objVec.begin(), objVec.end(),
                                     zeroVec.begin(), zeroVec.end()))
      return col;
  }

  return std::nullopt;
}

std::optional<unsigned>
SimplexSchedulerBase::findPrimalPivotRow(unsigned pivotColumn) {
  int minQuot = std::numeric_limits<int>::max();
  std::optional<unsigned> pivotRow;

  // Look for positive entries in the constraint matrix (~A part of the
  // tableau). If multiple candidates exist, take the one corresponding to the
  // minimum of the quotient:
  //   parametricConstant(row) / pivotCand
  for (unsigned row = firstConstraintRow; row < nRows; ++row) {
    int pivotCand = tableau[row][pivotColumn];
    if (pivotCand > 0) {
      // The constraint matrix has only {-1, 0, 1} entries by construction.
      assert(pivotCand == 1);
      int quot = getParametricConstant(row) / pivotCand;
      if (quot < minQuot) {
        minQuot = quot;
        pivotRow = row;
      }
    }
  }

  return pivotRow;
}

void SimplexSchedulerBase::multiplyRow(unsigned row, int factor) {
  assert(factor != 0);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[row][col] *= factor;
  // Also multiply the corresponding entry in the temporary column vector.
  implicitBasicVariableColumnVector[row] *= factor;
}

void SimplexSchedulerBase::addMultipleOfRow(unsigned sourceRow, int factor,
                                            unsigned targetRow) {
  assert(factor != 0 && sourceRow != targetRow);
  for (unsigned col = 0; col < nColumns; ++col)
    tableau[targetRow][col] += tableau[sourceRow][col] * factor;
  // Again, perform row operation on the temporary column vector as well.
  implicitBasicVariableColumnVector[targetRow] +=
      implicitBasicVariableColumnVector[sourceRow] * factor;
}

/// The pivot operation applies elementary row operations to the tableau in
/// order to make the \p pivotColumn (corresponding to a non-basic variable) a
/// unit vector (only the \p pivotRow'th entry is 1). Then, a basis exchange is
/// performed: the non-basic variable is swapped with the basic variable
/// associated with the pivot row.
void SimplexSchedulerBase::pivot(unsigned pivotRow, unsigned pivotColumn) {
  // The implicit columns are part of an identity matrix.
  implicitBasicVariableColumnVector[pivotRow] = 1;

  int pivotElem = tableau[pivotRow][pivotColumn];
  // The constraint matrix has only {-1, 0, 1} entries by construction.
  assert(pivotElem * pivotElem == 1);
  // Make `tableau[pivotRow][pivotColumn]` := 1
  multiplyRow(pivotRow, 1 / pivotElem);

  for (unsigned row = 0; row < nRows; ++row) {
    if (row == pivotRow)
      continue;

    int elem = tableau[row][pivotColumn];
    if (elem == 0)
      continue; // nothing to do

    // Make `tableau[row][pivotColumn]` := 0.
    addMultipleOfRow(pivotRow, -elem, row);
  }

  // Swap the pivot column with the implicitly constructed column vector.
  // We really only need to copy in one direction here, as the former pivot
  // column is a unit vector, which is not stored explicitly.
  for (unsigned row = 0; row < nRows; ++row) {
    tableau[row][pivotColumn] = implicitBasicVariableColumnVector[row];
    implicitBasicVariableColumnVector[row] = 0; // Reset for next pivot step.
  }

  // Look up numeric IDs of variables involved in this pivot operation.
  unsigned &nonBasicVar =
      nonBasicVariables[pivotColumn - firstNonBasicVariableColumn];
  unsigned &basicVar = basicVariables[pivotRow - firstConstraintRow];

  // Keep track of where start time variables are; ignore slack variables.
  if (nonBasicVar < startTimeLocations.size())
    startTimeLocations[nonBasicVar] = -pivotRow; // ...going into basis.
  if (basicVar < startTimeLocations.size())
    startTimeLocations[basicVar] = pivotColumn; // ...going out of basis.

  // Record the swap in the variable lists.
  std::swap(nonBasicVar, basicVar);
}

LogicalResult SimplexSchedulerBase::solveTableau() {
  // "Solving" technically means perfoming dual pivot steps until primal
  // feasibility is reached, i.e. the parametric constants in all constraint
  // rows are non-negative.
  while (auto pivotRow = findDualPivotRow()) {
    // Look for pivot elements.
    if (auto pivotCol = findDualPivotColumn(*pivotRow)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // llvm::errs() << "failed to find pivot column\n";
    // dumpTableau();
    // If we did not find a pivot column, then the entire row contained only
    // positive entries, and the problem is in principle infeasible. However, if
    // the entry in the `parameterTColumn` is positive, we can try to make the
    // LP feasible again by increasing the II.
    int entry1Col = tableau[*pivotRow][parameter1Column];
    int entrySCol = tableau[*pivotRow][parameterSColumn];
    int entryTCol = tableau[*pivotRow][parameterTColumn];
    if (entryTCol > 0 && allowIIIncreaseDuringSolve) {
      // Solve `effConst + T * entryTCol >= 0` for the smallest feasible T.
      // The S contribution must be folded into the constant: during
      // `scheduleAt`, `parameterS` is temporarily nonzero, so the row's
      // negative parametric constant may stem from the S term while
      // `entry1Col` alone is non-negative (the original formula, following
      // the paper's S = 0 setting, missed this and asserted).
      int effConst = entry1Col + parameterS * entrySCol;
      // The row's parametric constant `effConst + T * entryTCol` is
      // negative (it was a valid pivot row) and entryTCol > 0, so:
      assert(effConst < 0);
      int newParameterT = (-effConst - 1) / entryTCol + 1;
      if (newParameterT > parameterT) {
        parameterT = newParameterT;
        LLVM_DEBUG(dbgs() << "Increased II to " << parameterT << '\n');
        continue;
      }
    }

    // Otherwise, the linear program is infeasible.
    return failure();
  }

  // Optimal solution found!
  return success();
}

LogicalResult SimplexSchedulerBase::restoreDualFeasibility() {
  // Dual feasibility requires that all columns in the cost matrix are
  // non-lexico-negative. This property may be violated after changing the order
  // of the objective rows, and can be restored by performing primal pivot
  // steps.
  while (auto pivotCol = findPrimalPivotColumn()) {
    // Look for pivot elements.
    if (auto pivotRow = findPrimalPivotRow(*pivotCol)) {
      pivot(*pivotRow, *pivotCol);
      continue;
    }

    // Otherwise, the linear program is unbounded.
    return failure();
  }

  // Optimal solution found!
  return success();
}

bool SimplexSchedulerBase::isInBasis(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());
  int loc = startTimeLocations[startTimeVariable];
  if (-loc >= (int)firstConstraintRow)
    return true;
  if (loc >= (int)firstNonBasicVariableColumn)
    return false;
  llvm_unreachable("Invalid variable location");
}

unsigned SimplexSchedulerBase::freeze(unsigned startTimeVariable,
                                      unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  // Mark variable.
  frozenVariables[startTimeVariable] = timeStep;

  if (!isInBasis(startTimeVariable))
    // That's all for non-basic variables.
    return startTimeLocations[startTimeVariable];

  // We need to pivot this variable one out of basis.
  unsigned pivotRow = -startTimeLocations[startTimeVariable];

  // Here, positive pivot elements can be considered as well, hence finding a
  // suitable column should not fail.
  auto pivotCol = findDualPivotColumn(pivotRow, /* allowPositive= */ true);
  assert(pivotCol);

  // Perform the exchange.
  pivot(pivotRow, *pivotCol);

  // `startTimeVariable` is now represented by `pivotCol`.
  return *pivotCol;
}

void SimplexSchedulerBase::translate(unsigned column, int factor1, int factorS,
                                     int factorT) {
  for (unsigned row = 0; row < nRows; ++row) {
    auto &rowVec = tableau[row];
    int elem = rowVec[column];
    if (elem == 0)
      continue;

    rowVec[parameter1Column] += -elem * factor1;
    rowVec[parameterSColumn] += -elem * factorS;
    rowVec[parameterTColumn] += -elem * factorT;
  }
}

LogicalResult SimplexSchedulerBase::scheduleAt(unsigned startTimeVariable,
                                               unsigned timeStep) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(!frozenVariables.count(startTimeVariable));

  // auto solvedTest = solveTableau();
  // if (succeeded(solvedTest))
  //   llvm::errs() << "solved correctly before fixing\n";

  // Freeze variable and translate its column by parameter S.
  unsigned frozenCol = freeze(startTimeVariable, timeStep);
  translate(frozenCol, /* factor1= */ 0, /* factorS= */ 1, /* factorT= */ 0);

  // Temporarily set S to the desired value, and attempt to solve.
  parameterS = timeStep;
  auto solved = solveTableau();
  parameterS = 0;

  if (failed(solved)) {
    // llvm::errs() << "here failed\n";
    // The LP is infeasible with the new constraint. We could try other values
    // for S, but for now, we just roll back and signal failure to the driver.
    translate(frozenCol, /* factor1= */ 0, /* factorS= */ -1, /* factorT= */ 0);
    frozenVariables.erase(startTimeVariable);
    auto solvedAfterRollback = solveTableau();
    assert(succeeded(solvedAfterRollback));
    (void)solvedAfterRollback;
    return failure();
  }

  // Translate S by the other parameter(s). This means setting `factor1` to
  // `timeStep`.
  //
  // This translation does not change the values of the parametric constants,
  // hence we do not need to solve the tableau again.
  //
  // Note: I added a negation of the factors here, which is not mentioned in the
  // paper's text, but apparently used in the example. Without it, the intended
  // effect, i.e. making the S-column all-zero again, is not achieved.
  //
  // Note 2: For cyclic problems, the paper suggested to perform a modulo
  // decomposition: S = `factor1` + `factorT` * T, with `factor1` < T.
  // However, this makes the value baked into the tableau dependent on
  // `parameterT`, and it is unclear to me how to update it correctly when
  // changing the II. I found it much more robust to fix the operations to
  // absolute time steps, and manually shift them by the appropriate amount
  // whenever the II is incremented (cf. adding `phiJ`, `phiN` in the modulo
  // scheduler's `scheduleOperation` method).
  translate(parameterSColumn, /* factor1= */ -timeStep, /* factorS= */ 1,
            /* factorT= */ 0);

  // dumpTableau();
  return success();
}

void SimplexSchedulerBase::moveBy(unsigned startTimeVariable, unsigned amount) {
  assert(startTimeVariable < startTimeLocations.size());
  assert(frozenVariables.count(startTimeVariable));

  // Bookkeeping.
  frozenVariables[startTimeVariable] += amount;

  // Moving an already frozen variable means translating it by the desired
  // amount, and solving the tableau to restore primal feasibility...
  translate(startTimeLocations[startTimeVariable], /* factor1= */ amount,
            /* factorS= */ 0, /* factorT= */ 0);

  // ... however, we typically batch-move multiple operations (otherwise, the
  // tableau may become infeasible on intermediate steps), so actually defer
  // solving to the caller.
}

unsigned SimplexSchedulerBase::getStartTime(unsigned startTimeVariable) {
  assert(startTimeVariable < startTimeLocations.size());

  if (!isInBasis(startTimeVariable))
    // Non-basic variables that are not already fixed to a specific time step
    // are 0 at the end of the simplex algorithm.
    return frozenVariables.lookup(startTimeVariable);

  // For the variables currently in basis, we look up the solution in the
  // tableau.
  return getParametricConstant(-startTimeLocations[startTimeVariable]);
}

void SimplexSchedulerBase::dumpTableau() {
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  for (unsigned i = 0; i < nRows; ++i) {
    if (i == firstConstraintRow) {
      for (unsigned j = 0; j < nColumns; ++j) {
        if (j == firstNonBasicVariableColumn)
          dbgs() << "-+";
        dbgs() << "----";
      }
      dbgs() << '\n';
    }
    for (unsigned j = 0; j < nColumns; ++j) {
      if (j == firstNonBasicVariableColumn)
        dbgs() << " |";
      dbgs() << format(" %3d", tableau[i][j]);
    }
    if (i >= firstConstraintRow)
      dbgs() << format(" |< %2d", basicVariables[i - firstConstraintRow]);
    dbgs() << '\n';
  }
  for (unsigned j = 0; j < nColumns; ++j)
    dbgs() << "====";
  dbgs() << "==\n";
  dbgs() << format(" %3d %3d %3d | ", 1, parameterS, parameterT);
  for (unsigned j = firstNonBasicVariableColumn; j < nColumns; ++j)
    dbgs() << format(" %2d^",
                     nonBasicVariables[j - firstNonBasicVariableColumn]);
  dbgs() << '\n';
}

//===----------------------------------------------------------------------===//
// SimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult SimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Optimal solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// CyclicSimplexScheduler
//===----------------------------------------------------------------------===//

bool chainBroken(CyclicProblem &prob, Operation *src, Operation *dst) {
  auto opr = *prob.getLinkedOperatorType(src);
  if (src == dst && prob.getLatency(opr) == 0)
    return false;

  if (src == dst)
    return true;

  if (prob.getLatency(opr) == 0) {
    for (auto dep : prob.getDependences(src)) {
      if (dep.isAuxiliary())
        continue;
      if (chainBroken(prob, dep.getSource(), dst))
        return true;
    }
  }

  return false;
}

// bool isStore(Operation *op) {
//   return isa<loopschedule::LoopScheduleStoreOp,
//   mlir::affine::AffineWriteOpInterface, loopschedule::StoreInterface>(op);
// }

void CyclicSimplexScheduler::fillConstraintRow(SmallVector<int> &row,
                                               Problem::Dependence dep) {
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  // llvm::errs() << "source: ";
  // dep.getSource()->dump();
  // llvm::errs() << "dest: ";
  // dep.getDestination()->dump();
  if (auto dist = prob.getDistance(dep)) {
    // llvm::errs() << "dist: " << dist.value() << "\n";
    // the latency of the last op in a recurrence must be at least 1
    if (latency == 0) {
      // // src->dump();
      // // dst->dump();
      // if (!chainBroken(prob, src, dst)) {
      // llvm::errs() << "not chain broken\n";
      // if (!isStore(src))
      // if (dep.isSrcStore()) {
      // }
      if (!prob.isSrcStore(dep)) {
        latency = 1;
      }
      // llvm::errs() << "=======================\n";
    }
  }
  // llvm::errs() << "=======================\n";
  row[parameter1Column] = -latency; // note the negation
  if (src != dst) { // note that these coefficients just zero out in self-arcs.
    row[startTimeLocations[startTimeVariables[src]]] = 1;
    row[startTimeLocations[startTimeVariables[dst]]] = -1;
  }
  if (auto dist = prob.getDistance(dep))
    row[parameterTColumn] = *dist;
}

LogicalResult CyclicSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  parameterT = 1;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Optimal solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// SharedOperatorsSimplexScheduler
//===----------------------------------------------------------------------===//

static bool isLimited(Operation *op, SharedOperatorsProblem &prob) {
  auto maybeRsrcs = prob.getLinkedResourceTypes(op);
  if (!maybeRsrcs)
    return false;
  return llvm::any_of(*maybeRsrcs, [&](Problem::ResourceType rsrc) {
    return prob.getLimit(rsrc).value_or(0) > 0;
  });
}

LogicalResult SharedOperatorsSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "After solving resource-free problem:\n"; dumpTableau());

  // The *heuristic* part of this scheduler starts here:
  // We will now *choose* start times for operations using a shared operator
  // type, in a way that respects the allocation limits, and consecutively solve
  // the LP with these added constraints. The individual LPs are still solved to
  // optimality (meaning: the start times of the "last" operation is still
  // optimal w.r.t. the already fixed operations), however the heuristic choice
  // means we cannot guarantee the optimality for the overall problem.

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Build a priority list of the limited operations.
  //
  // We sort by the resource-free start times to produce a topological order of
  // the operations. Better priority functions are known, but require computing
  // additional properties, e.g. ASAP and ALAP times for mobility, or graph
  // analysis for height. Assigning operators (=resources) in this order at
  // least ensures that the (acyclic!) problem remains feasible throughout the
  // process.
  //
  // TODO: Implement more sophisticated priority function.
  std::stable_sort(limitedOps.begin(), limitedOps.end(),
                   [&](Operation *a, Operation *b) {
                     return getStartTime(startTimeVariables[a]) <
                            getStartTime(startTimeVariables[b]);
                   });

  // Store the number of operations using a resource type in a particular time
  // step.
  SmallDenseMap<Problem::ResourceType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    assert(maybeRsrcs && "Limited operation must have linked resource types");

    auto &rsrcs = *maybeRsrcs;

    // Find the first time step (beginning at the current start time in the
    // partial schedule) in which an operator instance is available.
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = getStartTime(startTimeVar);
    auto canSchedule = [&](unsigned candTime) {
      for (auto rsrc : rsrcs) {
        auto extraLimit = prob.getLimit(rsrc);
        if (extraLimit.has_value() &&
            reservationTable[rsrc].lookup(candTime) == *extraLimit)
          return false;
      }
      return true;
    };
    while (!canSchedule(candTime))
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the resource uses.
    for (auto rsrc : rsrcs)
      ++reservationTable[rsrc][candTime];

    LLVM_DEBUG(dbgs() << "After scheduling " << startTimeVar
                      << " to t=" << candTime << ":\n";
               dumpTableau());
  }

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Feasible solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloSimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult ModuloSimplexScheduler::checkLastOp() {
  auto *contOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return contOp->emitError("problem does not include last operation");

  // Determine which operations have no outgoing *intra*-iteration dependences.
  auto &ops = prob.getOperations();
  DenseSet<Operation *> sinks(ops.begin(), ops.end());
  for (auto *op : ops)
    for (auto &dep : prob.getDependences(op))
      if (prob.getDistance(dep).value_or(0) == 0)
        sinks.erase(dep.getSource());

  if (!sinks.contains(lastOp))
    return contOp->emitError("last operation is not a sink");
  if (sinks.size() > 1)
    return contOp->emitError("multiple sinks detected");

  return success();
}

LogicalResult ModuloSimplexScheduler::MRT::enter(Operation *op,
                                                 unsigned timeStep) {

  DenseSet<Problem::ResourceType> usedRsrcs;
  SmallDenseMap<Problem::ResourceType, TableType> tempTables(tables);

  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  assert(maybeRsrcs && "Operation must have linked resource types");

  auto &rsrcs = *maybeRsrcs;

  // llvm::errs() << "here1\n";
  unsigned slot = timeStep % sched.parameterT;
  // llvm::errs() << "slot = " << slot << "\n";
  for (auto rsrc : rsrcs) {
    // llvm::errs() << "resource type\n";
    auto lim = sched.prob.getLimit(rsrc);
    assert(lim.has_value() && "resource type does not have limit");
    assert(lim.value() > 0);

    auto &revTab = reverseTables[rsrc];
    assert(!revTab.count(op));

    auto &tempCell = tempTables[rsrc][slot];
    if (tempCell.size() >= lim)
      return failure();
    tempCell.insert(op);
    usedRsrcs.insert(rsrc);
  }

  for (auto rsrc : usedRsrcs) {
    // llvm::errs() << "usedRsrcs\n";
    auto &revTab = reverseTables[rsrc];
    assert(!revTab.count(op));
    auto &cell = tables[rsrc][slot];
    auto lim = sched.prob.getLimit(rsrc);
    assert(cell.size() < lim);
    cell.insert(op);
    revTab[op] = slot;
  }

  return success();
}

void ModuloSimplexScheduler::MRT::release(Operation *op) {
  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  assert(maybeRsrcs && "Operation must have linked resource types");

  auto &rsrcs = *maybeRsrcs;
  for (auto rsrc : rsrcs) {
    auto &revTab = reverseTables[rsrc];
    auto it = revTab.find(op);
    assert(it != revTab.end());
    tables[rsrc][it->second].erase(op);
    revTab.erase(it);
  }
}

bool ModuloSimplexScheduler::fillObjectiveRow(SmallVector<int> &row,
                                              unsigned obj) {
  switch (obj) {
  case OBJ_LATENCY:
    // Minimize start time of user-specified last operation.
    row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
    return true;
  case OBJ_AXAP:
    // Minimize sum of start times of all-but-the-last operation.
    for (auto *op : getProblem().getOperations())
      if (op != lastOp)
        row[startTimeLocations[startTimeVariables[op]]] = 1;
    return false;
  default:
    llvm_unreachable("Unsupported objective requested");
  }
}

void ModuloSimplexScheduler::updateMargins() {
  // Assumption: current secondary objective is "ASAP".
  // Negate the objective row once to effectively maximize the sum of start
  // times, which yields the "ALAP" times after solving the tableau. Then,
  // negate it again to restore the "ASAP" objective, and store these times as
  // well.
  for (auto *axapTimes : {&alapTimes, &asapTimes}) {
    multiplyRow(OBJ_AXAP, -1);
    // This should not fail for a feasible tableau.
    auto dualFeasRestored = restoreDualFeasibility();
    auto solved = solveTableau();
    assert(succeeded(dualFeasRestored) && succeeded(solved));
    (void)dualFeasRestored, (void)solved;

    for (unsigned stv = 0; stv < startTimeLocations.size(); ++stv)
      (*axapTimes)[stv] = getStartTime(stv);
  }
}

LogicalResult ModuloSimplexScheduler::scheduleOperation(Operation *n) {
  auto oprN = *prob.getLinkedOperatorType(n);
  unsigned stvN = startTimeVariables[n];

  // Get current state of the LP. We'll try to schedule at its current time step
  // in the partial solution, and the II-1 following time steps. Scheduling the
  // op to a later time step may increase the overall latency, however, as long
  // as the solution is still feasible, we prefer that over incrementing the II
  // to resolve resource conflicts.
  unsigned stN = getStartTime(stvN);
  unsigned ubN = stN + parameterT - 1;

  LLVM_DEBUG(dbgs() << "Attempting to schedule in [" << stN << ", " << ubN
                    << "]: " << *n << '\n');

  for (unsigned ct = stN; ct <= ubN; ++ct)
    if (succeeded(mrt.enter(n, ct))) {
      // llvm::errs() << "entered, schedule at " << ct << "\n";
      auto fixedN = scheduleAt(stvN, ct);
      if (succeeded(fixedN)) {
        LLVM_DEBUG(dbgs() << "Success at t=" << ct << " " << *n << '\n');
        return success();
      }
      // Problem became infeasible with `n` at `ct`, roll back the MRT
      // assignment. Also, no later time can be feasible, so stop the search
      // here.
      mrt.release(n);
      break;
    }

  // As a last resort, increase II to make room for the op. De Dinechin's
  // Theorem 1 lays out conditions/guidelines to transform the current partial
  // schedule for II to a valid one for a larger II'.

  LLVM_DEBUG(dbgs() << "Incrementing II to " << (parameterT + 1)
                    << " to resolve resource conflict for " << *n << '\n');

  // Note that the approach below is much simpler than in the paper
  // because of the fully-pipelined operators. In our case, it's always
  // sufficient to increment the II by one.

  // llvm::errs() << "parameterT: " << parameterT << "\n";

  // Decompose start time.
  unsigned phiN = stN / parameterT;
  unsigned tauN = stN % parameterT;

  // Keep track whether the following moves free at least one operator
  // instance in the slot desired by the current op - then it can stay there.
  unsigned deltaN = 1;

  // We're going to revisit the current partial schedule.
  SmallVector<Operation *> moved;
  for (Operation *j : scheduled) {
    auto oprJ = *prob.getLinkedOperatorType(j);
    unsigned stvJ = startTimeVariables[j];
    unsigned stJ = getStartTime(stvJ);
    unsigned phiJ = stJ / parameterT;
    unsigned tauJ = stJ % parameterT;
    unsigned deltaJ = 0;

    if (oprN == oprJ) {
      // To actually resolve the resource conflicts, we will move operations
      // that are "preceded" (cf. de Dinechin's ≺ relation) one slot to the
      // right.
      if (tauN < tauJ || (tauN == tauJ && phiN > phiJ) ||
          (tauN == tauJ && phiN == phiJ && stvN < stvJ)) {
        // TODO: Replace the last condition with a proper graph analysis.

        deltaJ = 1;
        moved.push_back(j);
        if (tauN == tauJ)
          deltaN = 0;
      }
    }

    // j->dump();
    // llvm::errs() << "phiJ: " << phiJ << "\n";
    // llvm::errs() << "deltaJ: " << deltaJ << "\n";
    // llvm::errs() << "========================\n";

    // Move operation.
    //
    // In order to keep the op in its current MRT slot `tauJ` after incrementing
    // the II, we add `phiJ`:
    //   stJ + phiJ = (phiJ * parameterT + tauJ) + phiJ
    //              = phiJ * (parameterT + 1) + tauJ
    //
    // Shifting an additional `deltaJ` time steps then moves the op to a
    // different MRT slot, in order to make room for the operation that caused
    // the resource conflict.
    moveBy(stvJ, phiJ + deltaJ);
  }

  // Finally, increment the II and solve to apply the moves.
  ++parameterT;
  auto solved = solveTableau();
  if (failed(solved)) {
    // Same fragility as the chaining variant: the heuristic move set can
    // leave the tableau infeasible. Propagate instead of asserting.
    LLVM_DEBUG(dbgs() << "Tableau infeasible after II increment to "
                      << parameterT << "; giving up on this attempt\n");
    return failure();
  }

  // Re-enter moved operations into their new slots.
  for (auto *m : moved)
    mrt.release(m);
  for (auto *m : moved) {
    auto enteredM = mrt.enter(m, getStartTime(startTimeVariables[m]));
    assert(succeeded(enteredM));
    (void)enteredM;
  }

  // Finally, schedule the operation. Again, adding `phiN` accounts for the
  // implicit shift caused by incrementing the II.
  // llvm::errs() << "II shift schedule at " << stN + phiN + deltaN << "\n";
  // llvm::errs() << "stN: " << stN << "\n";
  // llvm::errs() << "phiN: " << phiN << "\n";
  // llvm::errs() << "deltaN: " << deltaN << "\n";
  auto fixedN = scheduleAt(stvN, stN + phiN + deltaN);
  auto enteredN = mrt.enter(n, tauN + deltaN);
  // n->dump();
  assert(succeeded(enteredN));
  assert(succeeded(fixedN));
  (void)fixedN, (void)enteredN;
  return success();
}

unsigned ModuloSimplexScheduler::computeResMinII() {
  unsigned resMinII = 1;
  SmallDenseMap<Problem::ResourceType, unsigned> uses;
  for (auto *op : prob.getOperations()) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    if (!maybeRsrcs)
      continue;

    for (auto rsrc : *maybeRsrcs) {
      if (prob.getLimit(rsrc).value_or(0) > 0)
        ++uses[rsrc];
    }
  }

  for (auto pair : uses) {
    resMinII = std::max(
        resMinII, (unsigned)ceil(pair.second / *prob.getLimit(pair.first)));
  }

  // llvm::errs() << "resMinII = " << resMinII << "\n";
  return resMinII;
}

LogicalResult ModuloSimplexScheduler::schedule() {
  if (failed(checkLastOp()))
    return failure();

  parameterS = 0;
  parameterT = computeResMinII();
  LLVM_DEBUG(dbgs() << "ResMinII = " << parameterT << "\n");
  buildTableau();
  asapTimes.resize(startTimeLocations.size());
  alapTimes.resize(startTimeLocations.size());

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  for (auto *op : ops)
    if (isLimited(op, prob))
      unscheduled.push_back(op);

  // Main loop: Iteratively fix limited operations to time steps.
  while (!unscheduled.empty()) {
    // Update ASAP/ALAP times.
    updateMargins();

    // Heuristically (here: least amount of slack) pick the next operation to
    // schedule.
    auto *opIt =
        std::min_element(unscheduled.begin(), unscheduled.end(),
                         [&](Operation *opA, Operation *opB) {
                           auto stvA = startTimeVariables[opA];
                           auto stvB = startTimeVariables[opB];
                           auto slackA = alapTimes[stvA] - asapTimes[stvA];
                           auto slackB = alapTimes[stvB] - asapTimes[stvB];
                           return slackA < slackB;
                         });
    Operation *op = *opIt;
    unscheduled.erase(opIt);

    if (failed(scheduleOperation(op)))
      return failure();
    scheduled.push_back(op);
  }

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingSimplexScheduler
//===----------------------------------------------------------------------===//

void ChainingSimplexScheduler::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  fillConstraintRow(row, dep);
  // One _extra_ time step breaks the chain (note that the latency is negative
  // in the tableau).
  row[parameter1Column] -= 1;
}

LogicalResult ChainingSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
    return failure();

  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Optimal solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn)); // Problem is known to be acyclic at this point.
  (void)filledIn;

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingSharedOperatorsSimplexScheduler
//===----------------------------------------------------------------------===//

void ChainingSharedOperatorsSimplexScheduler::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  fillConstraintRow(row, dep);
  // One _extra_ time step breaks the chain (note that the latency is negative
  // in the tableau).
  row[parameter1Column] -= 1;
}

LogicalResult ChainingSharedOperatorsSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
    return failure();

  parameterS = 0;
  parameterT = 0;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "After solving resource-free problem:\n"; dumpTableau());

  // The *heuristic* part of this scheduler starts here:
  // We will now *choose* start times for operations using a shared operator
  // type, in a way that respects the allocation limits, and consecutively solve
  // the LP with these added constraints. The individual LPs are still solved to
  // optimality (meaning: the start times of the "last" operation is still
  // optimal w.r.t. the already fixed operations), however the heuristic choice
  // means we cannot guarantee the optimality for the overall problem.

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  SmallVector<Operation *> limitedOps;
  for (auto *op : ops)
    if (isLimited(op, prob))
      limitedOps.push_back(op);

  // Build a priority list of the limited operations.
  //
  // We sort by the resource-free start times to produce a topological order of
  // the operations. Better priority functions are known, but require computing
  // additional properties, e.g. ASAP and ALAP times for mobility, or graph
  // analysis for height. Assigning operators (=resources) in this order at
  // least ensures that the (acyclic!) problem remains feasible throughout the
  // process.
  //
  // TODO: Implement more sophisticated priority function.
  std::stable_sort(limitedOps.begin(), limitedOps.end(),
                   [&](Operation *a, Operation *b) {
                     return getStartTime(startTimeVariables[a]) <
                            getStartTime(startTimeVariables[b]);
                   });

  // Store the number of operations using an operator type in a particular time
  // step.
  SmallDenseMap<Problem::ResourceType, SmallDenseMap<unsigned, unsigned>>
      reservationTable;

  for (auto *op : limitedOps) {
    auto maybeRsrcs = prob.getLinkedResourceTypes(op);
    auto resourceTypes = maybeRsrcs.has_value()
                             ? *maybeRsrcs
                             : SmallVector<Problem::ResourceType>{};

    // Find the first time step (beginning at the current start time in the
    // partial schedule) in which an operator instance is available.
    unsigned startTimeVar = startTimeVariables[op];
    unsigned candTime = getStartTime(startTimeVar);
    auto canSchedule = [&](unsigned candTime) {
      for (auto rsrc : resourceTypes) {
        auto extraLimit = prob.getLimit(rsrc);
        if (extraLimit.has_value() &&
            reservationTable[rsrc].lookup(candTime) == *extraLimit)
          return false;
      }
      return true;
    };
    while (!canSchedule(candTime))
      ++candTime;

    // Fix the start time. As explained above, this cannot make the problem
    // infeasible.
    auto fixed = scheduleAt(startTimeVar, candTime);
    assert(succeeded(fixed));
    (void)fixed;

    // Record the resource uses.
    for (auto rsrc : resourceTypes)
      ++reservationTable[rsrc][candTime];

    LLVM_DEBUG(dbgs() << "After scheduling " << startTimeVar
                      << " to t=" << candTime << ":\n";
               dumpTableau());
  }

  assert(parameterT == 0);
  LLVM_DEBUG(
      dbgs() << "Final tableau:\n"; dumpTableau();
      dbgs() << "Feasible solution found with start time of last operation = "
             << -getParametricConstant(0) << '\n');

  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn)); // Problem is known to be acyclic at this point.
  (void)filledIn;

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingCyclicSimplexScheduler
//===----------------------------------------------------------------------===//

void ChainingCyclicSimplexScheduler::fillConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {

  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();
  unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
  if (auto dist = prob.getDistance(dep)) {
    // the latency of the last op in a recurrence must be at least 1
    if (latency == 0) {
      if (!prob.isSrcStore(dep)) {
        latency = 1;
      }
    }
  }
  row[parameter1Column] = -latency; // note the negation
  if (src != dst) { // note that these coefficients just zero out in self-arcs.
    row[startTimeLocations[startTimeVariables[src]]] = 1;
    row[startTimeLocations[startTimeVariables[dst]]] = -1;
  }
  if (auto dist = prob.getDistance(dep))
    row[parameterTColumn] = *dist;
}

void ChainingCyclicSimplexScheduler::fillAdditionalConstraintRow(
    SmallVector<int> &row, Problem::Dependence dep) {
  fillConstraintRow(row, dep);
  // One _extra_ time step breaks the chain (note that the latency is negative
  // in the tableau).
  row[parameter1Column] -= 1;
}

LogicalResult ChainingCyclicSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
    return failure();

  parameterS = 0;
  parameterT = 1;
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Optimal solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : prob.getOperations())
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn));
  (void)filledIn;

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingModuloSimplexScheduler
//===----------------------------------------------------------------------===//

LogicalResult ChainingModuloSimplexScheduler::checkLastOp() {
  auto *contOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return contOp->emitError("problem does not include last operation");

  // Determine which operations have no outgoing *intra*-iteration dependences.
  auto &ops = prob.getOperations();
  DenseSet<Operation *> sinks(ops.begin(), ops.end());
  for (auto *op : ops)
    for (auto &dep : prob.getDependences(op))
      if (prob.getDistance(dep).value_or(0) == 0)
        sinks.erase(dep.getSource());

  if (!sinks.contains(lastOp))
    return contOp->emitError("last operation is not a sink");
  if (sinks.size() > 1)
    return contOp->emitError("multiple sinks detected");

  return success();
}

LogicalResult ChainingModuloSimplexScheduler::MRT::enter(Operation *op,
                                                         unsigned timeStep) {

  DenseSet<Problem::ResourceType> usedRsrcs;
  SmallDenseMap<Problem::ResourceType, TableType> tempTables(tables);

  // llvm::errs() << "here1\n";
  unsigned slot = timeStep % sched.parameterT;
  // llvm::errs() << "slot = " << slot << "\n";
  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  if (maybeRsrcs.has_value()) {
    for (auto rsrc : maybeRsrcs.value()) {
      auto lim = sched.prob.getLimit(rsrc);
      assert(lim.has_value() && "resource type does not have limit");
      assert(lim.value() > 0);

      auto &revTab = reverseTables[rsrc];
      assert(!revTab.count(op));

      auto &tempCell = tempTables[rsrc][slot];
      if (tempCell.size() >= *lim)
        return failure();
      tempCell.insert(op);
      usedRsrcs.insert(rsrc);
    }
  }

  for (auto rsrc : usedRsrcs) {
    auto &revTab = reverseTables[rsrc];
    assert(!revTab.count(op));
    auto &cell = tables[rsrc][slot];
    auto lim = sched.prob.getLimit(rsrc);
    assert(cell.size() < lim);
    cell.insert(op);
    revTab[op] = slot;
  }

  return success();
}

void ChainingModuloSimplexScheduler::MRT::release(Operation *op) {
  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  if (maybeRsrcs.has_value()) {
    for (auto rsrc : *maybeRsrcs) {
      // llvm::errs() << "release resource\n";
      auto &revTab = reverseTables[rsrc];
      auto it = revTab.find(op);
      assert(it != revTab.end());
      tables[rsrc][it->second].erase(op);
      revTab.erase(it);
    }
  }
}

bool ChainingModuloSimplexScheduler::fillObjectiveRow(SmallVector<int> &row,
                                                      unsigned obj) {
  switch (obj) {
  case OBJ_LATENCY:
    // Minimize start time of user-specified last operation.
    row[startTimeLocations[startTimeVariables[lastOp]]] = 1;
    return true;
  case OBJ_AXAP:
    // Minimize sum of start times of all-but-the-last operation.
    for (auto *op : getProblem().getOperations())
      if (op != lastOp)
        row[startTimeLocations[startTimeVariables[op]]] = 1;
    return false;
  default:
    llvm_unreachable("Unsupported objective requested");
  }
}

void ChainingModuloSimplexScheduler::computeHeightPriorities() {
  heightPriority.clear();
  stableIndex.clear();
  auto &ops = prob.getOperations();
  for (auto [idx, op] : llvm::enumerate(ops)) {
    stableIndex[op] = idx;
    heightPriority[op] = 0;
  }
  // Bellman-Ford-style relaxation of
  //   height(src) >= height(dst) + latency(src) - II * distance(src, dst)
  // over all dependences. The initial LP being feasible at this II implies
  // every dependence cycle's total weight is non-positive, so this
  // converges within |ops| passes; the iteration cap is defensive.
  bool changed = true;
  for (unsigned iter = 0; changed && iter <= ops.size(); ++iter) {
    changed = false;
    for (auto *dst : ops) {
      int heightDst = heightPriority[dst];
      for (auto &dep : prob.getDependences(dst)) {
        Operation *srcOp = dep.getSource();
        int latency = 0;
        if (auto opr = prob.getLinkedOperatorType(srcOp))
          latency = (int)prob.getLatency(*opr).value_or(0);
        int distance = (int)prob.getDistance(dep).value_or(0);
        int cand = heightDst + latency - (int)parameterT * distance;
        auto it = heightPriority.find(srcOp);
        if (it != heightPriority.end() && cand > it->second) {
          it->second = cand;
          changed = true;
        }
      }
    }
  }
}

void ChainingModuloSimplexScheduler::MRT::gatherConflicts(
    Operation *op, unsigned timeStep, SmallVectorImpl<Operation *> &conflicts) {
  unsigned slot = timeStep % sched.parameterT;
  auto maybeRsrcs = sched.prob.getLinkedResourceTypes(op);
  if (!maybeRsrcs.has_value())
    return;
  DenseSet<Operation *> seen;
  for (auto rsrc : *maybeRsrcs) {
    auto lim = sched.prob.getLimit(rsrc);
    auto tabIt = tables.find(rsrc);
    if (tabIt == tables.end())
      continue;
    auto cellIt = tabIt->second.find(slot);
    if (cellIt == tabIt->second.end())
      continue;
    if (lim && cellIt->second.size() < *lim)
      continue; // room left for this resource; no conflict here
    for (auto *occupant : cellIt->second)
      if (seen.insert(occupant).second)
        conflicts.push_back(occupant);
  }
}

void ChainingModuloSimplexScheduler::unpin(Operation *op) {
  assert(pinnedOps.count(op));
  pinnedOps.erase(op);
  mrt.release(op);
  unscheduled.push_back(op);
  LLVM_DEBUG(dbgs() << "Unpinned: " << *op << '\n');
}

LogicalResult ChainingModuloSimplexScheduler::rebuildWithPins() {
  resetTableau();
  buildTableau();
  if (failed(solveTableau()))
    return failure();
  // Re-applying a subset of a previously feasible pin set stays feasible.
  for (auto &[op, time] : pinnedOps)
    if (failed(scheduleAt(startTimeVariables[op], time)))
      return failure();
  return success();
}

// Backtracking placement, following Rau's iterative modulo scheduling
// (MICRO-27, 1994, §3.1/3.4) with the LP adaptation of Canis et al.'s
// modulo SDC scheduler (FPL 2014, Alg. 1/2): try the op's II-wide window;
// if no slot is simultaneously MRT-free and LP-feasible, force-place it —
// evicting the MRT occupants at the chosen slot, and if the LP still
// refuses (a dependence conflict with pinned ops, e.g. an already-pinned
// consumer whose deadline the port-serialized accesses cannot meet),
// unpinning ALL other ops. Evicted ops return to the worklist; the
// `prevSched + 1` rule guarantees forward progress, and the caller's
// budget bounds total work before the II is incremented.
LogicalResult
ChainingModuloSimplexScheduler::scheduleOperation(Operation *n) {
  unsigned stvN = startTimeVariables[n];
  unsigned stN = getStartTime(stvN);

  LLVM_DEBUG(dbgs() << "Attempting to schedule in [" << stN << ", "
                    << stN + parameterT - 1 << "]: " << *n << '\n');

  // Happy path: a slot in the II-wide window that is both MRT-free and
  // LP-feasible. Unlike the previous implementation, an LP-infeasible slot
  // does not abort the search — later slots can still be feasible.
  for (unsigned ct = stN; ct < stN + parameterT; ++ct) {
    if (failed(mrt.enter(n, ct)))
      continue;
    if (succeeded(scheduleAt(stvN, ct))) {
      pinnedOps[n] = ct;
      prevSched[n] = ct;
      LLVM_DEBUG(dbgs() << "Success at t=" << ct << " " << *n << '\n');
      return success();
    }
    mrt.release(n);
  }

  // Forced placement. Pinning at the LP's own current solution value is
  // always feasible w.r.t. the pinned set, so `stN` plays the role of
  // Canis's resource-free minTime. The forward-progress rule keeps two
  // ops from displacing each other endlessly (Rau §3.4).
  unsigned evictTime;
  auto prevIt = prevSched.find(n);
  if (prevIt == prevSched.end() || stN > prevIt->second)
    evictTime = stN;
  else
    evictTime = prevIt->second + 1;

  // Resource conflict at the forced slot: evict the occupants.
  SmallVector<Operation *> conflicts;
  mrt.gatherConflicts(n, evictTime, conflicts);
  if (!conflicts.empty()) {
    for (auto *c : conflicts)
      unpin(c);
    if (failed(rebuildWithPins()))
      return failure();
  }
  if (failed(mrt.enter(n, evictTime)))
    return failure(); // stale MRT state; should not happen

  if (failed(scheduleAt(stvN, evictTime))) {
    // Dependence conflict with the remaining pinned ops. Take Canis's
    // aggressive branch: unpin everything else and re-place it later.
    SmallVector<Operation *> allPinned;
    for (auto &entry : pinnedOps)
      allPinned.push_back(entry.first);
    for (auto *p : allPinned)
      unpin(p);
    if (failed(rebuildWithPins()))
      return failure();
    stvN = startTimeVariables[n];
    if (failed(scheduleAt(stvN, evictTime))) {
      // Infeasible even with n as the only pin: a recurrence bounds n's
      // start time below `evictTime`. Let the driver raise the II.
      mrt.release(n);
      return failure();
    }
  }
  pinnedOps[n] = evictTime;
  prevSched[n] = evictTime;
  LLVM_DEBUG(dbgs() << "Forced placement at t=" << evictTime << " " << *n
                    << '\n');
  return success();
}

unsigned ChainingModuloSimplexScheduler::computeResMinII() {
  unsigned resMinII = 1;
  SmallDenseMap<Problem::ResourceType, unsigned> uses;
  for (auto *op : prob.getOperations()) {
    if (isLimited(op, prob)) {
      auto maybeRsrcs = prob.getLinkedResourceTypes(op);
      if (maybeRsrcs.has_value()) {
        for (auto rsrc : *maybeRsrcs) {
          ++uses[rsrc];
        }
      }
    }
  }

  for (auto pair : uses) {
    resMinII = std::max(
        resMinII, (unsigned)ceil(pair.second / *prob.getLimit(pair.first)));
  }

  // llvm::errs() << "resMinII = " << resMinII << "\n";
  return resMinII;
}

LogicalResult ChainingModuloSimplexScheduler::schedule() {
  if (failed(checkLastOp()) || failed(computeChainBreakingDependences(
                                   prob, cycleTime, additionalConstraints)))
    return failure();

  parameterS = 0;
  parameterT = std::max(computeResMinII() + iiIncrement, 1u);

  LLVM_DEBUG(dbgs() << "Attempting to schedule with II = " << parameterT
                    << "\n");
  buildTableau();

  LLVM_DEBUG(dbgs() << "Initial tableau:\n"; dumpTableau());

  if (failed(solveTableau()))
    return prob.getContainingOp()->emitError() << "problem is infeasible";

  // Static, height-based placement order (recomputed per II attempt since
  // inter-iteration edges weigh in at II*distance).
  computeHeightPriorities();

  // Determine which operations are subject to resource constraints.
  auto &ops = prob.getOperations();
  for (auto *op : ops)
    if (isLimited(op, prob))
      unscheduled.push_back(op);

  // Rau's placement budget: evicted ops return to the worklist, so bound
  // the total number of (re-)placements before conceding this II.
  unsigned budget = budgetRatio * ops.size();

  // Main loop: Iteratively fix limited operations to time steps. Evictions
  // requeue ops, so the same op may be placed several times.
  while (!unscheduled.empty()) {
    if (budget == 0) {
      LLVM_DEBUG(dbgs() << "Budget exhausted at II = " << parameterT
                        << "\n");
      return failure();
    }
    --budget;
    // Pick the highest-priority operation (Rau's height-based priority:
    // ops heading long dependence chains place first, so their consumers'
    // deadlines are computed against committed producers, not optimistic
    // floating times).
    auto *opIt =
        std::min_element(unscheduled.begin(), unscheduled.end(),
                         [&](Operation *opA, Operation *opB) {
                           int ha = heightPriority.lookup(opA);
                           int hb = heightPriority.lookup(opB);
                           if (ha != hb)
                             return ha > hb;
                           return stableIndex.lookup(opA) <
                                  stableIndex.lookup(opB);
                         });
    Operation *op = *opIt;
    unscheduled.erase(opIt);

    if (failed(scheduleOperation(op)))
      return failure();
    scheduled.push_back(op);
  }

  LLVM_DEBUG(dbgs() << "Final tableau:\n"; dumpTableau();
             dbgs() << "Solution found with II = " << parameterT
                    << " and start time of last operation = "
                    << -getParametricConstant(0) << '\n');

  prob.setInitiationInterval(parameterT);
  for (auto *op : ops)
    prob.setStartTime(op, getStartTime(startTimeVariables[op]));

  auto filledIn = computeStartTimesInCycle(prob);
  assert(succeeded(filledIn));
  (void)filledIn;

  return success();
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

LogicalResult scheduling::scheduleSimplex(Problem &prob, Operation *lastOp) {
  SimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(CyclicProblem &prob,
                                          Operation *lastOp) {
  CyclicSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(SharedOperatorsProblem &prob,
                                          Operation *lastOp) {
  SharedOperatorsSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ModuloProblem &prob,
                                          Operation *lastOp) {
  ModuloSimplexScheduler simplex(prob, lastOp);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ChainingProblem &prob,
                                          Operation *lastOp, float cycleTime) {
  ChainingSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ChainingSharedOperatorsProblem &prob,
                                          Operation *lastOp, float cycleTime) {
  ChainingSharedOperatorsSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ChainingCyclicProblem &prob,
                                          Operation *lastOp, float cycleTime) {
  ChainingCyclicSimplexScheduler simplex(prob, lastOp, cycleTime);
  return simplex.schedule();
}

LogicalResult scheduling::scheduleSimplex(ChainingModuloProblem &prob,
                                          Operation *lastOp, float cycleTime,
                                          std::optional<int32_t> ii) {
  // Rau's ModuloSchedule outer loop: attempt successively larger IIs, each
  // with a fresh scheduler whose backtracking placement is bounded by the
  // budget. The attempt cap only guards against pathological inputs; a
  // feasible problem schedules long before it (usually at the first II).
  constexpr unsigned kMaxAttempts = 64;
  for (unsigned attempt = 0; attempt < kMaxAttempts; ++attempt) {
    ChainingModuloSimplexScheduler simplex(prob, lastOp, cycleTime, ii);
    simplex.iiIncrement = attempt;
    if (succeeded(simplex.schedule()))
      return success();
    // A user-requested exact II leaves no room to retry.
    if (ii.has_value())
      return failure();
  }
  return prob.getContainingOp()->emitError()
         << "modulo scheduling failed to find a feasible II after "
         << kMaxAttempts << " attempts";
}
