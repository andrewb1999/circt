//===- OperatorLibraryAnalysis.h - operator library analyses --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving memory access dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_OPERATOR_LIBRARY_ANALYSIS_H
#define CIRCT_ANALYSIS_OPERATOR_LIBRARY_ANALYSIS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/OpLib/OpLibOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/AnalysisManager.h"
#include <utility>

namespace circt {
namespace analysis {

struct Operator {
  Operator(unsigned latency) : latency(latency) {}

  unsigned latency;

  std::optional<llvm::APFloat> incDelay;

  std::optional<llvm::APFloat> outDelay;

  // Must be calyx::CellInterface
  Operation *templateOp;

  Operation *opToMatch;

  std::map<unsigned, unsigned> operandToCellResultMapping;

  std::map<unsigned, unsigned> resultToCellResultMapping;

  std::optional<unsigned> clock;

  std::optional<unsigned> ce;

  std::optional<unsigned> reset;

  SmallVector<mlir::NamedAttribute> attrs;

  // Optional `oplib.hw_match` body for this operator. Null when the
  // operator has no hw_match. The FSM lowering pass reads this op's
  // discardable attributes (e.g. `hw_op = "comb.add"` for the comb-op
  // form, `extern_module = @<symbol>` for the extern-instance form) to
  // decide how to materialize the operator in HW.
  oplib::HwMatchOp hwMatchOp;
};

using PotentialOperatorsMap = std::map<StringRef, SmallVector<StringRef>>;

using OperatorMap = std::map<StringRef, Operator>;

struct OperatorLibraryAnalysis {
  // Construct the analysis from an oplib::Library.
  OperatorLibraryAnalysis(Operation *op);

  SmallVector<StringRef> getPotentialOperators(Operation *);

  StringRef getOperatorBySymbol(mlir::SymbolRefAttr);

  SmallVector<StringRef> getAllSupportedTargets();

  Operation *getOperatorTemplateOp(StringRef);

  unsigned int getOperatorLatency(StringRef);

  std::optional<float> getOperatorIncomingDelay(StringRef);

  std::optional<float> getOperatorOutgoingDelay(StringRef);

  std::optional<unsigned> getCEResultNum(StringRef operatorName);

  unsigned getCellResultForOperandNum(StringRef operatorName,
                                      unsigned operandNum);

  unsigned getCellResultForResultNum(StringRef operatorName,
                                     unsigned resultNum);

  // Returns the operator's `oplib.hw_match` op, or null if absent.
  oplib::HwMatchOp getHwMatchOp(StringRef operatorName);

private:
  PotentialOperatorsMap potentialOperatorsMap;
  OperatorMap operatorMap;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_OPERATOR_LIBRARY_ANALYSIS_H
