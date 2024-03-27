//===- AccessNameAnalysis.h
//----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_ACCESSNAMEANALYSIS_H
#define CIRCT_ANALYSIS_ACCESSNAMEANALYSIS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class Operation;

namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace circt {

struct AccessNameAnalysis {
  AccessNameAnalysis(Operation *op);

  // Returns the operation that has a given name.
  Operation *getOperationFromName(StringRef name);

  // Returns the name of a given Operation.
  llvm::StringRef getOperationName(Operation *);

  // Updates name mapping from the oldOp to the newOp.
  void replaceOp(Operation *oldOp, Operation *newOp);

  llvm::StringMap<Operation *> nameToOperationMap;
  llvm::MapVector<Operation *, StringRef> operationToNameMap;
};

} // namespace circt

#endif // CIRCT_ANALYSIS_ACCESSNAMEANALYSIS_H
