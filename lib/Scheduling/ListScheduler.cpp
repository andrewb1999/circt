//===- ListScheduler.cpp - List scheduler -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of a resource-constrained list scheduler for acyclic problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"

using namespace circt;
using namespace circt::scheduling;

LogicalResult scheduling::scheduleList(SharedOperatorsProblem &prob, Operation *lastOp) {
  return failure();
}
