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
#include "mlir/IR/Operation.h"
#include <cassert>
#include <vector>

using namespace circt;
using namespace circt::scheduling;

const int globalPortLimit = 2;

void appendReservTable(std::vector<int> &vec, unsigned int cycle) {
  while (vec.size() < cycle + 1) {
    vec.push_back(globalPortLimit);
  }
}

bool reservePort(std::vector<int> &vec, unsigned int atCycle) {
  appendReservTable(vec, atCycle);
  vec[atCycle]--;
  return vec[atCycle] >= 0;
}

LogicalResult scheduling::scheduleList(SharedOperatorsProblem &prob,
                                       Operation *lastOp) {

  std::vector<int> portReservations;
  return handleOperationsInTopologicalOrder(prob, [&](Operation *op) {
    // Operations with no predecessors are scheduled at time step 0
    assert(prob.getLinkedOperatorType(op).has_value());
    auto operatorType = prob.getLinkedOperatorType(op).value().str();
    if (prob.getDependences(op).empty()) {

      if (operatorType == "ld" || operatorType == "st") {
        if (reservePort(portReservations, 0))
          prob.setStartTime(op, 0);
        else
          return failure();
      } else {
        prob.setStartTime(op, 0);
      }

      return success();
    }

    // op has at least one predecessor. Compute start time as:
    //   max_{p : preds} startTime[p] + latency[linkedOpr[p]]
    unsigned startTime = 0;
    for (auto &dep : prob.getDependences(op)) {
      Operation *pred = dep.getSource();
      auto predStart = prob.getStartTime(pred);
      if (!predStart)
        // pred is not yet scheduled, give up and try again later
        return failure();

      // pred is already scheduled
      auto predOpr = *prob.getLinkedOperatorType(pred);
      startTime = std::max(startTime, *predStart + *prob.getLatency(predOpr));
    }
    if (operatorType == "ld" || operatorType == "st") {
      if (reservePort(portReservations, startTime))
        prob.setStartTime(op, startTime);
      else
        return failure();
    } else {
      prob.setStartTime(op, startTime);
    }
    return success();
  });
}
