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
#include <algorithm>
#include <cassert>
#include <map>
#include <utility>
#include <vector>

using namespace circt;
using namespace circt::scheduling;

bool takeReservation(
    std::map<std::pair<std::string, unsigned int>, int> &reservationTable,
    SharedOperatorsProblem &prob, Operation *op, unsigned int cycle) {
  auto operatorType = prob.getLinkedOperatorType(op);
  assert(operatorType.has_value());
  if (!prob.getLimit(operatorType.value()).has_value()) {
    return true;
  }

  auto typeLimit = prob.getLimit(operatorType.value()).value();
  assert(typeLimit > 0);
  auto key = std::pair(operatorType.value().str(), cycle);
  if (reservationTable.count(key) == 0) {
    reservationTable.insert(std::pair(key, (int)typeLimit - 1));
    return true;
  }
  reservationTable[key]--;
  return reservationTable[key] >= 0;
}

bool testReservation(
    std::map<std::pair<std::string, unsigned int>, int> &reservationTable,
    SharedOperatorsProblem &prob, Operation *op, unsigned int cycle) {
  auto operatorType = prob.getLinkedOperatorType(op);
  assert(operatorType.has_value());
  if (!prob.getLimit(operatorType.value()).has_value()) {
    return true;
  }

  auto typeLimit = prob.getLimit(operatorType.value()).value();
  assert(typeLimit > 0);
  auto key = std::pair(operatorType.value().str(), cycle);
  if (reservationTable.count(key) == 0) {
    reservationTable.insert(std::pair(key, (int)typeLimit));
    return true;
  }
  return reservationTable[key] > 0;
}

bool testSchedule(
    std::map<std::pair<std::string, unsigned int>, int> &reservationTable,
    SharedOperatorsProblem &prob, Operation *op, unsigned int cycle) {
  auto operatorType = prob.getLinkedOperatorType(op);
  assert(operatorType.has_value());
  auto operatorLatency = prob.getLatency(operatorType.value());
  assert(operatorLatency.has_value());
  bool result = testReservation(reservationTable, prob, op, cycle);
  for (unsigned int i = cycle + 1; i < cycle + operatorLatency.value(); i++)
    result &= testReservation(reservationTable, prob, op, i);

  return result;
}

void takeSchedule(
    std::map<std::pair<std::string, unsigned int>, int> &reservationTable,
    SharedOperatorsProblem &prob, Operation *op, unsigned int cycle) {
  auto operatorType = prob.getLinkedOperatorType(op);
  assert(operatorType.has_value());
  auto operatorLatency = prob.getLatency(operatorType.value());
  assert(operatorLatency.has_value());
  takeReservation(reservationTable, prob, op, cycle);
  for (unsigned int i = cycle + 1; i < cycle + operatorLatency.value(); i++)
    takeReservation(reservationTable, prob, op, i);
  prob.setStartTime(op, cycle);
}

LogicalResult scheduling::scheduleList(SharedOperatorsProblem &prob,
                                       Operation *lastOp) {

  std::map<std::pair<std::string, unsigned int>, int> reservationTable;

  SmallVector<Operation *> unscheduledOps;
  unsigned int totalLatency = 0;

  // Schedule Ops with no Dependencies
  for (auto it = prob.getOperations().rbegin();
       it != prob.getOperations().rend(); it++) {
    auto *op = *it;
    if (op == lastOp)
      continue;
    if (prob.getDependences(op).empty()) {
      if (testSchedule(reservationTable, prob, op, 0)) {
        takeSchedule(reservationTable, prob, op, 0);
      } else {
        unscheduledOps.push_back(op);
      }
    } else {
      // Dependencies are not fulfilled
      unscheduledOps.push_back(op);
    }
  }

  while (!unscheduledOps.empty()) {
    SmallVector<Operation *> worklist;
    worklist.insert(worklist.begin(), unscheduledOps.rbegin(),
                    unscheduledOps.rend());
    unscheduledOps.clear();

    for (auto *op : worklist) {
      unsigned int schedCycle = 0;
      bool ready = true;
      for (auto dep : prob.getDependences(op)) {
        auto depStart = prob.getStartTime(dep.getSource());
        if (!depStart.has_value()) {
          unscheduledOps.push_back(op);
          ready = false;
          break;
        }
        schedCycle = std::max(
            schedCycle,
            depStart.value() +
                prob.getLatency(
                        prob.getLinkedOperatorType(dep.getSource()).value())
                    .value());
      }
      if (ready) {
        unsigned int earliest = schedCycle;
        while (!testSchedule(reservationTable, prob, op, earliest))
          earliest++;
        takeSchedule(reservationTable, prob, op, earliest);
        totalLatency = std::max(totalLatency, earliest);
      }
    }
  }
  prob.setStartTime(lastOp, totalLatency);
  return success();
}
