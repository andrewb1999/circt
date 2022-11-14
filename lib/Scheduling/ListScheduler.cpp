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
#include <string>
#include <unordered_map>
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

std::map<Operation *, int> getALAPPriorities(Problem::OperationSet opSet,
                                             Problem &prob, Operation *lastOp) {
  std::map<Operation *, int> map;
  std::map<Operation *, bool> isSink;
  assert(opSet.contains(lastOp));
  for (auto *op : opSet)
    isSink.insert(std::pair(op, true));
  SmallVector<Operation *> reverseTopoSort;
  SmallVector<Operation *> unhandledOps;
  unhandledOps.insert(unhandledOps.begin(), opSet.begin(), opSet.end());
  const unsigned int fSize = opSet.size();
  while (reverseTopoSort.size() < fSize) {
    Problem::OperationSet workList;
    for (auto it = unhandledOps.rbegin(); it != unhandledOps.rend(); it++)
      workList.insert(*it);

    unhandledOps.clear();

    for (auto *op : workList) {
      bool noDep = (lastOp != op && prob.getDependences(op).empty()) ||
                   (lastOp == op && workList.size() == 1);
      if (!noDep && lastOp != op) {
        bool depsAdded = true;
        for (auto pred : prob.getDependences(op)) {
          assert(opSet.contains(pred.getSource()));
          isSink[pred.getSource()] = false;
          depsAdded &= !workList.contains(pred.getSource());
        }
        noDep |= depsAdded;
      }
      if (noDep) {
        reverseTopoSort.push_back(op);
      } else {
        unhandledOps.push_back(op);
      }
    }
  }

  for (int i = reverseTopoSort.size() - 1; i >= 0; i--) {
    auto *op = reverseTopoSort.data()[i];
    int priority = opSet.size();
    if (isSink[op]) {
      map.insert(std::pair(op, opSet.size()));
    } else {
      for (auto *sop : opSet) {
        for (auto dep : prob.getDependences(sop)) {
          if (dep.getSource() == op) {
            assert(map.count(dep.getDestination()) == 1);
            priority = std::min(priority, map[dep.getDestination()]);
          }
        }
      }
      map.insert(std::pair(op, priority - 1));
    }
  }

  return map;
}

bool priorityCmp(std::pair<Operation *, int> &a,
                 std::pair<Operation *, int> &b) {
  return a.second < b.second;
}

LogicalResult scheduling::scheduleList(SharedOperatorsProblem &prob,
                                       Operation *lastOp) {

  std::map<std::pair<std::string, unsigned int>, int> reservationTable;
  SmallVector<Operation *> unscheduledOps;
  unsigned int totalLatency = 0;
  auto map = getALAPPriorities(prob.getOperations(), prob, lastOp);
  SmallVector<std::pair<Operation *, int>> mapSet;
  mapSet.insert(mapSet.begin(), map.begin(), map.end());
  // sort by low priority to high priority, high priority == sinks
  std::sort(mapSet.begin(), mapSet.end(), priorityCmp);
  // Schedule Ops with no Dependencies
  for (auto &pair : mapSet) {
    auto *op = pair.first;
    // llvm::errs() << op->getName() << ": " << std::to_string(pair.second) <<
    // "\n";
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
    worklist.insert(worklist.begin(), unscheduledOps.begin(),
                    unscheduledOps.end());
    unscheduledOps.clear();

    for (auto *op : worklist) {
      // llvm::errs() << op->getName() << "\n";
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
