

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace circt::scheduling;
using namespace circt::analysis;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

int loadMLIR(mlir::MLIRContext &context,
             std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
             mlir::OwningOpRef<ModuleOp> &module) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());
  module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
  return 0;
}

int runModule(std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
              raw_ostream &os) {

  // Register dialects and passes in current context
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<calyx::CalyxDialect>();
  circt::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
  // context.allowUnregisteredDialects(true);
  context.printOpOnDiagnostic(true);
  context.loadAllAvailableDialects();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  // mlir::registerSCFForToWhileLoopPass();

  mlir::OwningOpRef<ModuleOp> module;
  if (int error = loadMLIR(context, std::move(ownedBuffer), module))
    return error;
  auto funcOp = *module->getOps<func::FuncOp>().begin();
  auto *funcBody = &funcOp.getFunctionBody();
  Region *loopBody = nullptr;
  Operation *forOp = nullptr;
  Operation *lastOp = nullptr;
  if (!funcBody->getOps<scf::ForOp>().empty()) {
    forOp = *funcBody->getOps<scf::ForOp>().begin();
    loopBody = &(*funcBody->getOps<scf::ForOp>().begin()).getLoopBody();
  } else if (!funcBody->getOps<AffineForOp>().empty()) {
    forOp = *funcBody->getOps<AffineForOp>().begin();
    loopBody = &(*funcBody->getOps<AffineForOp>().begin()).getLoopBody();
  } else {
    llvm::errs() << " Could not find a loop!\n";
    module->dump();
  }
  assert(loopBody != nullptr);
  auto problem = scheduling::SharedOperatorsProblem::get(forOp);
  MemoryDependenceAnalysis dependences(funcOp);

  Problem::OperatorType addOpr = problem.getOrInsertOperatorType("add");
  problem.setLatency(addOpr, 1);
  Problem::OperatorType mulOpr = problem.getOrInsertOperatorType("mul");
  problem.setLatency(mulOpr, 2);
  Problem::OperatorType ldOpr = problem.getOrInsertOperatorType("ld");
  problem.setLatency(ldOpr, 2);
  Problem::OperatorType stOpr = problem.getOrInsertOperatorType("st");
  problem.setLatency(stOpr, 1);
  Problem::OperatorType chainOpr = problem.getOrInsertOperatorType("chain");
  problem.setLatency(chainOpr, 0);

  WalkResult result = loopBody->walk([&](Operation *op) {
    problem.insertOperation(op);
    auto opDep = dependences.getDependences(op);
    for (const auto &dep : opDep) {
      if (dep.dependenceType == mlir::DependenceResult::NoDependence)
        continue;
      llvm::errs() << "dependence between \n";
      dep.source->dump();
      op->dump();
      auto pDep = Problem::Dependence(dep.source, op);
      assert(problem.insertDependence(pDep).succeeded());
    }
    return mlir::TypeSwitch<Operation *, WalkResult>(op)
        .Case<AddIOp>([&](Operation *combOp) {
          // Add can be combinational
          problem.setLinkedOperatorType(combOp, addOpr);
          return WalkResult::advance();
        })
        .Case<AffineLoadOp, memref::LoadOp>([&](Operation *seqOp) {
          // Make loads 2 cycles
          problem.setLinkedOperatorType(seqOp, ldOpr);
          return WalkResult::advance();
        })
        .Case<AffineStoreOp, memref::StoreOp>([&](Operation *seqOp) {
          // Stores only 1 cycle
          problem.setLinkedOperatorType(seqOp, stOpr);
          return WalkResult::advance();
        })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Multiply cause why not
          problem.setLinkedOperatorType(mcOp, mulOpr);
          return WalkResult::advance();
        })
        .Case<AffineYieldOp>([&](Operation *yieldOp) {
          // set last op to yield.
          lastOp = yieldOp;
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          llvm::errs() << "Unhandled OP: ";
          badOp->dump();
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    forOp->emitError("unsupported operation ");

  assert(scheduleList(problem, lastOp).succeeded());

  auto scheduleOps = problem.getOperations();
  for (auto *schedOp : scheduleOps) {
    assert(problem.getStartTime(schedOp).has_value());
    llvm::errs() << schedOp->getName() << ": "
                 << std::to_string(problem.getStartTime(schedOp).value())
                 << "\n";
  }

  return 0;
}

int main(int argc, char **argv) {

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MLIR modular optimizer driver\n");

  std::string errorMessage;
  auto input = mlir::openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 3;
  }

  // Processes the memory buffer with a new MLIRContext.
  mlir::ChunkBufferHandler processBuffer =
      [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer, raw_ostream &os) {
        runModule(std::move(ownedBuffer), os);
        return success();
      };

  if (failed(splitAndProcessBuffer(std::move(input), processBuffer,
                                   output->os()))) {
    return 4;
  }
}
