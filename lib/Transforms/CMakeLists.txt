add_circt_library(CIRCTTransforms
  CellIFT.cpp
  FlattenMemRefs.cpp
  HierarchicalRunner.cpp
  IndexSwitchToIf.cpp
  InsertMergeBlocks.cpp
  ConvertIndexToUInt.cpp
  MapArithToComb.cpp
  MaximizeSSA.cpp
  InsertMergeBlocks.cpp
  IfOpHoisting.cpp
  IndexSwitchToIf.cpp
  ArithStrengthReduction.cpp
  IndexRemoval.cpp
  PrintOpCount.cpp
  MemoryBanking.cpp
  PrintOpCount.cpp
  SCFWhileLoopFlattening.cpp
  StripDebugInfoWithPred.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLoopSchedule
  CIRCTSeq
  CIRCTOpCountAnalysis
  CIRCTSCFWhileTripCountAnalysis
  MLIRArithDialect
  MLIRArithTransforms
  MLIRControlFlowDialect
  MLIRControlFlowTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineDialect
  MLIRSCFDialect

  DEPENDS
  CIRCTTransformsPassIncGen
  MLIRArithDialect
)
