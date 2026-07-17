add_circt_dialect_library(CIRCTLoopScheduleTransforms
  MarkMemoryAccesses.cpp
  ConstructMemoryDependencies.cpp
  BitwidthReductionForLoopSchedule.cpp
  UnrollSubLoops.cpp
  LoopScheduleBinding.cpp
  LoopScheduleTestbenchGeneration.cpp

  DEPENDS
  CIRCTLoopScheduleTransformsIncGen
  CIRCTLoopScheduleAttributesIncGen

  LINK_LIBS PUBLIC
  CIRCTLoopSchedule
  CIRCTLoopScheduleStartTimeAnalysis
  CIRCTOperatorLibraryAnalysis
  CIRCTBinding
  CIRCTHW
  CIRCTSV
  CIRCTComb
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
