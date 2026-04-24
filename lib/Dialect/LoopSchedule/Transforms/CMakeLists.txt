add_circt_dialect_library(CIRCTLoopScheduleTransforms
  MarkMemoryAccesses.cpp
  ConstructMemoryDependencies.cpp
  UnrollForLoopSchedule.cpp
  PipelineForLoopSchedule.cpp
  BitwidthReductionForLoopSchedule.cpp
  UnrollSubLoops.cpp
  UnrollMarkedLoops.cpp
  LoopScheduleBinding.cpp
  LoopScheduleTestbenchGeneration.cpp

  DEPENDS
  CIRCTLoopScheduleTransformsIncGen
  CIRCTLoopScheduleAttributesIncGen

  LINK_LIBS PUBLIC
  CIRCTLoopSchedule
  CIRCTBinding
  CIRCTHW
  CIRCTSV
  CIRCTComb
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
