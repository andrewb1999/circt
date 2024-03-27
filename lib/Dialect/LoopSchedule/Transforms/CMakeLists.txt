add_circt_dialect_library(CIRCTLoopScheduleTransforms
  MarkMemoryAccesses.cpp
  ConstructMemoryDependencies.cpp

  DEPENDS
  CIRCTLoopScheduleTransformsIncGen
  CIRCTLoopScheduleAttributesIncGen

  LINK_LIBS PUBLIC
  CIRCTLoopSchedule
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
