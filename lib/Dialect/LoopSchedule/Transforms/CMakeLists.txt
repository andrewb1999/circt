add_circt_dialect_library(CIRCTLoopScheduleTransforms
  BitwidthMinimization.cpp

  DEPENDS
  CIRCTLoopScheduleTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTLoopSchedule
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
