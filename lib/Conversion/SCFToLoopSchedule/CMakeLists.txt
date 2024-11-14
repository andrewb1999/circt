add_circt_conversion_library(CIRCTSCFToLoopSchedule
  SCFToLoopSchedule.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTOperatorLibraryAnalysis
  CIRCTLoopSchedule
  MLIRArithTransforms
)
