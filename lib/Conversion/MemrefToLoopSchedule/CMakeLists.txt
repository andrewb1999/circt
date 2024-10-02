add_circt_conversion_library(CIRCTMemrefToLoopSchedule
  MemrefToLoopSchedule.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTLoopSchedule
  MLIRArithTransforms
  )
