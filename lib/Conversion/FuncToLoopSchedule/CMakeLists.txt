add_circt_library(CIRCTFuncToLoopSchedule
  FuncToLoopSchedule.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTLoopSchedule
  MLIRArithTransforms
  )
