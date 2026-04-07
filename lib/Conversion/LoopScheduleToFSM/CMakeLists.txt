add_circt_conversion_library(CIRCTLoopScheduleToFSM
  LoopScheduleToFSM.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTFSM
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTLoopSchedule
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
)
