add_circt_conversion_library(CIRCTAffineToSCF
  AffineToSCF.cpp

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
