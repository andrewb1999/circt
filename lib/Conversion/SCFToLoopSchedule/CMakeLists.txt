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

# The CP-SAT scheduler (scheduler=cpsat) is built with OR-Tools.
if(ortools_FOUND)
  target_compile_definitions(obj.CIRCTSCFToLoopSchedule PRIVATE
                             SCHEDULING_OR_TOOLS)
endif()
