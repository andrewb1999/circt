add_circt_conversion_library(CIRCTSCFToLoopSchedule
  SCFToLoopScheduleCPSAT.cpp
  SCFToLoopScheduleLowering.cpp
  SCFToLoopScheduleSimplex.cpp

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

# The CP-SAT pass (convert-scf-to-loopschedule-cpsat) is built with OR-Tools.
if(ortools_FOUND)
  target_compile_definitions(obj.CIRCTSCFToLoopSchedule PRIVATE
                             SCHEDULING_OR_TOOLS)
endif()
