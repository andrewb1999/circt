set(LLVM_OPTIONAL_SOURCES
  DebugAnalysis.cpp
  AccessNameAnalysis.cpp
  DebugInfo.cpp
  DependenceAnalysis.cpp
  SchedulingAnalysis.cpp
  LoopScheduleDependenceAnalysis.cpp
  TestPasses.cpp
)

add_circt_library(CIRCTDebugAnalysis
  DebugAnalysis.cpp
  DebugInfo.cpp

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  MLIRIR
)

add_circt_library(CIRCTDependenceAnalysis
  DependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineUtils
  MLIRTransformUtils
)

add_circt_library(CIRCTLoopScheduleDependenceAnalysis
  LoopScheduleDependenceAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineUtils
  MLIRTransformUtils
)

add_circt_library(CIRCTSchedulingAnalysis
  SchedulingAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  CIRCTDependenceAnalysis
  CIRCTScheduling
)

add_circt_library(CIRCTAccessNameAnalysis
  AccessNameAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
)

add_circt_library(CIRCTAnalysisTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTDebugAnalysis
  CIRCTDependenceAnalysis
  CIRCTSchedulingAnalysis
  CIRCTAccessNameAnalysis
  CIRCTLoopScheduleDependenceAnalysis
  CIRCTHW
  MLIRPass
)
