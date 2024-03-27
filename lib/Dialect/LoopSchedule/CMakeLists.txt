add_circt_dialect_library(CIRCTLoopSchedule
  LoopScheduleOps.cpp
  LoopScheduleAttributes.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRArithDialect
  MLIRLoopScheduleIncGen
  CIRCTLoopScheduleAttributesIncGen
)

add_subdirectory(Transforms)
