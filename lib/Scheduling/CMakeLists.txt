set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  CPSATSchedulers.cpp
  LPSchedulers.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  Utilities.cpp
  )

set(SCHEDULING_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  Utilities.cpp
  )

set(SCHEDULING_LIBS
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  )

if(ortools_FOUND)
  list(APPEND SCHEDULING_SOURCES LPSchedulers.cpp CPSATSchedulers.cpp)
  list(APPEND SCHEDULING_LIBS ortools::ortools)
endif()

add_circt_library(CIRCTScheduling
  ${SCHEDULING_SOURCES}

  LINK_LIBS PUBLIC
  ${SCHEDULING_LIBS}
  )

if(ortools_FOUND)
  # The object library LLVM compiles the sources in takes the include
  # directories of LINK_LIBS but not the ortools target's compile
  # definitions (OR_PROTO_DLL, which its protobuf headers need, ...).
  get_target_property(ortools_defs ortools::ortools
                      INTERFACE_COMPILE_DEFINITIONS)
  target_compile_definitions(obj.CIRCTScheduling PRIVATE ${ortools_defs})
endif()
