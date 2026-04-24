set(BINDING_SOURCES
  Problems.cpp
  GreedyBinder.cpp
  )

add_circt_library(CIRCTBinding
  ${BINDING_SOURCES}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  )
