add_circt_dialect(Calyx calyx)
add_circt_dialect_doc(Calyx calyx)

set(LLVM_TARGET_DEFINITIONS Calyx.td)

mlir_tablegen(CalyxEnums.h.inc -gen-enum-decls)
mlir_tablegen(CalyxEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRCalyxEnumsIncGen)
add_dependencies(circt-headers MLIRCalyxEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxInterfaces.td)
mlir_tablegen(CalyxInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CalyxInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCalyxInterfacesIncGen)
add_dependencies(circt-headers MLIRCalyxInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxLoweringInterfaces.td)
mlir_tablegen(CalyxLoweringInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(CalyxLoweringInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRCalyxLoweringInterfacesIncGen)
add_dependencies(circt-headers MLIRCalyxLoweringInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxLoweringInterfaces.td)
mlir_tablegen(CalyxLoweringTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(CalyxLoweringTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRCalyxLoweringTypeInterfacesIncGen)
add_dependencies(circt-headers MLIRCalyxLoweringTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS CalyxPasses.td)
mlir_tablegen(CalyxPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTCalyxTransformsIncGen)
add_circt_doc(CalyxPasses CalyxPasses -gen-pass-doc)
