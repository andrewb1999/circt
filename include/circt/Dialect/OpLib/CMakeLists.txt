add_circt_dialect(OpLib oplib)
add_circt_dialect_doc(OpLib oplib)

set(LLVM_TARGET_DEFINITIONS OpLib.td)

mlir_tablegen(OpLibAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(OpLibAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTOpLibAttributesIncGen)
add_dependencies(circt-headers CIRCTOpLibAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS OpLibInterfaces.td)
mlir_tablegen(OpLibInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpLibInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTOpLibInterfacesIncGen)
add_dependencies(circt-headers CIRCTOpLibInterfacesIncGen)
