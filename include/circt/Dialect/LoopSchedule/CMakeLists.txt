add_circt_dialect(LoopSchedule loopschedule)
add_circt_dialect_doc(LoopSchedule loopschedule)

set(LLVM_TARGET_DEFINITIONS LoopSchedulePasses.td)
mlir_tablegen(LoopSchedulePasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTLoopScheduleTransformsIncGen)

set(LLVM_TARGET_DEFINITIONS LoopScheduleInterfaces.td)
mlir_tablegen(LoopScheduleInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(LoopScheduleInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTLoopScheduleInterfacesIncGen)
add_dependencies(circt-headers CIRCTLoopScheduleInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS LoopSchedule.td)
mlir_tablegen(LoopScheduleAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(LoopScheduleAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(CIRCTLoopScheduleAttributesIncGen)
add_dependencies(circt-headers CIRCTLoopScheduleAttributesIncGen)

# Generate Pass documentation.
add_circt_doc(LoopSchedulePasses LoopSchedulePasses -gen-pass-doc)
