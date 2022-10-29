//===- PipelinePasses.h - Pipeline pass entry points ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_STG_STGPASSES_H
#define CIRCT_DIALECT_STG_STGPASSES_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
namespace stg {

// std::unique_ptr<mlir::Pass> createExplicitRegsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/STG/STGPasses.h.inc"

} // namespace stg
} // namespace circt

#endif // CIRCT_DIALECT_STG_STGPASSES_H
