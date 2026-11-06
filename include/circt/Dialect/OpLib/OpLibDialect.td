//===- OpLibDialect.td - OpLib dialect ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OPLIB_OPLIBDIALECT_TD
#define CIRCT_DIALECT_OPLIB_OPLIBDIALECT_TD

include "mlir/IR/DialectBase.td"

def OpLib_Dialect : Dialect {
  let name = "oplib";
  let cppNamespace = "::circt::oplib";
  let summary = "Operator library dialect";
  // TODO: Migrate custom assembly formats to prop-dict.
  let useStrictPropertiesInAssemblyFormat = 0;

  let useDefaultAttributePrinterParser = true;

  let extraClassDeclaration = [{
    /// Register all SSP attributes.
    void registerAttributes();
  }];
}

#endif // CIRCT_DIALECT_OPLIB_OPLIBDIALECT_TD
