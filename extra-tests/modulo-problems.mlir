// RUN: circt-opt %s -ssp-roundtrip=verify
// RUN: circt-opt %s -ssp-schedule=scheduler=simplex | FileCheck %s -check-prefixes=CHECK,SIMPLEX

// CHECK-LABEL: four_read_pipeline
// SIMPLEX-SAME: [II<4>]
ssp.instance @four_read_pipeline of "ModuloProblem" {
  library {
    operator_type @AMemPort [latency<1>, limit<1>]
    operator_type @ResReadPort [latency<0>]
    operator_type @ResWritePort [latency<0>]
    operator_type @Add [latency<4>, limit<1>]
  }
  graph {
    %0 = operation<@AMemPort>()
    %1 = operation<@ResReadPort> (@store_res0 [dist<1>])
    %2 = operation<@Add>(%0, %1)
    operation<@ResWritePort> @store_res0(%2)
    %3 = operation<@AMemPort>()
    %4 = operation<@ResReadPort> (@store_res1 [dist<1>])
    %5 = operation<@Add>(%3, %4)
    operation<@ResWritePort> @store_res1(%5, @store_res0 [dist<0>])
  }
}
