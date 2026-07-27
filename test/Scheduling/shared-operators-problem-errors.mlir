// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// A zero-latency operator MAY use a limited resource — use is accounted for by
// start time, so an operation that finishes in the cycle it starts still
// occupies its resource for that cycle (an FWFT FIFO read is the motivating
// case: combinational data, one read-enable strobe). It is oversubscribed by
// the same rule as any other operator.

// expected-error@+1 {{Resource type 'limited_rsrc' is oversubscribed}}
ssp.instance @zero_latency_oversubscribed of "SharedOperatorsProblem" {
  library {
    operator_type @limited [latency<0>]
  }
  resource {
    resource_type @limited_rsrc [limit<1>]
  }
  graph {
    operation<@limited>() uses[@limited_rsrc] [t<0>]
    operation<@limited>() uses[@limited_rsrc] [t<0>]
  }
}

// -----

// expected-error@+1 {{Resource type 'limited_rsrc' is oversubscribed}}
ssp.instance @oversubscribed of "SharedOperatorsProblem" {
  library {
    operator_type @limited [latency<1>]
  }
  resource {
    resource_type @limited_rsrc [limit<2>]
  }
  graph {
    operation<@limited>() uses[@limited_rsrc] [t<0>]
    operation<@limited>() uses[@limited_rsrc] [t<0>]
    operation<@limited>() uses[@limited_rsrc] [t<0>]
  }
}
