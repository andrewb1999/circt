// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-grand-central)' -split-input-file -verify-diagnostics %s

// expected-error @+1 {{more than one 'ExtractGrandCentralAnnotation' was found, but exactly one must be provided}}
firrtl.circuit "MoreThanOneExtractGrandCentralAnnotation" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {name = "foo",
        tpe = "sifive.enterprise.grandcentral.AugmentedGroundType"}]},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}] } {
  firrtl.module @MoreThanOneExtractGrandCentralAnnotation() {}
}

// -----

firrtl.circuit "NonGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64,
        name = "foo"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {
    %_vector = firrtl.verbatim.expr "???" : () -> !firrtl.vector<uint<2>, 1>
    %ref_vector = firrtl.ref.send %_vector : !firrtl.vector<uint<2>, 1>
    %vector = firrtl.ref.resolve %ref_vector : !firrtl.ref<vector<uint<2>, 1>>
    // expected-error @+1 {{'firrtl.node' op cannot be added to interface with id '0' because it is not a ground type}}
    %a = firrtl.node %vector {
      annotations = [
        {
          class = "sifive.enterprise.grandcentral.AugmentedGroundType",
          id = 1 : i64
        }
      ]
    } : !firrtl.vector<uint<2>, 1>
  }
  firrtl.module private @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "view"}
    ]} {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @NonGroundType() {
    firrtl.instance dut @DUT()
  }
}

// -----

// expected-error @+1 {{missing 'id' in root-level BundleType}}
firrtl.circuit "NonGroundType" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module @NonGroundType() {}
}

// -----

firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "View",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  firrtl.module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  firrtl.module private @Bar(in %a: !firrtl.uint<1>) {}
  firrtl.module private @DUT(in %a: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "View"}]} {
    // expected-error @+1 {{'firrtl.instance' op is marked as an interface element, but this should be impossible due to how the Chisel Grand Central API works}}
    %bar_a = firrtl.instance bar @Bar(in a: !firrtl.uint<1> [
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}])
    firrtl.connect %bar_a, %a : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @Foo() {
    %dut_a = firrtl.instance dut @DUT(in a: !firrtl.uint<1>)
  }
}

// -----

firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "View",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 1 : i64,
        name = "foo"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  firrtl.module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  firrtl.module private @DUT(in %a: !firrtl.uint<1>) attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "View"}]} {
    // expected-error @+1 {{'firrtl.mem' op is marked as an interface element, but this does not make sense (is there a scattering bug or do you have a malformed hand-crafted MLIR circuit?)}}
    %memory_b_r = firrtl.mem Undefined {
      annotations = [
        {a},
        {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
         id = 1 : i64}],
      depth = 16 : i64,
      name = "memory_b",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @Foo() {
    %dut_a = firrtl.instance dut @DUT(in a: !firrtl.uint<1>)
  }
}

// -----

// expected-error @+1 {{'firrtl.circuit' op has an AugmentedGroundType with 'id == 42' that does not have a scattered leaf to connect to in the circuit}}
firrtl.circuit "Foo" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Bar",
     elements = [
       {class = "sifive.enterprise.grandcentral.AugmentedGroundType",
        id = 42 : i64,
        name = "baz"}],
     id = 0 : i64},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]}  {
  firrtl.module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  firrtl.module private @DUT() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "View"}]} {
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @Foo() {
    firrtl.instance dut @DUT()
  }
}


// -----
// expected-error @+1 {{'firrtl.circuit' op contains a 'companion' with id '0', but does not contain a GrandCentral 'parent' with the same id}}
firrtl.circuit "multiInstance2" attributes {
  annotations = [
    {class = "sifive.enterprise.grandcentral.AugmentedBundleType",
     defName = "Foo",
     id = 0 : i64,
     name = "View"},
    {class = "sifive.enterprise.grandcentral.ExtractGrandCentralAnnotation",
     directory = "gct-dir",
     filename = "gct-dir/bindings.sv"}]} {
  firrtl.module private @View_companion() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
       defName = "Foo",
       id = 0 : i64,
       name = "View"}]} {}
  // expected-error @+1 {{'firrtl.module' op is marked as a GrandCentral 'parent', but it is instantiated more than once}}
  firrtl.module private @DUTE() attributes {
    annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "view"}
    ]} {
    %a = firrtl.wire : !firrtl.uint<2>
    %b = firrtl.wire : !firrtl.uint<4>
    firrtl.instance View_companion @View_companion()
  }
  firrtl.module @multiInstance2() {
    firrtl.instance dut sym @s1 @DUTE() // expected-note {{parent is instantiated here}}
    firrtl.instance dut1 sym @s2 @DUTE() // expected-note {{parent is instantiated here}}
  }
}
