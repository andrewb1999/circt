// RUN: circt-opt --canonicalize %s | FileCheck %s
// RUN: circt-opt --export-verilog %s | FileCheck %s --check-prefix=VERILOG

// `comb.keep_tree` keeps an add or multiply the node of its tree it was
// written as: it is not merged into a user of its kind, absorbs no operand
// of its kind, an add does not become a multiply, and it is written with its
// parentheses. A scheduler that timed the tree op by op (an adder tree
// balanced to fit a cycle, a shift-and-add) keeps its shape in the Verilog,
// where `a + b + c` is grouped from the left: a chain of adders to a
// synthesis tool that does not rebalance it.

// CHECK-LABEL: hw.module @kept
// CHECK-NEXT:    %[[S:.+]] = comb.add %b, %c {comb.keep_tree} : i32
// CHECK-NEXT:    %[[T:.+]] = comb.add %a, %[[S]] {comb.keep_tree} : i32
// CHECK-NEXT:    %[[P:.+]] = comb.mul %b, %c {comb.keep_tree} : i32
// CHECK-NEXT:    %[[Q:.+]] = comb.mul %a, %[[P]] {comb.keep_tree} : i32
// CHECK-NEXT:    hw.output %[[T]], %[[Q]]
// VERILOG-LABEL: module kept(
// VERILOG:         assign y = a + (b + c);
// VERILOG-NEXT:    assign z = a * (b * c);
hw.module @kept(in %a : i32, in %b : i32, in %c : i32, out y : i32, out z : i32) {
  %0 = comb.add %b, %c {comb.keep_tree} : i32
  %1 = comb.add %a, %0 {comb.keep_tree} : i32
  %2 = comb.mul %b, %c {comb.keep_tree} : i32
  %3 = comb.mul %a, %2 {comb.keep_tree} : i32
  hw.output %1, %3 : i32, i32
}

// Either side marked keeps the two apart.
// CHECK-LABEL: hw.module @mixed
// CHECK-NEXT:    %[[S:.+]] = comb.add %b, %c : i32
// CHECK-NEXT:    %[[T:.+]] = comb.add %a, %[[S]] {comb.keep_tree} : i32
// CHECK-NEXT:    %[[U:.+]] = comb.add %b, %c {comb.keep_tree} : i32
// CHECK-NEXT:    %[[V:.+]] = comb.add %a, %[[U]] : i32
// CHECK-NEXT:    hw.output %[[T]], %[[V]]
hw.module @mixed(in %a : i32, in %b : i32, in %c : i32, out y : i32, out z : i32) {
  %0 = comb.add %b, %c : i32
  %1 = comb.add %a, %0 {comb.keep_tree} : i32
  %2 = comb.add %b, %c {comb.keep_tree} : i32
  %3 = comb.add %a, %2 : i32
  hw.output %1, %3 : i32, i32
}

// Unmarked, the tree is one add, and written left to right either way.
// CHECK-LABEL: hw.module @flattened
// CHECK-NEXT:    %[[S:.+]] = comb.add %a, %b, %c : i32
// CHECK-NEXT:    hw.output %[[S]]
// VERILOG-LABEL: module flattened(
// VERILOG:         assign y = a + b + c;
hw.module @flattened(in %a : i32, in %b : i32, in %c : i32, out y : i32) {
  %0 = comb.add %b, %c : i32
  %1 = comb.add %a, %0 : i32
  hw.output %1 : i32
}


// An add of a multiple of x and x is not folded into the multiply: the
// scheduler timed an add (the same holds for x + (x << c), a shift-and-add
// that would become x * (2^c + 1)).
// CHECK-LABEL: hw.module @mul_add
// CHECK:         %[[M:.+]] = comb.mul %a, %c5_i32 : i32
// CHECK-NEXT:    %[[T:.+]] = comb.add %a, %[[M]] {comb.keep_tree} : i32
// CHECK-NEXT:    %[[N:.+]] = comb.mul %a, %c6_i32 : i32
// CHECK-NEXT:    hw.output %[[T]], %[[N]]
hw.module @mul_add(in %a : i32, out y : i32, out z : i32) {
  %c5 = hw.constant 5 : i32
  %0 = comb.mul %a, %c5 : i32
  %1 = comb.add %a, %0 {comb.keep_tree} : i32
  %2 = comb.mul %a, %c5 : i32
  %3 = comb.add %a, %2 : i32
  hw.output %1, %3 : i32, i32
}
