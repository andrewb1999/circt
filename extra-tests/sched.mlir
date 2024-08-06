ssp.instance @minII_feasible of "ModuloProblem" [II<3>] {
  library {
    operator_type @loadA [latency<1>, limit<1>]
    operator_type @loadTmp [latency<0>]
    operator_type @storeTmp [latency<0>]
    operator_type @mul [latency<4>, limit<1>]
    operator_type @yield [latency<0>]
  }
  graph {
    %0 = operation<@loadA>()
    %1 = operation<@loadTmp>(@tmp [dist<1>])
    %2 = operation<@mul> @mul1(%0, %1)
    %3 = operation<@storeTmp> @tmp(%2)
    operation<@yield> @last(%0, %1, %2, %3)
  }
}
