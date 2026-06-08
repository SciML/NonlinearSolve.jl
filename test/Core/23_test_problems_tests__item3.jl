using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (SimpleHalley(; autodiff = AutoForwardDiff()),)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = [1, 5, 15, 16, 18]

test_on_library(problems, dicts, alg_ops, broken_tests)
