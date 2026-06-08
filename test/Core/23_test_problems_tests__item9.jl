using NonlinearSolve
include("setup_robustnesstesting.jl")

# PT relies on the root being a stable equilibrium for convergence, so it won't work on
# most problems
alg_ops = (PseudoTransient(),)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = [1, 2, 3, 11, 15, 16]

test_on_library(problems, dicts, alg_ops, broken_tests)
