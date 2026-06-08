using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (RobustMultiNewton(), FastShortcutNonlinearPolyalg())

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = []
broken_tests[alg_ops[2]] = []

test_on_library(problems, dicts, alg_ops, broken_tests)
