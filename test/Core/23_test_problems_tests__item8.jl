using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (
    Klement(),
    Klement(; init_jacobian = Val(:true_jacobian_diagonal)),
    SimpleKlement(),
)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = [1, 2, 4, 5, 11, 18, 22]
broken_tests[alg_ops[2]] = [2, 4, 5, 7, 18, 22]
broken_tests[alg_ops[3]] = [1, 2, 4, 5, 11, 22]

test_on_library(problems, dicts, alg_ops, broken_tests)
