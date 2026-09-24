using NonlinearSolve
include("setup_robustnesstesting.jl")

using LinearSolve

alg_ops = (
    LevenbergMarquardt(),
    LevenbergMarquardt(; α_geodesic = 0.1),
    LevenbergMarquardt(; linsolve = CholeskyFactorization()),
)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = [11, 21]
broken_tests[alg_ops[2]] = [11, 21]
broken_tests[alg_ops[3]] = [11, 21]

test_on_library(problems, dicts, alg_ops, broken_tests)
