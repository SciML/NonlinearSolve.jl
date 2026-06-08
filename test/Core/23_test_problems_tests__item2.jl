using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (
    NewtonRaphson(),
    SimpleNewtonRaphson(),
)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
# Generalized Rosenbrock function regressed when LinearSolve defaulted
# residualsafety to false (SciML/LinearSolve.jl@93b2a2d3, 2026-04-08).
broken_tests[alg_ops[1]] = [1]

test_on_library(problems, dicts, alg_ops, broken_tests)
