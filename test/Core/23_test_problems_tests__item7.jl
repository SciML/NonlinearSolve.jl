using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (
    Broyden(),
    Broyden(; init_jacobian = Val(:true_jacobian)),
    Broyden(; update_rule = Val(:bad_broyden)),
    Broyden(; init_jacobian = Val(:true_jacobian), update_rule = Val(:bad_broyden)),
    SimpleBroyden(),
)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[2]] = [1, 5, 8, 11, 18]
broken_tests[alg_ops[4]] = [5, 6, 8, 11]
if Sys.isapple()
    broken_tests[alg_ops[1]] = [1, 5, 11]
    broken_tests[alg_ops[3]] = [1, 5, 6, 9, 11]
    if VERSION ≥ v"1.12"
        # Test #4 (Wood function) passes on v1.12+
        broken_tests[alg_ops[5]] = [1, 5, 11]
    elseif VERSION ≥ v"1.11-"
        broken_tests[alg_ops[5]] = [1, 4, 5, 11]
    else
        broken_tests[alg_ops[5]] = [1, 5, 11]
    end
else
    broken_tests[alg_ops[1]] = [1, 5, 11]
    broken_tests[alg_ops[3]] = [1, 5, 6, 9, 11]
    broken_tests[alg_ops[5]] = [1, 5, 11]
end

test_on_library(problems, dicts, alg_ops, broken_tests, 1.0e-3)
