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
broken_tests[alg_ops[4]] = [5, 6, 11]

# Problem #1 (Generalized Rosenbrock) with bad_broyden + true_jacobian sits on a
# knife-edge: ulp-level differences in the Jacobian inverse initialization flip it
# between converging and not, so it randomly passes and fails. Skip rather than mark
# broken, since an unexpected pass would also error. See SciML/NonlinearSolve.jl#1083.
skip_tests = Dict(alg => Int[] for alg in alg_ops)
skip_tests[alg_ops[4]] = [1]
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

test_on_library(problems, dicts, alg_ops, broken_tests, 1.0e-3; skip_tests)
