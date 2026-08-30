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
broken_tests[alg_ops[2]] = [1, 5, 11, 18]
broken_tests[alg_ops[4]] = [5, 6, 11]

# Problems #1 (Generalized Rosenbrock) and #8 (Brown almost linear) with bad_broyden +
# true_jacobian sit on a knife-edge: ulp-level differences in the Jacobian inverse
# initialization flip them between converging and not, so which side they land on is
# BLAS/CPU dependent. Skip rather than mark broken, since an unexpected pass would also
# error — both have failed in both directions across runners. See
# SciML/NonlinearSolve.jl#1083 and SciML/NonlinearSolve.jl#1096. Problem #8 with plain
# true_jacobian Broyden (alg #2) sits on the same knife-edge for problems #4 (Wood) and
# #8: dependency-level factorization changes make them converge on different runners.
skip_tests = Dict(alg => Int[] for alg in alg_ops)
skip_tests[alg_ops[2]] = [4, 8]
skip_tests[alg_ops[4]] = [1, 8]
if Sys.isapple()
    broken_tests[alg_ops[1]] = [1, 5, 11]
    broken_tests[alg_ops[3]] = [1, 6, 9, 11]
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
    broken_tests[alg_ops[3]] = [1, 6, 9, 11]
    broken_tests[alg_ops[5]] = [1, 5, 11]
end

test_on_library(problems, dicts, alg_ops, broken_tests, 1.0e-3; skip_tests)
