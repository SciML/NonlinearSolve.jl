using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (
    DFSane(),
    SimpleDFSane(),
)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = [1, 2, 3, 5, 21]
if Sys.isapple()
    if VERSION ≥ v"1.11-"
        broken_tests[alg_ops[2]] = [1, 2, 3, 5, 6, 11, 21]
    else
        broken_tests[alg_ops[2]] = [1, 2, 3, 5, 6, 21]
    end
else
    broken_tests[alg_ops[2]] = [1, 2, 3, 5, 6, 11, 21]
end

test_on_library(problems, dicts, alg_ops, broken_tests)
