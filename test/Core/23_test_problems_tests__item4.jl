using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Simple),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.NLsolve),
    SimpleTrustRegion(),
    SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
)

broken_tests = Dict(alg => Int[] for alg in alg_ops)
broken_tests[alg_ops[1]] = [11, 21]
broken_tests[alg_ops[2]] = [11, 21]
broken_tests[alg_ops[3]] = [11, 21]
broken_tests[alg_ops[4]] = [8, 11, 21]
broken_tests[alg_ops[5]] = [21]
broken_tests[alg_ops[6]] = [11, 21]
broken_tests[alg_ops[7]] = [3, 15, 16, 21]
broken_tests[alg_ops[8]] = [15, 16]

test_on_library(problems, dicts, alg_ops, broken_tests)
