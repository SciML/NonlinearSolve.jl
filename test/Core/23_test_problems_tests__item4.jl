using NonlinearSolve
include("setup_robustnesstesting.jl")

alg_ops = (
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Simple, subproblem = TrustRegionSubproblem.Dogleg),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan, subproblem = TrustRegionSubproblem.Dogleg),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei, subproblem = TrustRegionSubproblem.Dogleg),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan, subproblem = TrustRegionSubproblem.Dogleg),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin, subproblem = TrustRegionSubproblem.Dogleg),
    TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.NLsolve, subproblem = TrustRegionSubproblem.Dogleg),
    SimpleTrustRegion(),
    SimpleTrustRegion(; nlsolve_update_rule = Val(true)),
    TrustRegion(),
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
broken_tests[alg_ops[9]] = [11, 21]

test_on_library(problems, dicts, alg_ops, broken_tests)
