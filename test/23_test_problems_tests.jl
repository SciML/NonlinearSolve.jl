@testsetup module RobustnessTesting
using NonlinearSolve, LinearAlgebra, LinearSolve, NonlinearProblemLibrary, Test

problems = NonlinearProblemLibrary.problems
dicts = NonlinearProblemLibrary.dicts

function test_on_library(
        problems, dicts, alg_ops, broken_tests, ϵ = 1e-4; skip_tests = nothing)
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, copy(x))
        @testset "$idx: $(dict["title"]) | alg #$(alg_id)" for (alg_id, alg) in
                                                               enumerate(alg_ops)
            try
                sol = solve(nlprob, alg; maxiters = 10000)
                problem(res, sol.u, nothing)

                skip = skip_tests !== nothing && idx in skip_tests[alg]
                if skip
                    @test_skip norm(res, Inf) ≤ ϵ
                    continue
                end
                broken = idx in broken_tests[alg] ? true : false
                @test norm(res, Inf)≤ϵ broken=broken
            catch err
                @error err
                broken = idx in broken_tests[alg] ? true : false
                if broken
                    @test false broken=true
                else
                    @test 1 == 2
                end
            end
        end
    end
end

export test_on_library, problems, dicts
end

@testitem "23 Test Problems: PolyAlgorithms" setup=[RobustnessTesting] tags=[:nopre] begin
    alg_ops=(RobustMultiNewton(), FastShortcutNonlinearPolyalg())

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[]
    broken_tests[alg_ops[2]]=[]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: NewtonRaphson" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops=(
        NewtonRaphson(),
        SimpleNewtonRaphson()
    )

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[1]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: Halley" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops=(SimpleHalley(; autodiff = AutoForwardDiff()),)

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[1, 5, 15, 16, 18]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: TrustRegion" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops=(
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Simple),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.NLsolve),
        SimpleTrustRegion(),
        SimpleTrustRegion(; nlsolve_update_rule = Val(true))
    )

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[11, 21]
    broken_tests[alg_ops[2]]=[11, 21]
    broken_tests[alg_ops[3]]=[11, 21]
    broken_tests[alg_ops[4]]=[8, 11, 21]
    broken_tests[alg_ops[5]]=[21]
    broken_tests[alg_ops[6]]=[11, 21]
    broken_tests[alg_ops[7]]=[3, 15, 16, 21]
    broken_tests[alg_ops[8]]=[15, 16]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: LevenbergMarquardt" setup=[RobustnessTesting] tags=[:core] begin
    using LinearSolve

    alg_ops=(
        LevenbergMarquardt(),
        LevenbergMarquardt(; α_geodesic = 0.1),
        LevenbergMarquardt(; linsolve = CholeskyFactorization())
    )

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[11, 21]
    broken_tests[alg_ops[2]]=[11, 21]
    broken_tests[alg_ops[3]]=[11, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: DFSane" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops=(
        DFSane(),
        SimpleDFSane()
    )

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[1, 2, 3, 5, 21]
    if Sys.isapple()
        if VERSION≥v"1.11-"
            broken_tests[alg_ops[2]]=[1, 2, 3, 5, 6, 11, 21]
        else
            broken_tests[alg_ops[2]]=[1, 2, 3, 5, 6, 21]
        end
    else
        broken_tests[alg_ops[2]]=[1, 2, 3, 5, 6, 11, 21]
    end

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: Broyden" setup=[RobustnessTesting] tags=[:core] retries=3 begin
    alg_ops=(
        Broyden(),
        Broyden(; init_jacobian = Val(:true_jacobian)),
        Broyden(; update_rule = Val(:bad_broyden)),
        Broyden(; init_jacobian = Val(:true_jacobian), update_rule = Val(:bad_broyden)),
        SimpleBroyden()
    )

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[2]]=[1, 5, 8, 11, 18]
    broken_tests[alg_ops[4]]=[5, 6, 8, 11]
    if Sys.isapple()
        broken_tests[alg_ops[1]]=[1, 5, 11]
        broken_tests[alg_ops[3]]=[1, 5, 6, 9, 11]
        if VERSION≥v"1.11-"
            broken_tests[alg_ops[5]]=[1, 4, 5, 11]
        else
            broken_tests[alg_ops[5]]=[1, 5, 11]
        end
    else
        broken_tests[alg_ops[1]]=[1, 5, 11, 15]
        broken_tests[alg_ops[3]]=[1, 5, 6, 9, 11, 16]
        broken_tests[alg_ops[5]]=[1, 5, 11]
    end

    test_on_library(problems, dicts, alg_ops, broken_tests, 1e-3)
end

@testitem "23 Test Problems: Klement" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops=(
        Klement(),
        Klement(; init_jacobian = Val(:true_jacobian_diagonal)),
        SimpleKlement()
    )

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[1, 2, 4, 5, 11, 18, 22]
    broken_tests[alg_ops[2]]=[2, 4, 5, 7, 18, 22]
    broken_tests[alg_ops[3]]=[1, 2, 4, 5, 11, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: PseudoTransient" setup=[RobustnessTesting] tags=[:core] begin
    # PT relies on the root being a stable equilibrium for convergence, so it won't work on
    # most problems
    alg_ops=(PseudoTransient(),)

    broken_tests=Dict(alg=>Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]]=[1, 2, 3, 11, 15, 16]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end
