using NonlinearSolve, LinearAlgebra, LinearSolve, NonlinearProblemLibrary, Test

problems = NonlinearProblemLibrary.problems
dicts = NonlinearProblemLibrary.dicts

function test_on_library(problems, dicts, alg_ops, broken_tests, ϵ = 1e-4)
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, x)
        @testset "$idx: $(dict["title"])" begin
            for alg in alg_ops
                try
                    sol = solve(nlprob, alg, abstol = 1e-18, reltol = 1e-18)
                    problem(res, sol.u, nothing)
                    broken = idx in broken_tests[alg] ? true : false
                    @test norm(res)≤ϵ broken=broken
                catch
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
end

@testset "NewtonRaphson 23 Test Problems" begin
    alg_ops = (NewtonRaphson(),)

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 6]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "TrustRegion 23 Test Problems" begin
    alg_ops = (TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Simple),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.NLsolve))

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [6, 11, 21]
    broken_tests[alg_ops[2]] = [6, 11, 21]
    broken_tests[alg_ops[3]] = [1, 6, 11, 12, 15, 16, 21]
    broken_tests[alg_ops[4]] = [1, 6, 8, 11, 16, 21, 22]
    broken_tests[alg_ops[5]] = [6, 21]
    broken_tests[alg_ops[6]] = [6, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "LevenbergMarquardt 23 Test Problems" begin
    alg_ops = (LevenbergMarquardt(; linsolve = NormalCholeskyFactorization()),
        LevenbergMarquardt(; α_geodesic = 0.1, linsolve = NormalCholeskyFactorization()))

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [3, 6, 11, 21]
    broken_tests[alg_ops[2]] = [3, 6, 11, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "DFSane 23 Test Problems" begin
    alg_ops = (DFSane(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 3, 5, 6, 8, 12, 13, 14, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "GeneralBroyden 23 Test Problems" begin
    alg_ops = (GeneralBroyden(),
        GeneralBroyden(; linesearch = LiFukushimaLineSearch(; beta = 0.1)),
        GeneralBroyden(; linesearch = BackTracking()))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 3, 4, 5, 6, 11, 12, 13, 14, 21]
    broken_tests[alg_ops[2]] = [1, 2, 3, 4, 5, 6, 9, 11, 13, 22]
    broken_tests[alg_ops[3]] = [1, 2, 4, 5, 6, 11, 12, 13, 14, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "GeneralKlement 23 Test Problems" begin
    alg_ops = (GeneralKlement(),
        GeneralKlement(; linesearch = BackTracking()),
        GeneralKlement(; linesearch = HagerZhang()))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 6, 7, 11, 13, 22]
    broken_tests[alg_ops[2]] = [1, 2, 4, 5, 6, 7, 11, 12, 13, 22]
    broken_tests[alg_ops[3]] = [1, 2, 4, 5, 6, 8, 11, 12, 13, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

# NOTE: Not adding LimitedMemoryBroyden here since it fails on most of the preoblems
