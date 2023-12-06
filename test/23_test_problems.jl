using NonlinearSolve, LinearAlgebra, LinearSolve, NonlinearProblemLibrary, Test

problems = NonlinearProblemLibrary.problems
dicts = NonlinearProblemLibrary.dicts

function test_on_library(problems, dicts, alg_ops, broken_tests, ϵ = 1e-4;
        skip_tests = nothing)
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, copy(x))
        @testset "$idx: $(dict["title"])" begin
            for alg in alg_ops
                try
                    sol = solve(nlprob, alg; maxiters = 10000,
                        termination_condition = AbsNormTerminationMode())
                    problem(res, sol.u, nothing)

                    skip = skip_tests !== nothing && idx in skip_tests[alg]
                    if skip
                        @test_skip norm(res) ≤ ϵ
                        continue
                    end
                    broken = idx in broken_tests[alg] ? true : false
                    @test norm(res)≤ϵ broken=broken
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
    broken_tests[alg_ops[3]] = [6, 11, 21]
    broken_tests[alg_ops[4]] = [6, 11, 21]
    broken_tests[alg_ops[5]] = [6, 21]
    broken_tests[alg_ops[6]] = [6, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "LevenbergMarquardt 23 Test Problems" begin
    alg_ops = (LevenbergMarquardt(), LevenbergMarquardt(; α_geodesic = 0.1),
        LevenbergMarquardt(; linsolve = CholeskyFactorization()))

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [3, 6, 11, 17, 21]
    broken_tests[alg_ops[2]] = [3, 6, 11, 17, 21]
    broken_tests[alg_ops[3]] = [6, 11, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "DFSane 23 Test Problems" begin
    alg_ops = (DFSane(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 3, 5, 6, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "GeneralBroyden 23 Test Problems" begin
    alg_ops = (GeneralBroyden(),
        GeneralBroyden(; init_jacobian = Val(:true_jacobian)),
        GeneralBroyden(; update_rule = Val(:bad_broyden)),
        GeneralBroyden(; init_jacobian = Val(:true_jacobian),
            update_rule = Val(:bad_broyden)))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 5, 6, 11]
    broken_tests[alg_ops[2]] = [1, 5, 6, 8, 11, 18]
    broken_tests[alg_ops[3]] = [1, 4, 5, 6, 9, 11]
    broken_tests[alg_ops[4]] = [1, 5, 6, 8, 11]

    skip_tests = Dict(alg => Int[] for alg in alg_ops)
    skip_tests[alg_ops[1]] = [2, 22]
    skip_tests[alg_ops[2]] = [2, 22]
    skip_tests[alg_ops[3]] = [2, 22]
    skip_tests[alg_ops[4]] = [2, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests; skip_tests)
end

@testset "GeneralKlement 23 Test Problems" begin
    alg_ops = (GeneralKlement(),
        GeneralKlement(; init_jacobian = Val(:true_jacobian)),
        GeneralKlement(; init_jacobian = Val(:true_jacobian_diagonal)))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 6, 11, 22]
    broken_tests[alg_ops[2]] = [1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 17, 21, 22, 23]
    broken_tests[alg_ops[3]] = [2, 4, 5, 6, 7, 18, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "PseudoTransient 23 Test Problems" begin
    alg_ops = (PseudoTransient(; alpha_initial = 10.0),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 6, 9, 18, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end
