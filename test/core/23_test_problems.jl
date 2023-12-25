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
                    sol = solve(nlprob, alg; maxiters = 10000)
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

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "TrustRegion 23 Test Problems" begin
    alg_ops = (TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Simple),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.NLsolve))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [11, 21]
    broken_tests[alg_ops[2]] = [11, 21]
    broken_tests[alg_ops[3]] = [11, 21]
    broken_tests[alg_ops[4]] = [11, 21]
    broken_tests[alg_ops[5]] = [21]
    broken_tests[alg_ops[6]] = [21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "LevenbergMarquardt 23 Test Problems" begin
    alg_ops = (LevenbergMarquardt(), LevenbergMarquardt(; α_geodesic = 0.1),
        LevenbergMarquardt(; linsolve = CholeskyFactorization()))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [11, 21]
    broken_tests[alg_ops[2]] = [11, 21]
    broken_tests[alg_ops[3]] = [11, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "DFSane 23 Test Problems" begin
    alg_ops = (DFSane(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 3, 5, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "Broyden 23 Test Problems" begin
    alg_ops = (Broyden(), Broyden(; init_jacobian = Val(:true_jacobian)),
        Broyden(; update_rule = Val(:bad_broyden)),
        Broyden(; init_jacobian = Val(:true_jacobian), update_rule = Val(:bad_broyden)),
        Broyden(; update_rule = Val(:diagonal)),
        Broyden(; init_jacobian = Val(:true_jacobian), update_rule = Val(:diagonal)))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 5, 11]
    broken_tests[alg_ops[2]] = [1, 5, 8, 11, 18]
    broken_tests[alg_ops[3]] = [1, 5, 9, 11]
    broken_tests[alg_ops[4]] = [1, 5, 6, 8, 11]
    broken_tests[alg_ops[5]] = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 21]
    broken_tests[alg_ops[6]] = [2, 3, 4, 5, 6, 8, 9, 11, 12, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "Klement 23 Test Problems" begin
    alg_ops = (Klement(), Klement(; init_jacobian = Val(:true_jacobian_diagonal)))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 11, 22]
    broken_tests[alg_ops[2]] = [2, 4, 5, 7, 18, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "PseudoTransient 23 Test Problems" begin
    alg_ops = (PseudoTransient(; alpha_initial = 10.0),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 9, 18, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end
