using SimpleNonlinearSolve, LinearAlgebra, NonlinearProblemLibrary, Test

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
                    sol = solve(nlprob, alg;
                        termination_condition = AbsNormTerminationMode())
                    problem(res, sol.u, nothing)

                    skip = skip_tests !== nothing && idx in skip_tests[alg]
                    if skip
                        @test_skip norm(res) ≤ ϵ
                        continue
                    end
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

@testset "SimpleNewtonRaphson 23 Test Problems" begin
    alg_ops = (SimpleNewtonRaphson(),)

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [6]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "SimpleTrustRegion 23 Test Problems" begin
    alg_ops = (SimpleTrustRegion(),)

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [3, 6, 15, 16, 21]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "SimpleDFSane 23 Test Problems" begin
    alg_ops = (SimpleDFSane(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 13, 15, 16, 17, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "SimpleBroyden 23 Test Problems" begin
    alg_ops = (SimpleBroyden(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 6, 11, 12, 13, 14]

    skip_tests = Dict(alg => Int[] for alg in alg_ops)
    skip_tests[alg_ops[1]] = [22]

    test_on_library(problems, dicts, alg_ops, broken_tests; skip_tests)
end

@testset "SimpleKlement 23 Test Problems" begin
    alg_ops = (SimpleKlement(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 19, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end
