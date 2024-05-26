@testsetup module RobustnessTesting
using LinearAlgebra, NonlinearProblemLibrary, DiffEqBase, Test

problems = NonlinearProblemLibrary.problems
dicts = NonlinearProblemLibrary.dicts

function test_on_library(
        problems, dicts, alg_ops, broken_tests, ϵ = 1e-4; skip_tests = nothing)
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, copy(x))
        @testset "$idx: $(dict["title"])" begin
            for alg in alg_ops
                try
                    sol = solve(
                        nlprob, alg; termination_condition = AbsNormTerminationMode())
                    problem(res, sol.u, nothing)

                    skip = skip_tests !== nothing && idx in skip_tests[alg]
                    if skip
                        @test_skip norm(res) ≤ ϵ
                        continue
                    end
                    broken = idx in broken_tests[alg] ? true : false
                    @test norm(res)≤ϵ broken=broken
                catch e
                    @error e
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

export problems, dicts, test_on_library
end

@testitem "23 Test Problems: SimpleNewtonRaphson" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops = (SimpleNewtonRaphson(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = []

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: SimpleHalley" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops = (SimpleHalley(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    if Sys.isapple()
        broken_tests[alg_ops[1]] = [1, 5, 11, 15, 16, 18]
    else
        broken_tests[alg_ops[1]] = [1, 5, 15, 16, 18]
    end

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: SimpleTrustRegion" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops = (SimpleTrustRegion(), SimpleTrustRegion(; nlsolve_update_rule = Val(true)))

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [3, 15, 16, 21]
    broken_tests[alg_ops[2]] = [15, 16]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: SimpleDFSane" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops = (SimpleDFSane(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    if Sys.isapple()
        broken_tests[alg_ops[1]] = [1, 2, 3, 5, 6, 21]
    else
        broken_tests[alg_ops[1]] = [1, 2, 3, 4, 5, 6, 11, 21]
    end

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: SimpleBroyden" retries=5 setup=[RobustnessTesting] tags=[:core] begin
    alg_ops = (SimpleBroyden(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 5, 11]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testitem "23 Test Problems: SimpleKlement" setup=[RobustnessTesting] tags=[:core] begin
    alg_ops = (SimpleKlement(),)

    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 11, 12, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end
