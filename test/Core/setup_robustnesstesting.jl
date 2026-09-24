using NonlinearSolve, LinearAlgebra, LinearSolve, NonlinearProblemLibrary, Test

problems = NonlinearProblemLibrary.problems
dicts = NonlinearProblemLibrary.dicts

function test_on_library(
        problems, dicts, alg_ops, broken_tests, ϵ = 1.0e-4; skip_tests = nothing
    )
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
                @test norm(res, Inf) ≤ ϵ broken = broken
            catch err
                @error err
                broken = idx in broken_tests[alg] ? true : false
                if broken
                    @test false broken = true
                else
                    @test 1 == 2
                end
            end
        end
    end
    return
end

export test_on_library, problems, dicts
