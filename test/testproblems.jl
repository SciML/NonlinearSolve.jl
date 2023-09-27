using NonlinearSolve, LinearAlgebra, NonlinearProblemLibrary

problems = NonlinearProblemLibrary.problems
dicts = NonlinearProblemLibrary.dicts

function test_on_library(problems, dicts, alg_ops, broken_tests, ϵ = 1e-5)
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, x)
        @testset "$(dict["title"])" begin
            for alg in alg_ops
                sol = solve(nlprob, alg, abstol = 1e-15, reltol = 1e-15)
                problem(res, sol.u, nothing)
                broken = idx in broken_tests[alg] ? true : false
                @test norm(res)<=ϵ skip=broken
            end
        end
    end
end

function print_results_on_library(problems, dicts, alg_ops, names)
    results = []
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, x)
        push!(results, Dict())
        for (alg, name) in zip(alg_ops, names)
            sol = solve(nlprob, alg, abstol = 1e-15, reltol = 1e-15)
            problem(res, sol.u, nothing)
            results[idx][name] = norm(res)
        end
    end

    i_str = i_str = rpad("nr", 3, " ")
    title_str = rpad("Problem", 50, " ")
    n_str = rpad("n", 5, " ")
    norm_str = prod(rpad(name, 20, " ") for name in names)
    println("$i_str $title_str $n_str $norm_str")

    for (i, dict) in enumerate(dicts)
        local i_str = rpad(string(i), 3, " ")
        local title_str = rpad(dict["title"], 50, " ")
        local n_str = rpad(string(dict["n"]), 5, " ")
        local norm_str = ""
        for name in names
            norm_str *= rpad(string(trunc(results[i][name]; sigdigits = 5)), 20, " ")
        end
        println("$i_str $title_str $n_str $norm_str")
    end
end

function detect_broken_tests(problems, dicts, alg_ops, ϵ = 1e-5)
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    for (idx, (problem, dict)) in enumerate(zip(problems, dicts))
        x = dict["start"]
        res = similar(x)
        nlprob = NonlinearProblem(problem, x)
        for alg in alg_ops
            sol = solve(nlprob, alg, abstol = 1e-15, reltol = 1e-15)
            problem(res, sol.u, nothing)
            if norm(res) > ϵ
                push!(broken_tests[alg], idx)
            end
        end
    end
    return broken_tests
end

# NewtonRaphson
@testset "NewtonRaphson test problem library" begin
    alg_ops = (NewtonRaphson(),)

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 6]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

@testset "TrustRegion test problem library" begin
    alg_ops = (TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Simple),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Fan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Hei),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Yuan),
        TrustRegion(; radius_update_scheme = RadiusUpdateSchemes.Bastin))

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 21, 22]
    broken_tests[alg_ops[2]] = [1, 2, 3, 4, 5, 6, 7, 9, 15, 16, 21, 22]
    broken_tests[alg_ops[3]] = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 18, 19, 21, 22]
    broken_tests[alg_ops[4]] = [1, 2, 3, 4, 5, 6, 8, 9, 15, 16, 21, 22]
    broken_tests[alg_ops[5]] = [1, 2, 3, 5, 6, 9, 11, 15, 16, 18, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end

# LevenbergMarquardt seems to still be buggy    
@testset "TrustRegion test problem library" begin
    alg_ops = (LevenbergMarquardt(),)

    # dictionary with indices of test problems where method does not converge to small residual
    broken_tests = Dict(alg => Int[] for alg in alg_ops)
    broken_tests[alg_ops[1]] = [1, 2, 4, 5, 6, 7, 8, 11, 15, 16, 17, 18, 19, 21, 22]

    test_on_library(problems, dicts, alg_ops, broken_tests)
end
