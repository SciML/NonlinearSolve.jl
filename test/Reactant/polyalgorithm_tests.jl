struct FixedCandidate <: SciMLBase.AbstractNonlinearAlgorithm
    value::Float32
end

function SciMLBase.__solve(
        prob::NonlinearLeastSquaresProblem, alg::FixedCandidate, args...; kwargs...
    )
    u = one.(prob.u0) .* alg.value
    return NonlinearSolveBase.build_nonlinear_solution(
        prob, alg, u, prob.f(u, prob.p); retcode = NonlinearSolveBase.maybe_traced(ReturnCode.MaxIters)
    )
end

@testset "All-failed least-squares candidates at extreme residual scales" begin
    function rank_candidates(u, p)
        return solve(
            NonlinearLeastSquaresProblem((u, p) -> u .* p, u, p),
            NonlinearSolvePolyAlgorithm((FixedCandidate(2), FixedCandidate(1)))
        )
    end
    u = ones(Float32, 1)
    for p in (Float32[1.0f20], Float32[1.0f-30])
        host = rank_candidates(u, p)
        compiled = Reactant.@jit rank_candidates(Reactant.to_rarray(u), Reactant.to_rarray(p))
        for sol in (host, compiled)
            @test sol.retcode == ReturnCode.MaxIters
            @test Array(sol.u) == u
            @test Array(sol.resid) == p
        end
    end
end

struct AliasingCandidate <: SciMLBase.AbstractNonlinearAlgorithm
    value::Float32
    retcode::ReturnCode.T
end

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::AliasingCandidate, args...; alias_u0, kwargs...
    )
    u = alias_u0 ? prob.u0 : copy(prob.u0)
    u .= alg.value
    return NonlinearSolveBase.build_nonlinear_solution(
        prob, alg, u, prob.f(u, prob.p); retcode = NonlinearSolveBase.maybe_traced(alg.retcode)
    )
end

@testset "Aliased polyalgorithm candidates: $first_code, $second_code" for (
        first_code, second_code, first_u, second_u, expected_u, expected_code,
    ) in (
        (ReturnCode.Success, ReturnCode.Success, 0, 1, 0, ReturnCode.Success),
        (ReturnCode.MaxIters, ReturnCode.MaxIters, 2, 4, 2, ReturnCode.MaxIters),
        (ReturnCode.ExactSolutionLeft, ReturnCode.Success, 0, 1, 0, ReturnCode.ExactSolutionLeft),
        (ReturnCode.ExactSolutionRight, ReturnCode.Success, 0, 1, 0, ReturnCode.ExactSolutionRight),
        (ReturnCode.StalledSuccess, ReturnCode.Success, 0, 1, 1, ReturnCode.Success),
    )
    f(u, p) = u .* (u .- 1)
    alg = NonlinearSolvePolyAlgorithm(
        (
            AliasingCandidate(first_u, first_code), AliasingCandidate(second_u, second_code),
        )
    )
    run_candidates(u) = solve(
        NonlinearProblem(f, u), alg; alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true)
    )
    u = Float32[3]
    ru = Reactant.to_rarray(copy(u))
    host = run_candidates(u)
    compiled = Reactant.@jit run_candidates(ru)
    for (sol, input) in ((host, u), (compiled, ru))
        @test Array(sol.u) == Float32[expected_u]
        @test Array(input) == Float32[expected_u]
        @test Array(sol.resid) == f(Array(sol.u), nothing)
        @test sol.retcode == expected_code
    end
end
