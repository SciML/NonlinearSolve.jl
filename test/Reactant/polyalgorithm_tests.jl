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

endpoint_residual(u, p) = u .* (u .- 1)
shifted_residual(u, p) = u .- 1

@testset "Aliased polyalgorithm candidates: $first_code, $second_code" for (
        f, u0, first_code, second_code, first_u, second_u, expected_u, expected_code,
    ) in (
        (endpoint_residual, Float32[3], ReturnCode.Success, ReturnCode.Success, 0, 1, 0, ReturnCode.Success),
        (endpoint_residual, Float32[3], ReturnCode.MaxIters, ReturnCode.MaxIters, 2, 4, 2, ReturnCode.MaxIters),
        (endpoint_residual, Float32[3], ReturnCode.ExactSolutionLeft, ReturnCode.Success, 0, 1, 0, ReturnCode.ExactSolutionLeft),
        (endpoint_residual, Float32[3], ReturnCode.ExactSolutionRight, ReturnCode.Success, 0, 1, 0, ReturnCode.ExactSolutionRight),
        (endpoint_residual, Float32[3], ReturnCode.StalledSuccess, ReturnCode.Success, 0, 1, 1, ReturnCode.Success),
        (shifted_residual, Float32[2, 2], ReturnCode.Success, ReturnCode.Success, 1, 3, 1, ReturnCode.Success),
        (shifted_residual, Float32[2, 2], ReturnCode.MaxIters, ReturnCode.MaxIters, 1, 3, 1, ReturnCode.MaxIters),
    )
    alg = NonlinearSolvePolyAlgorithm(
        (
            AliasingCandidate(first_u, first_code), AliasingCandidate(second_u, second_code),
        )
    )
    run_candidates(u) = solve(
        NonlinearProblem(f, u), alg; alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = true)
    )
    u = copy(u0)
    ru = Reactant.to_rarray(copy(u))
    host = run_candidates(u)
    compiled = Reactant.@jit run_candidates(ru)
    for (sol, input) in ((host, u), (compiled, ru))
        @test Array(sol.u) == fill(Float32(expected_u), size(u0))
        @test Array(input) == fill(Float32(expected_u), size(u0))
        @test Array(sol.resid) == f(Array(sol.u), nothing)
        @test sol.retcode == expected_code
    end
end
