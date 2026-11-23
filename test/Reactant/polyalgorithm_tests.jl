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
