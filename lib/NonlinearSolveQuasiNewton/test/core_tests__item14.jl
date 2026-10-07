using NonlinearSolveQuasiNewton, ForwardDiff, SciMLBase, StaticArrays, Test
import CommonSolve

# `ad_calls[]` counts evaluations of `f` on dual numbers, i.e. the work of AD Jacobians.
const ad_calls = Ref(0)
function f!(du, u, p)
    eltype(u) <: ForwardDiff.Dual && (ad_calls[] += 1)
    du .= u .^ 3 .+ 2 .* u .- p
    return nothing
end
function f(u, p)
    eltype(u) <: ForwardDiff.Dual && (ad_calls[] += 1)
    return u .^ 3 .+ 2 .* u .- p
end

u0, p = collect(range(0.5, 1.5; length = 8)), 3.0
probs = (
    NonlinearProblem(NonlinearFunction{true}(f!), u0, p),
    NonlinearProblem(NonlinearFunction{false}(f), u0, p),
    NonlinearProblem(NonlinearFunction{false}(f), SVector{8}(u0), p),
)
algs = (
    Broyden(; init_jacobian = Val(:true_jacobian)),
    Klement(; init_jacobian = Val(:true_jacobian)),
    Klement(; init_jacobian = Val(:true_jacobian_diagonal)),
)

@testset "first step reuses the true Jacobian from init: $(alg.name)" for alg in algs
    for prob in probs
        cache = CommonSolve.init(prob, alg)
        calls, njacs = ad_calls[], cache.stats.njacs
        CommonSolve.step!(cache)
        @test ad_calls[] == calls
        @test cache.stats.njacs == njacs
        @test SciMLBase.successful_retcode(CommonSolve.solve!(cache))
    end

    cache = CommonSolve.init(probs[1], alg)
    cache.u .= 2 .* u0
    calls = ad_calls[]
    CommonSolve.step!(cache)
    @test ad_calls[] > calls

    for (u0_new, p_new) in ((2 .* u0, p), (u0, 5.0), (2 .* u0, 5.0))
        fresh = solve(remake(probs[1]; u0 = u0_new, p = p_new), alg)
        cache = CommonSolve.init(probs[1], alg)
        CommonSolve.solve!(cache)
        reinit!(cache, u0_new; p = p_new)
        calls = ad_calls[]
        CommonSolve.step!(cache)
        @test ad_calls[] > calls
        reinit!(cache, u0_new; p = p_new)
        reinit_sol = CommonSolve.solve!(cache)
        @test reinit_sol.u == fresh.u
        @test reinit_sol.retcode == fresh.retcode
        @test reinit_sol.stats.njacs == fresh.stats.njacs
    end
end
