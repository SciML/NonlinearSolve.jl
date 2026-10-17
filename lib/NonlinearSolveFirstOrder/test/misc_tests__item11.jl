using ADTypes, ForwardDiff, LinearAlgebra, LinearSolve, NonlinearSolveFirstOrder, SciMLBase
using SparseArrays, SparseConnectivityTracer, SparseMatrixColorings, StaticArrays
import CommonSolve

# `ad_calls[]` counts evaluations of `f` on dual numbers, i.e. the work of AD Jacobians.
function counted_residual(ad_calls)
    function f!(du, u, p)
        eltype(u) <: ForwardDiff.Dual && (ad_calls[] += 1)
        du .= u .^ 3 .+ 2 .* u .- p
        return nothing
    end
    function f(u, p)
        eltype(u) <: ForwardDiff.Dual && (ad_calls[] += 1)
        return u .^ 3 .+ 2 .* u .- p
    end
    return f!, f
end

# Number of AD evaluations and Jacobian counts spent by the first `step!` after `init`.
function first_step_cost(prob, alg, ad_calls)
    cache = CommonSolve.init(prob, alg)
    calls, njacs = ad_calls[], cache.stats.njacs
    CommonSolve.step!(cache)
    return ad_calls[] - calls, cache.stats.njacs - njacs, cache
end

ad_calls = Ref(0)
f!, f = counted_residual(ad_calls)
u0, p = collect(range(0.5, 1.5; length = 8)), 3.0
algs = (NewtonRaphson(), TrustRegion(), LevenbergMarquardt(), PseudoTransient())
dense_probs = (
    NonlinearProblem(NonlinearFunction{true}(f!), u0, p),
    NonlinearProblem(NonlinearFunction{false}(f), u0, p),
    NonlinearProblem(NonlinearFunction{false}(f), SVector{8}(u0), p),
)
sparse_prob = NonlinearProblem(
    NonlinearFunction{true}(f!; sparsity = TracerSparsityDetector()), u0, p
)
cases = (
    ((alg, prob) for alg in algs for prob in (dense_probs..., sparse_prob))...,
    (
        (NewtonRaphson(; autodiff = AutoForwardDiff(; chunksize = 2)), prob)
            for prob in dense_probs
    )...,
)

@testset "first step reuses the Jacobian from init" begin
    for (alg, prob) in cases
        calls, njacs, cache = first_step_cost(prob, alg, ad_calls)
        @test calls == 0
        @test njacs == 0
        @test SciMLBase.successful_retcode(CommonSolve.solve!(cache))
    end
end

@testset "Jacobians not evaluated at init are evaluated on the first step" begin
    prototype = sparse(Diagonal(ones(length(u0))))
    prototype_prob = NonlinearProblem(
        NonlinearFunction{true}(f!; jac_prototype = prototype), u0, p
    )
    calls, njacs, _ = first_step_cost(prototype_prob, NewtonRaphson(), ad_calls)
    @test calls > 0
    @test njacs == 1

    jac_calls = Ref(0)
    jac!(J, u, p) = (jac_calls[] += 1; J .= Diagonal(3 .* u .^ 2 .+ 2); nothing)
    jac_prob = NonlinearProblem(NonlinearFunction{true}(f!; jac = jac!), u0, p)
    cache = CommonSolve.init(jac_prob, NewtonRaphson())
    @test jac_calls[] == 0
    CommonSolve.step!(cache)
    @test jac_calls[] == 1

    # Matrix-free Krylov solves differentiate through JVPs, never a Jacobian.
    _, njacs, _ = first_step_cost(dense_probs[1], NewtonRaphson(; linsolve = KrylovJL_GMRES()), ad_calls)
    @test njacs == 0
end

@testset "a moved or reinitialized state invalidates the Jacobian from init" begin
    cache = CommonSolve.init(dense_probs[1], NewtonRaphson())
    cache.u .= 2 .* u0
    calls = ad_calls[]
    CommonSolve.step!(cache)
    @test ad_calls[] > calls

    for alg in algs, (u0_new, p_new) in ((2 .* u0, p), (u0, 5.0), (2 .* u0, 5.0))
        fresh = solve(remake(dense_probs[1]; u0 = u0_new, p = p_new), alg)
        cache = CommonSolve.init(dense_probs[1], alg)
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
