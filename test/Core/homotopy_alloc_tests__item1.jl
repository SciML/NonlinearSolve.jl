using NonlinearSolve

using CommonSolve
using SciMLBase

import NonlinearSolveBase as NLB

# The continuation drivers wrap their residual in a `FixLambda` (sweep) / `AugmentedHomotopy`
# / `HomotopyResidual` (arclength) *functor* and build the inner `NonlinearFunction` at the
# `AutoSpecialize` default. `AutoSpecialize` wraps an in-place residual in a chunksize-1
# `FunctionWrappersWrapper` whose fixed dual signatures let `init(inner_prob, inner)` infer to
# a concrete cache, so the driver's reused inner cache and every per-step `solve!` are
# allocation-free. That wrapper must be constructed with its arglist types bound as type
# parameters (`_make_fww_iip`, ForwardDiff ext, mirroring DiffEqBase's `_make_fww`); the naive
# `FunctionWrappersWrapper(f, argtypes, rettypes)` convenience constructor's `map` closure
# instead widens a functor's wrapper — and every cache built from the wrapped problem — to
# `Any`, boxing every field read of every interior solve. These tests pin that guarantee.

Hiip!(du, u, p, λ) = (du[1] = (1 - λ) * (u[1] - p[1]) + λ * (u[1]^2 - p[1]); nothing)
prob = HomotopyProblem(Hiip!, [4.0], [4.0])

# --- the wrapped inner problem infers a concrete cache (the property that makes stepping
# allocation-free); a functor residual must not widen `init` to `Any` ---
fixλ = NLB.FixLambda(prob.f, 0.3)
fλ = NLB._sweep_nonlinear_function(Val(true), prob.f, fixλ)
@test SciMLBase.specialization(fλ) === SciMLBase.AutoSpecialize
inner_prob = NonlinearProblem(fλ, [4.0], [4.0])
inner_rts = Base.return_types(SciMLBase.init, (typeof(inner_prob), typeof(NewtonRaphson())))
@test length(inner_rts) == 1
@test inner_rts[1] !== Any
@test isconcretetype(inner_rts[1])

# --- the wrapper itself is concretely inferred for a functor (the direct unit under test) ---
wrapped = NLB.maybe_wrap_nonlinear_f(inner_prob)
@test NLB.is_fw_wrapped(wrapped)
@test isconcretetype(only(Base.return_types(NLB.maybe_wrap_nonlinear_f, (typeof(inner_prob),))))

# Measure the cache-only solve independently of the end-to-end continuation slopes below.
function cache_only_solve_allocations(prob)
    u0 = copy(prob.u0)
    cache = SciMLBase.init(prob, NewtonRaphson())
    result = NLB._solve_without_solution!(cache)
    SciMLBase.reinit!(cache, u0)
    NLB._solve_without_solution!(cache)
    SciMLBase.reinit!(cache, u0)
    GC.gc()
    allocs = @allocated NLB._solve_without_solution!(cache)
    return result, result === cache, allocs
end

cache_result, same_cache, cache_solve_allocs = cache_only_solve_allocations(inner_prob)
@test cache_result isa NLB.AbstractNonlinearSolveCache
@test same_cache
@test SciMLBase.successful_retcode(cache_result.retcode)
VERSION >= v"1.11" && @test cache_solve_allocs == 0

function cached_sweep_retcode!(cache, u0, p, abstol)
    SciMLBase.reinit!(cache, u0; p, abstol)
    return CommonSolve.solve!(cache).retcode
end

function cached_sweep_allocations(prob)
    alg = HomotopySweep(; inner = NewtonRaphson(), adaptive = false, nsteps = 10)
    cache = SciMLBase.init(prob, alg; abstol = 0.0)
    abstol = 1.0e-10
    cached_sweep_retcode!(cache, prob.u0, prob.p, abstol)
    cached_sweep_retcode!(cache, prob.u0, prob.p, abstol)
    GC.gc()
    allocs = @allocated cached_sweep_retcode!(cache, prob.u0, prob.p, abstol)
    return cache, allocs
end

sweep_cache, sweep_cache_allocs = cached_sweep_allocations(prob)
@test SciMLBase.successful_retcode(CommonSolve.solve!(sweep_cache))
VERSION >= v"1.11" && @test sweep_cache_allocs == 0

# --- end-to-end per-step allocation with a clean `NewtonRaphson` inner. The slope between two
# fixed-step runs cancels compile/one-time-init cost, leaving pure per-step allocation. Gated
# on Julia 1.11+: the LinearSolve factorization-workspace reuse that removes the last
# per-step bytes is only active there (v1.10 is allowed to allocate the pivot vector). ---
nbig = 50
function Hbig!(r, u, p, λ)
    n = length(u)
    for i in 1:n
        acc = u[i]
        i > 1 && (acc += 0.25 * u[i - 1])
        i < n && (acc += 0.25 * u[i + 1])
        r[i] = acc + λ * u[i]^3 - p.c[i]
    end
    return nothing
end
cbig = [1.0 + 0.25 * ((i > 1) + (i < nbig)) + 1.0 for i in 1:nbig]
prob_big = HomotopyProblem(Hbig!, ones(nbig), (c = cbig,); λspan = (0.0, 1.0))

function sweep_per_step(prob; N1 = 50, N2 = 250)
    mk = M -> HomotopySweep(; inner = NewtonRaphson(), adaptive = false, nsteps = M)
    solve(prob, mk(5))
    solve(prob, mk(5))
    GC.gc()
    a1 = @allocated solve(prob, mk(N1))
    GC.gc()
    a2 = @allocated solve(prob, mk(N2))
    return (a2 - a1) / (N2 - N1)
end

function arclength_per_step(prob; N1 = 50, N2 = 250)
    mk = M -> ArcLengthContinuation(;
        inner = NewtonRaphson(), adaptive = false, initial_step_factor = 1.0e-4, maxsteps = M
    )
    solve(prob, mk(5))
    solve(prob, mk(5))
    GC.gc()
    a1 = @allocated solve(prob, mk(N1))
    GC.gc()
    a2 = @allocated solve(prob, mk(N2))
    return (a2 - a1) / (N2 - N1)
end

if VERSION >= v"1.11"
    # The pre-fix functor-boxing cost 96 B/step (sweep) / 320 B/step (arclength); with the
    # concrete wrapper both are 0. A small nonzero budget absorbs allocator noise while still
    # failing loudly if the `Any`-widening regresses.
    @test sweep_per_step(prob_big) < 48
    @test arclength_per_step(prob_big) < 48
end

# --- the fixed-step measurement configuration must not change the answer ---
@test SciMLBase.successful_retcode(solve(prob_big, HomotopySweep(; inner = NewtonRaphson())))
let sol = solve(prob_big, ArcLengthContinuation(; inner = NewtonRaphson()))
    @test SciMLBase.successful_retcode(sol)
    @test maximum(abs, sol.u .- 1.0) < 1.0e-6
end
