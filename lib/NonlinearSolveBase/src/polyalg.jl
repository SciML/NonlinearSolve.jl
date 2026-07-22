"""
    NonlinearSolvePolyAlgorithm(algs; start_index::Int = 1, store_original = Val(false))

A general way to define PolyAlgorithms for `NonlinearProblem` and
`NonlinearLeastSquaresProblem`. This is a container for a tuple of algorithms that will be
tried in order until one succeeds. If none succeed, then the algorithm with the lowest
residual is returned.

### Arguments

  - `algs`: a tuple of algorithms to try in-order! (If this is not a Tuple, then the
    returned algorithm is not type-stable).

### Keyword Arguments

  - `start_index`: the index to start at. Defaults to `1`.
  - `store_original`: Whether to store the winning sub-algorithm's solution in the
    `original` field of the returned solution. Default `Val(false)` to keep the
    return type simple (required for Enzyme AD compatibility). Set to `Val(true)`
    for debugging to inspect the sub-algorithm's solution.

### Example

```julia
using NonlinearSolve

alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
```

### Best-subalgorithm retention (`reinit!(cache; retain_best = true)`)

When the polyalgorithm's cache is reused across a sequence of warm-started solves
(`init`/`reinit!`/`solve!`), each `reinit!` by default restarts the ladder at
`start_index`, so every solve re-fails the same cheap subalgorithms before reaching
the one that works. Passing `retain_best = true` to `reinit!` makes the next `solve!`
start from the subalgorithm that produced the most recent success instead. Escalation
is preserved: if that subalgorithm fails, the ladder continues upward as usual, and
once the last subalgorithm fails it wraps around to the subalgorithms that were
skipped at the start (they occasionally succeed where the retained one stagnates), so
retention never tries fewer subalgorithms than a full ladder run. When no success has
been recorded yet, `retain_best = true` starts the ladder at `start_index` as usual.
The default (`retain_best = false`) is the status-quo full restart.

The wrap-around never goes below the algorithm's `start_index`, so retention never
attempts a subalgorithm the algorithm itself excludes.

Retention periodically *re-probes* the skipped cheaper subalgorithms: when the
retained subalgorithm sits above `start_index`, every
`RETAIN_REPROBE_INTERVAL`-th (8th) retained `reinit!` starts one solve from
`start_index` again. A cheap subalgorithm that failed once transiently (escalating
`best` to an expensive one that then always succeeds) is therefore rediscovered
instead of being locked out for the rest of the warm-started sequence, at a bounded
cost of at most one status-quo-style ladder step per interval.

A `retain_best = true` `reinit!` also reinitializes the subalgorithm caches *lazily*:
only the starting subalgorithm's cache is reinitialized up front, and each further
one is reinitialized at the moment escalation reaches it. Since every subcache
`reinit!` evaluates the residual at the new `u0`, this reduces the per-`reinit!`
residual cost from one evaluation per subalgorithm to one per subalgorithm actually
attempted (usually just the retained one). A consequence is that the shared `stats`
of a solution produced after mid-solve escalation only reflect the subalgorithms run
since the last deferred reinitialization, i.e. effectively the winning subalgorithm's
own effort. Updating solver options in the same `reinit!` eagerly updates every
subcache so that later escalation observes the new values.
"""
@concrete struct NonlinearSolvePolyAlgorithm <: AbstractNonlinearSolveAlgorithm
    static_length <: Val
    algs <: Tuple
    start_index::Int
    store_original <: Val
end

function NonlinearSolvePolyAlgorithm(algs; start_index::Int = 1, store_original = Val(false))
    @assert 0 < start_index ≤ length(algs)
    algs = Tuple(algs)
    return NonlinearSolvePolyAlgorithm(Val(length(algs)), algs, start_index, store_original)
end

@concrete mutable struct NonlinearSolvePolyAlgorithmCache <: AbstractNonlinearSolveCache
    static_length <: Val
    prob <: AbstractNonlinearProblem

    caches <: Tuple
    alg <: NonlinearSolvePolyAlgorithm

    best::Int
    current::Int
    nsteps::Int

    # Best-subalgorithm retention (see the algorithm docstring): `retain_best` arms it
    # for the next solve, `start_current` records where that solve's ladder started
    # (so wrap-around knows where the already-attempted stretch begins), and `wrapped`
    # marks that the ladder already wrapped past the end back to index 1.
    # `deferred_u0`/`deferred_p` carry the `reinit!` arguments for the subcaches whose
    # reinitialization was deferred until escalation reaches them.
    retain_best::Bool
    start_current::Int
    wrapped::Bool
    retain_count::Int
    deferred_u0
    deferred_p

    stats::NLStats
    total_time::Float64
    maxtime

    retcode::ReturnCode.T
    force_stop::Bool

    maxiters::Int
    internalnorm

    u0
    u0_aliased
    alias_u0::Bool

    initializealg

    verbose
end

function update_initial_values!(cache::NonlinearSolvePolyAlgorithmCache, u0, p)
    foreach(cache.caches) do subcache
        update_initial_values!(subcache, u0, p)
    end
    cache.prob = SciMLBase.remake(cache.prob; u0, p)
    return cache
end

function NonlinearSolveBase.get_abstol(cache::NonlinearSolvePolyAlgorithmCache)
    return NonlinearSolveBase.get_abstol(cache.caches[cache.current])
end
function NonlinearSolveBase.get_reltol(cache::NonlinearSolvePolyAlgorithmCache)
    return NonlinearSolveBase.get_reltol(cache.caches[cache.current])
end

function SII.symbolic_container(cache::NonlinearSolvePolyAlgorithmCache)
    return cache.caches[cache.current]
end
function SII.state_values(cache::NonlinearSolvePolyAlgorithmCache)
    return SII.state_values(SII.symbolic_container(cache))
end
function SII.parameter_values(cache::NonlinearSolvePolyAlgorithmCache)
    return SII.parameter_values(SII.symbolic_container(cache))
end

function Base.show(io::IO, ::MIME"text/plain", cache::NonlinearSolvePolyAlgorithmCache)
    println(io, "NonlinearSolvePolyAlgorithmCache with \
                 $(Utils.unwrap_val(cache.static_length)) algorithms:")
    best_alg = ifelse(cache.best == -1, "nothing", cache.best)
    println(io, lazy"    Best Algorithm: $(best_alg)")
    println(
        io, lazy"    Current Algorithm: [$(cache.current) / $(Utils.unwrap_val(cache.static_length))]"
    )
    println(io, lazy"    nsteps: $(cache.nsteps)")
    println(io, lazy"    retcode: $(cache.retcode)")
    print(io, "    Current Cache: ")
    return NonlinearSolveBase.show_nonlinearsolve_cache(io, cache.caches[cache.current], 4)
end

function SciMLBase.reinit!(cache::NonlinearSolvePolyAlgorithmCache; kwargs...)
    return InternalAPI.reinit!(cache; kwargs...)
end
function SciMLBase.reinit!(cache::NonlinearSolvePolyAlgorithmCache, u0; kwargs...)
    return InternalAPI.reinit!(cache; u0, kwargs...)
end
# Every RETAIN_REPROBE_INTERVAL-th retained `reinit!` whose retained subalgorithm sits
# above `start_index` starts one solve from `start_index` again instead. Without this,
# a single transient failure of a cheap subalgorithm escalates `best` to an expensive
# one that then succeeds on every later warm-started solve, locking the cheap winner
# out forever (measured: a near-linear n = 100 sweep where Broyden/Klement steady-state
# beats Newton ran ~2x slower under pure sticky-best than under the full-restart status
# quo). The re-probe rediscovers the cheaper subalgorithm at a bounded cost: at most
# one status-quo-style ladder step per interval.
const RETAIN_REPROBE_INTERVAL = 8

function InternalAPI.reinit!(
        cache::NonlinearSolvePolyAlgorithmCache, args...; p = cache.prob.p, u0 = cache.u0,
        retain_best::Bool = false, kwargs...
    )
    cache.retain_best = retain_best
    cache.retain_count = retain_best ? cache.retain_count + 1 : 0
    retained = retain_best && 1 ≤ cache.best ≤ Utils.unwrap_val(cache.static_length)
    reprobe = retained && cache.best > cache.alg.start_index &&
        cache.retain_count % RETAIN_REPROBE_INTERVAL == 0
    cache.current = (retained && !reprobe) ? cache.best : cache.alg.start_index
    cache.start_current = cache.current
    cache.wrapped = false
    if retain_best
        cache.deferred_u0 = u0
        cache.deferred_p = p
    end
    if retain_best && isempty(kwargs)
        # Lazy subcache reinitialization: every subcache `reinit!` evaluates the
        # residual at the new `u0` (`Utils.reinit_common!`), so eagerly
        # reinitializing all N subcaches costs N residual calls per warm-started
        # solve even though a retained solve usually runs only the starting
        # subalgorithm. Reinitialize just that one here and defer the rest to the
        # moment escalation actually reaches them (`deferred_subcache_reinit!`).
        subcache = cache.caches[cache.current]
        InternalAPI.reinit!(subcache, args...; u = get_u(subcache), p, u0, kwargs...)
    else
        foreach(cache.caches) do cache
            InternalAPI.reinit!(cache, args...; u = get_u(cache), p, u0, kwargs...)
        end
    end
    InternalAPI.reinit!(cache.stats)
    cache.nsteps = 0
    return cache.total_time = 0.0
end

# Reinitializes a subcache whose reinitialization was deferred by a
# `retain_best = true` `reinit!`, at the moment ladder escalation reaches it.
# Mirrors the kwargs of the eager loop in `reinit!` above.
function deferred_subcache_reinit!(cache::NonlinearSolvePolyAlgorithmCache, subcache)
    InternalAPI.reinit!(
        subcache; u = get_u(subcache), p = cache.deferred_p, u0 = cache.deferred_u0
    )
    return nothing
end

# `reinit!` with best-subalgorithm retention when the cache supports it (see the
# `NonlinearSolvePolyAlgorithm` docstring), a plain `reinit!` otherwise. The
# continuation drivers call this on their warm-started tracking steps so the inner
# default polyalgorithm resumes from the subalgorithm that won the previous step
# instead of re-failing the cheaper ones every step.
reinit_retaining!(cache, u0) = SciMLBase.reinit!(cache, u0)
function reinit_retaining!(cache::NonlinearSolvePolyAlgorithmCache, u0)
    return SciMLBase.reinit!(cache, u0; retain_best = true)
end
reinit_retaining!(cache, u0, p) = SciMLBase.reinit!(cache, u0; p)
function reinit_retaining!(cache::NonlinearSolvePolyAlgorithmCache, u0, p)
    return SciMLBase.reinit!(cache, u0; p, retain_best = true)
end

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm, args...;
        stats = NLStats(0, 0, 0, 0, 0), maxtime = nothing, maxiters = 1000,
        internalnorm::IN = L2_NORM, alias = NonlinearAliasSpecifier(alias_u0 = false), verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveDefaultInit(), kwargs...
    ) where {IN}
    if haskey(kwargs, :alias_u0)
        alias = NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    if alias_u0 && !ArrayInterface.ismutable(prob.u0)
        @SciMLMessage("`alias_u0` has been set to `true`, but `u0` is 
            immutable (checked using `ArrayInterface.ismutable``).", verbose, :alias_u0_immutable)
        alias_u0 = false  # If immutable don't care about aliasing
    end

    if verbose isa Bool
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(None())
        end
    elseif verbose isa AbstractVerbosityPreset
        verbose = NonlinearVerbosity(verbose)
    end

    u0 = prob.u0
    u0_aliased = alias_u0 ? copy(u0) : u0
    alias_u0 && (prob = SciMLBase.remake(prob; u0 = u0_aliased))

    cache = NonlinearSolvePolyAlgorithmCache(
        alg.static_length, prob,
        map(alg.algs) do solver
            # `maxiters` must reach the subcaches: the polyalg's `solve!` runs each
            # subcache to ITS termination, so a cap kept only at the polyalg level is
            # silently ignored on the init/solve! path (the one-shot `__solve` path
            # forwards it to every subsolver, and this must match).
            SciMLBase.__init(
                prob, solver, args...;
                stats, maxtime, maxiters, internalnorm, alias, verbose,
                initializealg = SciMLBase.NoInit(), kwargs...
            )
        end,
        alg, -1, alg.start_index, 0, false, alg.start_index, false, 0, u0, prob.p,
        stats, 0.0, maxtime,
        ReturnCode.Default, false, maxiters, internalnorm,
        u0, u0_aliased, alias_u0, initializealg, verbose
    )
    run_initialization!(cache)
    return cache
end

@generated function InternalAPI.step!(
        cache::NonlinearSolvePolyAlgorithmCache{Val{N}}, args...; kwargs...
    ) where {N}
    calls = []
    cache_syms = [gensym("cache") for i in 1:N]
    for i in 1:N
        push!(
            calls,
            quote
                $(cache_syms[i]) = cache.caches[$(i)]
                if $(i) == cache.current
                    InternalAPI.step!($(cache_syms[i]), args...; kwargs...)
                    $(cache_syms[i]).nsteps += 1
                    if !NonlinearSolveBase.not_terminated($(cache_syms[i]))
                        # If a NonlinearLeastSquaresProblem StalledSuccess, try the next
                        # solver to see if you get a lower residual
                        if SciMLBase.successful_retcode($(cache_syms[i]).retcode) &&
                                $(cache_syms[i]).retcode != ReturnCode.StalledSuccess
                            cache.best = $(i)
                            cache.force_stop = true
                            cache.retcode = $(cache_syms[i]).retcode
                        elseif cache.wrapped && $(i + 1) ≥ cache.start_current
                            # a wrapped-around ladder stops where the original
                            # (retained) start already ran; every subalgorithm has
                            # now been attempted
                            minfu, idx = findmin_caches(cache.prob, cache.caches)
                            cache.best = idx
                            cache.retcode = cache.caches[idx].retcode
                            cache.force_stop = true
                        else
                            cache.current = $(i + 1)
                            $(
                                i == N ? :(nothing) : quote
                                        if cache.retain_best
                                            deferred_subcache_reinit!(
                                                cache, cache.caches[$(i + 1)]
                                            )
                                    end
                                    end
                            )
                        end
                    end
                    return
                end
            end
        )
    end

    push!(
        calls, quote
            if !(1 ≤ cache.current ≤ length(cache.caches))
                if cache.retain_best && !cache.wrapped &&
                        cache.start_current > cache.alg.start_index
                    # retention started the ladder mid-way; wrap around to the
                    # skipped cheaper subalgorithms (never below the algorithm's own
                    # `start_index`) before giving up
                    cache.wrapped = true
                    cache.current = cache.alg.start_index
                    deferred_subcache_reinit!(cache, cache.caches[cache.current])
                    return
                end
                minfu, idx = findmin_caches(cache.prob, cache.caches)
                cache.best = idx
                cache.retcode = cache.caches[idx].retcode
                cache.force_stop = true
                return
            end
        end
    )

    return Expr(:block, calls...)
end

# `original` is determined by runtime polyalg branch choice, so we deliberately
# avoid specializing the returned `NonlinearSolution` on its concrete type. The
# old `Any`-typed slot left the solution type with a non-concrete field, which
# trips Enzyme's `MixedReturnException` when the polyalg sits inside a
# reverse-mode differentiated function (#878). To keep the default Enzyme-
# friendly while preserving opt-in introspection, the polyalg now carries a
# `store_original::Val` field (default `Val(false)`) and we branch here on it:
#  - `Val(false)`: drop the payload and pin the type slot to `Nothing`. Type
#    is fully concrete and Enzyme is happy.
#  - `Val(true)`: keep the payload with type slot `Any` (matches the legacy
#    behavior). Useful for debugging; not Enzyme-differentiable.
function build_solution_less_specialize(
        prob::AbstractNonlinearProblem, alg, u, resid;
        retcode = ReturnCode.Default, original = nothing, left = nothing,
        right = nothing, stats = nothing, trace = nothing,
        store_original::Val = Val(false), kwargs...
    )
    if store_original isa Val{true}
        return SciMLBase.NonlinearSolution{
            eltype(eltype(u)), ndims(u), typeof(u), typeof(resid), typeof(prob),
            typeof(alg), Any, typeof(left), typeof(stats), typeof(trace),
        }(
            u, resid, prob, alg, retcode, original, left, right, stats, trace
        )
    end
    return SciMLBase.NonlinearSolution{
        eltype(eltype(u)), ndims(u), typeof(u), typeof(resid), typeof(prob),
        typeof(alg), Nothing, typeof(left), typeof(stats), typeof(trace),
    }(
        u, resid, prob, alg, retcode, nothing, left, right, stats, trace
    )
end

function findmin_caches(prob::AbstractNonlinearProblem, caches)
    resids = map(caches) do cache
        cache === nothing && return nothing
        return NonlinearSolveBase.get_fu(cache)
    end
    return findmin_resids(prob, resids)
end

@views function findmin_resids(prob::AbstractNonlinearProblem, caches)
    norm_fn = prob isa NonlinearLeastSquaresProblem ? Base.Fix2(norm, 2) :
        Base.Fix2(norm, Inf)
    idx = findfirst(Base.Fix2(!==, nothing), caches)
    # This is an internal function so we assume that inputs are consistent and there is
    # atleast one non-`nothing` value
    fx_idx = norm_fn(caches[idx])
    idx == length(caches) && return fx_idx, idx
    fmin = @closure xᵢ -> begin
        xᵢ === nothing && return oftype(fx_idx, Inf)
        fx = norm_fn(xᵢ)
        return ifelse(isnan(fx), oftype(fx, Inf), fx)
    end
    x_min, x_min_idx = findmin(fmin, caches[(idx + 1):length(caches)])
    x_min < fx_idx && return x_min, x_min_idx + idx
    return fx_idx, idx
end
