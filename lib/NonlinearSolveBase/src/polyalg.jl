"""
    NonlinearSolvePolyAlgorithm(algs; start_index::Int = 1)

A general way to define PolyAlgorithms for `NonlinearProblem` and
`NonlinearLeastSquaresProblem`. This is a container for a tuple of algorithms that will be
tried in order until one succeeds. If none succeed, then the algorithm with the lowest
residual is returned.

### Arguments

  - `algs`: a tuple of algorithms to try in-order! (If this is not a Tuple, then the
    returned algorithm is not type-stable).

### Keyword Arguments

  - `start_index`: the index to start at. Defaults to `1`.

### Example

```julia
using NonlinearSolve

alg = NonlinearSolvePolyAlgorithm((NewtonRaphson(), Broyden()))
```
"""
@concrete struct NonlinearSolvePolyAlgorithm <: AbstractNonlinearSolveAlgorithm
    static_length <: Val
    algs <: Tuple
    start_index::Int
end

function NonlinearSolvePolyAlgorithm(algs; start_index::Int = 1)
    @assert 0 < start_index ≤ length(algs)
    algs = Tuple(algs)
    return NonlinearSolvePolyAlgorithm(Val(length(algs)), algs, start_index)
end

@concrete mutable struct NonlinearSolvePolyAlgorithmCache <: AbstractNonlinearSolveCache
    static_length <: Val
    prob <: AbstractNonlinearProblem

    caches <: Tuple
    alg <: NonlinearSolvePolyAlgorithm

    best::Int
    current::Int
    nsteps::Int

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
    NonlinearSolveBase.get_abstol(cache.caches[cache.current])
end
function NonlinearSolveBase.get_reltol(cache::NonlinearSolvePolyAlgorithmCache)
    NonlinearSolveBase.get_reltol(cache.caches[cache.current])
end

function SII.symbolic_container(cache::NonlinearSolvePolyAlgorithmCache)
    return cache.caches[cache.current]
end
function SII.state_values(cache::NonlinearSolvePolyAlgorithmCache)
    SII.state_values(SII.symbolic_container(cache))
end
function SII.parameter_values(cache::NonlinearSolvePolyAlgorithmCache)
    SII.parameter_values(SII.symbolic_container(cache))
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
    NonlinearSolveBase.show_nonlinearsolve_cache(io, cache.caches[cache.current], 4)
end

function InternalAPI.reinit!(
        cache::NonlinearSolvePolyAlgorithmCache, args...; p = cache.p, u0 = cache.u0
)
    foreach(cache.caches) do cache
        InternalAPI.reinit!(cache, args...; p, u0)
    end
    cache.current = cache.alg.start_index
    InternalAPI.reinit!(cache.stats)
    cache.nsteps = 0
    cache.total_time = 0.0
end

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm, args...;
        stats = NLStats(0, 0, 0, 0, 0), maxtime = nothing, maxiters = 1000,
        internalnorm::IN = L2_NORM, alias_u0 = false, verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveDefaultInit(), kwargs...
) where {IN}
    if alias_u0 && !ArrayInterface.ismutable(prob.u0)
        @SciMLMessage("`alias_u0` has been set to `true`, but `u0` is 
            immutable (checked using `ArrayInterface.ismutable``).", verbose, :alias_u0_immutable)
        alias_u0 = false  # If immutable don't care about aliasing
    end

    if verbose isa Bool
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(Verbosity.None())
        end
    elseif verbose isa Verbosity.Type
        verbose = NonlinearVerbosity(verbose)
    end

    u0 = prob.u0
    u0_aliased = alias_u0 ? copy(u0) : u0
    alias_u0 && (prob = SciMLBase.remake(prob; u0 = u0_aliased))

    cache = NonlinearSolvePolyAlgorithmCache(
        alg.static_length, prob,
        map(alg.algs) do solver
            SciMLBase.__init(
                prob, solver, args...;
                stats, maxtime, internalnorm, alias_u0, verbose,
                initializealg = SciMLBase.NoInit(), kwargs...
            )
        end,
        alg, -1, alg.start_index, 0, stats, 0.0, maxtime,
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
        push!(calls,
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
                        else
                            cache.current = $(i + 1)
                        end
                    end
                    return
                end
            end)
    end

    push!(calls, quote
        if !(1 ≤ cache.current ≤ length(cache.caches))
            minfu, idx = findmin_caches(cache.prob, cache.caches)
            cache.best = idx
            cache.retcode = cache.caches[idx].retcode
            cache.force_stop = true
            return
        end
    end)

    return Expr(:block, calls...)
end

# Original is often determined on runtime information especially for PolyAlgorithms so it
# is best to never specialize on that
function build_solution_less_specialize(
        prob::AbstractNonlinearProblem, alg, u, resid;
        retcode = ReturnCode.Default, original = nothing, left = nothing,
        right = nothing, stats = nothing, trace = nothing, kwargs...
)
    return SciMLBase.NonlinearSolution{
        eltype(eltype(u)), ndims(u), typeof(u), typeof(resid), typeof(prob),
        typeof(alg), Any, typeof(left), typeof(stats), typeof(trace)
    }(
        u, resid, prob, alg, retcode, original, left, right, stats, trace
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
