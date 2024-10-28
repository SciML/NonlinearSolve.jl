function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...;
        kwargs...
)
    cache = SciMLBase.init(prob, alg, args...; kwargs...)
    return SciMLBase.solve!(cache)
end

function CommonSolve.solve!(cache::AbstractNonlinearSolveCache)
    while not_terminated(cache)
        SciMLBase.step!(cache)
    end

    # The solver might have set a different `retcode`
    if cache.retcode == ReturnCode.Default
        cache.retcode = ifelse(
            cache.nsteps ≥ cache.maxiters, ReturnCode.MaxIters, ReturnCode.Success
        )
    end

    update_from_termination_cache!(cache.termination_cache, cache)

    update_trace!(
        cache.trace, cache.nsteps, get_u(cache), get_fu(cache), nothing, nothing, nothing;
        last = Val(true)
    )

    return SciMLBase.build_solution(
        cache.prob, cache.alg, get_u(cache), get_fu(cache);
        cache.retcode, cache.stats, cache.trace
    )
end

"""
    step!(
        cache::AbstractNonlinearSolveCache;
        recompute_jacobian::Union{Nothing, Bool} = nothing
    )

Performs one step of the nonlinear solver.

### Keyword Arguments

  - `recompute_jacobian`: allows controlling whether the jacobian is recomputed at the
    current step. If `nothing`, then the algorithm determines whether to recompute the
    jacobian. If `true` or `false`, then the jacobian is recomputed or not recomputed,
    respectively. For algorithms that don't use jacobian information, this keyword is
    ignored with a one-time warning.
"""
function SciMLBase.step!(cache::AbstractNonlinearSolveCache, args...; kwargs...)
    not_terminated(cache) || return

    has_time_limit(cache) && (time_start = time())

    res = @static_timeit cache.timer "solve" begin
        InternalAPI.step!(cache, args...; kwargs...)
    end

    cache.stats.nsteps += 1
    cache.nsteps += 1

    if has_time_limit(cache)
        cache.total_time += time() - time_start

        if !cache.force_stop && cache.retcode == ReturnCode.Default &&
           cache.total_time ≥ cache.maxtime
            cache.retcode = ReturnCode.MaxTime
            cache.force_stop = true
        end
    end

    return res
end

# Some algorithms don't support creating a cache and doing `solve!`, this unfortunately
# makes it difficult to write generic code that supports caching. For the algorithms that
# don't have a `__init` function defined, we create a "Fake Cache", which just calls
# `__solve` from `solve!`
# Warning: This doesn't implement all the necessary interface functions
@concrete mutable struct NonlinearSolveNoInitCache <: AbstractNonlinearSolveCache
    prob
    alg
    args
    kwargs::Any
end

function SciMLBase.reinit!(
        cache::NonlinearSolveNoInitCache, u0 = cache.prob.u0; p = cache.prob.p, kwargs...
)
    cache.prob = SciMLBase.remake(cache.prob; u0, p)
    cache.kwargs = merge(cache.kwargs, kwargs)
    return cache
end

function Base.show(io::IO, ::MIME"text/plain", cache::NonlinearSolveNoInitCache)
    print(io, "NonlinearSolveNoInitCache(alg = $(cache.alg))")
end

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...;
        kwargs...
)
    return NonlinearSolveNoInitCache(prob, alg, args, kwargs)
end

function CommonSolve.solve!(cache::NonlinearSolveNoInitCache)
    return CommonSolve.solve(cache.prob, cache.alg, cache.args...; cache.kwargs...)
end
