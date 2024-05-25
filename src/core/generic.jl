function SciMLBase.__solve(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        alg::AbstractNonlinearSolveAlgorithm, args...; stats=empty_nlstats(), kwargs...)
    cache = SciMLBase.__init(prob, alg, args...; stats, kwargs...)
    return solve!(cache)
end

function not_terminated(cache::AbstractNonlinearSolveCache)
    return !cache.force_stop && cache.nsteps < cache.maxiters
end

function SciMLBase.solve!(cache::AbstractNonlinearSolveCache)
    while not_terminated(cache)
        step!(cache)
    end

    # The solver might have set a different `retcode`
    if cache.retcode == ReturnCode.Default
        cache.retcode = ifelse(
            cache.nsteps ≥ cache.maxiters, ReturnCode.MaxIters, ReturnCode.Success)
    end

    update_from_termination_cache!(cache.termination_cache, cache)

    update_trace!(cache.trace, cache.nsteps, get_u(cache),
        get_fu(cache), nothing, nothing, nothing; last = True)

    return SciMLBase.build_solution(cache.prob, cache.alg, get_u(cache), get_fu(cache);
        cache.retcode, cache.stats, cache.trace)
end

"""
    step!(cache::AbstractNonlinearSolveCache;
        recompute_jacobian::Union{Nothing, Bool} = nothing)

Performs one step of the nonlinear solver.

### Keyword Arguments

  - `recompute_jacobian`: allows controlling whether the jacobian is recomputed at the
    current step. If `nothing`, then the algorithm determines whether to recompute the
    jacobian. If `true` or `false`, then the jacobian is recomputed or not recomputed,
    respectively. For algorithms that don't use jacobian information, this keyword is
    ignored with a one-time warning.
"""
function SciMLBase.step!(cache::AbstractNonlinearSolveCache{iip, timeit},
        args...; kwargs...) where {iip, timeit}
    not_terminated(cache) || return
    timeit && (time_start = time())
    res = @static_timeit cache.timer "solve" begin
        __step!(cache, args...; kwargs...)
    end

    cache.stats.nsteps += 1
    cache.nsteps += 1

    if timeit
        cache.total_time += time() - time_start
        if !cache.force_stop &&
           cache.retcode == ReturnCode.Default &&
           cache.total_time ≥ cache.maxtime
            cache.retcode = ReturnCode.MaxTime
            cache.force_stop = true
        end
    end

    return res
end
