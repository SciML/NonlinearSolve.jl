function SciMLBase.__solve(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem},
        alg::AbstractNonlinearSolveAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    return solve!(cache)
end

function not_terminated(cache::AbstractNonlinearSolveCache)
    return !cache.force_stop && get_nsteps(cache) < cache.maxiters
end

function SciMLBase.solve!(cache::AbstractNonlinearSolveCache)
    while not_terminated(cache)
        step!(cache)
    end

    # The solver might have set a different `retcode`
    if cache.retcode == ReturnCode.Default
        cache.retcode = ifelse(
            get_nsteps(cache) ≥ cache.maxiters, ReturnCode.MaxIters, ReturnCode.Success)
    end

    update_from_termination_cache!(cache.termination_cache, cache)

    update_trace!(cache.trace, get_nsteps(cache), get_u(cache),
        get_fu(cache), nothing, nothing, nothing; last = True)

    stats = ImmutableNLStats(get_nf(cache), get_njacs(cache), get_nfactors(cache),
        get_nsolve(cache), get_nsteps(cache))

    return SciMLBase.build_solution(cache.prob, cache.alg, get_u(cache),
        get_fu(cache); cache.retcode, stats, cache.trace)
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
function SciMLBase.step!(cache::AbstractNonlinearSolveCache{iip, timeit}, args...;
        kwargs...) where {iip, timeit}
    not_terminated(cache) || return
    timeit && (time_start = time())
    res = @static_timeit cache.timer "solve" begin
        __step!(cache, args...; kwargs...)
    end

    hasfield(typeof(cache), :nsteps) && (cache.nsteps += 1)

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
