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
        if cache.nsteps == cache.maxiters
            cache.retcode = ReturnCode.MaxIters
        else
            cache.retcode = ReturnCode.Success
        end
    end

    update_from_termination_cache!(cache.termination_cache, cache)

    update_trace!(cache.trace, get_nsteps(cache), get_u(cache), get_fu(cache), nothing,
        nothing, nothing; last = True)

    stats = SciMLBase.NLStats(get_nf(cache), get_njacs(cache), get_nfactors(cache),
        get_nsolve(cache), get_nsteps(cache))

    return SciMLBase.build_solution(cache.prob, cache.alg, get_u(cache), get_fu(cache);
        cache.retcode, stats, cache.trace)
end

function SciMLBase.step!(cache::AbstractNonlinearSolveCache, args...; kwargs...)
    time_start = time()
    res = @timeit_debug cache.timer "solve" begin
        __step!(cache, args...; kwargs...)
    end
    cache.nsteps += 1
    cache.total_time += time() - time_start

    if !cache.force_stop && cache.retcode == ReturnCode.Default &&
       cache.total_time ≥ cache.maxtime
        cache.retcode = ReturnCode.MaxTime
        cache.force_stop = true
    end

    return res
end
