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

    #     trace = __getproperty(cache, Val{:trace}())
    #     if trace !== nothing
    #         update_trace!(trace, cache.stats.nsteps, get_u(cache), get_fu(cache), nothing,
    #             nothing, nothing; last = Val(true))
    #     end

    return SciMLBase.build_solution(cache.prob, cache.alg, get_u(cache), get_fu(cache);
        cache.retcode)

    # return SciMLBase.build_solution(cache.prob, cache.alg, get_u(cache), get_fu(cache);
    #     cache.retcode, cache.stats, trace)
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
