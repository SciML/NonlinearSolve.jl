function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...;
        kwargs...
)
    cache = SciMLBase.__init(prob, alg, args...; kwargs...)
    return CommonSolve.solve!(cache)
end

function CommonSolve.solve!(cache::AbstractNonlinearSolveCache)
    while not_terminated(cache)
        CommonSolve.step!(cache)
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

@generated function CommonSolve.solve!(cache::NonlinearSolvePolyAlgorithmCache{Val{N}}) where {N}
    calls = [quote
        1 ≤ cache.current ≤ $(N) || error("Current choices shouldn't get here!")
    end]

    cache_syms = [gensym("cache") for i in 1:N]
    sol_syms = [gensym("sol") for i in 1:N]
    u_result_syms = [gensym("u_result") for i in 1:N]

    push!(calls,
        quote
            if cache.retcode == ReturnCode.InitialFailure
                u = $(SII.state_values)(cache)
                return build_solution_less_specialize(
                    cache.prob, cache.alg, u, $(Utils.evaluate_f)(cache.prob, u);
                    retcode = cache.retcode
                )
            end
        end)

    for i in 1:N
        push!(calls,
            quote
                $(cache_syms[i]) = cache.caches[$(i)]
                if $(i) == cache.current
                    cache.alias_u0 && copyto!(cache.u0_aliased, cache.u0)
                    $(sol_syms[i]) = CommonSolve.solve!($(cache_syms[i]))
                    if SciMLBase.successful_retcode($(sol_syms[i]))
                        stats = $(sol_syms[i]).stats
                        if cache.alias_u0
                            copyto!(cache.u0, $(sol_syms[i]).u)
                            $(u_result_syms[i]) = cache.u0
                        else
                            $(u_result_syms[i]) = $(sol_syms[i]).u
                        end
                        fu = NonlinearSolveBase.get_fu($(cache_syms[i]))
                        return build_solution_less_specialize(
                            cache.prob, cache.alg, $(u_result_syms[i]), fu;
                            retcode = $(sol_syms[i]).retcode, stats,
                            original = $(sol_syms[i]), trace = $(sol_syms[i]).trace
                        )
                    elseif cache.alias_u0
                        # For safety we need to maintain a copy of the solution
                        $(u_result_syms[i]) = copy($(sol_syms[i]).u)
                    end
                    cache.current = $(i + 1)
                end
            end)
    end

    resids = map(Base.Fix2(Symbol, :resid), cache_syms)
    for (sym, resid) in zip(cache_syms, resids)
        push!(calls, :($(resid) = @isdefined($(sym)) ? $(sym).resid : nothing))
    end
    push!(calls, quote
        fus = tuple($(Tuple(resids)...))
        minfu, idx = findmin_caches(cache.prob, fus)
    end)
    for i in 1:N
        push!(calls,
            quote
                if idx == $(i)
                    u = cache.alias_u0 ? $(u_result_syms[i]) :
                        NonlinearSolveBase.get_u(cache.caches[$(i)])
                end
            end)
    end
    push!(calls,
        quote
            retcode = cache.caches[idx].retcode
            if cache.alias_u0
                copyto!(cache.u0, u)
                u = cache.u0
            end
            return build_solution_less_specialize(
                cache.prob, cache.alg, u, fus[idx];
                retcode, cache.stats, cache.caches[idx].trace
            )
        end)

    return Expr(:block, calls...)
end

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm,
        args...; kwargs...)
    __generated_polysolve(prob, alg, args...; kwargs...)
end

@generated function __generated_polysolve(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm{Val{N}}, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias_u0 = false, verbose = true,
        initializealg = NonlinearSolveDefaultInit(), kwargs...
) where {N}
    sol_syms = [gensym("sol") for _ in 1:N]
    prob_syms = [gensym("prob") for _ in 1:N]
    u_result_syms = [gensym("u_result") for _ in 1:N]
    calls = [quote
        current = alg.start_index
        if alias_u0 && !ArrayInterface.ismutable(prob.u0)
            verbose && @warn "`alias_u0` has been set to `true`, but `u0` is \
                              immutable (checked using `ArrayInterface.ismutable`)."
            alias_u0 = false  # If immutable don't care about aliasing
        end
    end]

    push!(calls,
        quote
            prob, success = $(run_initialization!)(prob, initializealg, prob)
            if !success
                u = $(SII.state_values)(prob)
                return build_solution_less_specialize(
                    prob, alg, u, $(Utils.evaluate_f)(prob, u);
                    retcode = $(ReturnCode.InitialFailure))
            end
        end)

    push!(calls, quote
        u0 = prob.u0
        u0_aliased = alias_u0 ? zero(u0) : u0
    end)
    for i in 1:N
        cur_sol = sol_syms[i]
        push!(calls,
            quote
                if current == $(i)
                    if alias_u0
                        copyto!(u0_aliased, u0)
                        $(prob_syms[i]) = SciMLBase.remake(prob; u0 = u0_aliased)
                    else
                        $(prob_syms[i]) = prob
                    end
                    $(cur_sol) = SciMLBase.__solve(
                        $(prob_syms[i]), alg.algs[$(i)], args...;
                        stats, alias_u0, verbose, kwargs...
                    )
                    if SciMLBase.successful_retcode($(cur_sol))
                        if alias_u0
                            copyto!(u0, $(cur_sol).u)
                            $(u_result_syms[i]) = u0
                        else
                            $(u_result_syms[i]) = $(cur_sol).u
                        end
                        return build_solution_less_specialize(
                            prob, alg, $(u_result_syms[i]), $(cur_sol).resid;
                            $(cur_sol).retcode, $(cur_sol).stats,
                            $(cur_sol).trace, original = $(cur_sol)
                        )
                    elseif alias_u0
                        # For safety we need to maintain a copy of the solution
                        $(u_result_syms[i]) = copy($(cur_sol).u)
                    end
                    current = $(i + 1)
                end
            end)
    end

    resids = map(Base.Fix2(Symbol, :resid), sol_syms)
    for (sym, resid) in zip(sol_syms, resids)
        push!(calls, :($(resid) = @isdefined($(sym)) ? $(sym).resid : nothing))
    end

    push!(calls, quote
        resids = tuple($(Tuple(resids)...))
        minfu, idx = findmin_resids(prob, resids)
    end)

    for i in 1:N
        push!(calls,
            quote
                if idx == $(i)
                    if alias_u0
                        copyto!(u0, $(u_result_syms[i]))
                        $(u_result_syms[i]) = u0
                    else
                        $(u_result_syms[i]) = $(sol_syms[i]).u
                    end
                    return build_solution_less_specialize(
                        prob, alg, $(u_result_syms[i]), $(sol_syms[i]).resid;
                        $(sol_syms[i]).retcode, $(sol_syms[i]).stats,
                        $(sol_syms[i]).trace, original = $(sol_syms[i])
                    )
                end
            end)
    end
    push!(calls, :(error("Current choices shouldn't get here!")))

    return Expr(:block, calls...)
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
function CommonSolve.step!(cache::AbstractNonlinearSolveCache, args...; kwargs...)
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
    initializealg

    retcode::ReturnCode.T
end

function get_abstol(cache::NonlinearSolveNoInitCache)
    get(cache.kwargs, :abstol, get_tolerance(nothing, eltype(cache.prob.u0)))
end
function get_reltol(cache::NonlinearSolveNoInitCache)
    get(cache.kwargs, :reltol, get_tolerance(nothing, eltype(cache.prob.u0)))
end

get_u(cache::NonlinearSolveNoInitCache) = SII.state_values(cache.prob)

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
        initializealg = NonlinearSolveDefaultInit(),
        kwargs...
)
    cache = NonlinearSolveNoInitCache(
        prob, alg, args, kwargs, initializealg, ReturnCode.Default)
    run_initialization!(cache)
    return cache
end

function CommonSolve.solve!(cache::NonlinearSolveNoInitCache)
    if cache.retcode == ReturnCode.InitialFailure
        u = SII.state_values(cache)
        return SciMLBase.build_solution(
            cache.prob, cache.alg, u, Utils.evaluate_f(cache.prob, u); cache.retcode)
    end
    return CommonSolve.solve(cache.prob, cache.alg, cache.args...; cache.kwargs...)
end
