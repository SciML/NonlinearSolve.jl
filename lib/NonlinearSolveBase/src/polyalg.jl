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
end

function SII.symbolic_container(cache::NonlinearSolvePolyAlgorithmCache)
    return cache.caches[cache.current]
end
SII.state_values(cache::NonlinearSolvePolyAlgorithmCache) = cache.u0

function Base.show(io::IO, ::MIME"text/plain", cache::NonlinearSolvePolyAlgorithmCache)
    println(io, "NonlinearSolvePolyAlgorithmCache with \
                 $(Utils.unwrap_val(cache.static_length)) algorithms:")
    best_alg = ifelse(cache.best == -1, "nothing", cache.best)
    println(io, "    Best Algorithm: $(best_alg)")
    println(
        io, "    Current Algorithm: [$(cache.current) / $(Utils.unwrap_val(cache.static_length))]"
    )
    println(io, "    nsteps: $(cache.nsteps)")
    println(io, "    retcode: $(cache.retcode)")
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
        internalnorm = L2_NORM, alias_u0 = false, verbose = true, kwargs...
)
    if alias_u0 && !ArrayInterface.ismutable(prob.u0)
        verbose && @warn "`alias_u0` has been set to `true`, but `u0` is \
                          immutable (checked using `ArrayInterface.ismutable`)."
        alias_u0 = false  # If immutable don't care about aliasing
    end

    u0 = prob.u0
    u0_aliased = alias_u0 ? copy(u0) : u0
    alias_u0 && (prob = SciMLBase.remake(prob; u0 = u0_aliased))

    return NonlinearSolvePolyAlgorithmCache(
        alg.static_length, prob,
        map(alg.algs) do solver
            SciMLBase.__init(
                prob, solver, args...;
                stats, maxtime, internalnorm, alias_u0, verbose, kwargs...
            )
        end,
        alg, -1, alg.start_index, 0, stats, 0.0, maxtime,
        ReturnCode.Default, false, maxiters, internalnorm,
        u0, u0_aliased, alias_u0
    )
end

@generated function CommonSolve.solve!(cache::NonlinearSolvePolyAlgorithmCache{Val{N}}) where {N}
    calls = [quote
        1 ≤ cache.current ≤ $(N) || error("Current choices shouldn't get here!")
    end]

    cache_syms = [gensym("cache") for i in 1:N]
    sol_syms = [gensym("sol") for i in 1:N]
    u_result_syms = [gensym("u_result") for i in 1:N]

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
                        if SciMLBase.successful_retcode($(cache_syms[i]).retcode)
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

@generated function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm{Val{N}}, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias_u0 = false, verbose = true, kwargs...
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
        u0 = prob.u0
        u0_aliased = alias_u0 ? zero(u0) : u0
    end]
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
