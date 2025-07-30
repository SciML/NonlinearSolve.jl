struct EvalFunc{F} <: Function
    f::F
end
(f::EvalFunc)(args...) = f.f(args...)

"""
```julia
solve(prob::NonlinearProblem, alg::Union{AbstractNonlinearAlgorithm,Nothing}; kwargs...)
```

## Arguments

The only positional argument is `alg` which is optional. By default, `alg = nothing`.
If `alg = nothing`, then `solve` dispatches to the NonlinearSolve.jl automated
algorithm selection (if `using NonlinearSolve` was done, otherwise it will
error with a `MethodError`).

## Keyword Arguments

The NonlinearSolve.jl universe has a large set of common arguments available
for the `solve` function. These arguments apply to `solve` on any problem type and
are only limited by limitations of the specific implementations.

Many of the defaults depend on the algorithm or the package the algorithm derives
from. Not all of the interface is provided by every algorithm.
For more detailed information on the defaults and the available options
for specific algorithms / packages, see the manual pages for the solvers of specific
problems.

#### Error Control

* `abstol`: Absolute tolerance.
* `reltol`: Relative tolerance.

### Miscellaneous

* `maxiters`: Maximum number of iterations before stopping. Defaults to 1e5.
* `verbose`: Toggles whether warnings are thrown when the solver exits early.
  Defaults to true.

### Sensitivity Algorithms (`sensealg`)

`sensealg` is used for choosing the way the automatic differentiation is performed.
    For more information, see the documentation for SciMLSensitivity:
    https://docs.sciml.ai/SciMLSensitivity/stable/
"""
function solve(prob::AbstractNonlinearProblem, args...; sensealg = nothing,
        u0 = nothing, p = nothing, wrap = Val(true), kwargs...)
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end

    if haskey(prob.kwargs, :alias_u0)
        @warn "The `alias_u0` keyword argument is deprecated. Please use a NonlinearAliasSpecifier, e.g. `alias = NonlinearAliasSpecifier(alias_u0 = true)`."
        alias_spec = NonlinearAliasSpecifier(alias_u0 = prob.kwargs[:alias_u0])
    elseif haskey(kwargs, :alias_u0)
        @warn "The `alias_u0` keyword argument is deprecated. Please use a NonlinearAliasSpecifier, e.g. `alias = NonlinearAliasSpecifier(alias_u0 = true)`."
        alias_spec = NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end

    if haskey(prob.kwargs, :alias) && prob.kwargs[:alias] isa Bool
        alias_spec = NonlinearAliasSpecifier(alias = prob.kwargs[:alias])
    elseif haskey(kwargs, :alias) && kwargs[:alias] isa Bool
        alias_spec = NonlinearAliasSpecifier(alias = kwargs[:alias])
    end

    if haskey(prob.kwargs, :alias) && prob.kwargs[:alias] isa NonlinearAliasSpecifier
        alias_spec = prob.kwargs[:alias]
    elseif haskey(kwargs, :alias) && kwargs[:alias] isa NonlinearAliasSpecifier
        alias_spec = kwargs[:alias]
    else
        alias_spec = NonlinearAliasSpecifier(alias_u0 = false)
    end

    alias_u0 = alias_spec.alias_u0

    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p

    if wrap isa Val{true}
        wrap_sol(solve_up(prob,
            sensealg,
            u0,
            p,
            args...;
            alias_u0 = alias_u0,
            originator = SciMLBase.ChainRulesOriginator(),
            kwargs...))
    else
        solve_up(prob,
            sensealg,
            u0,
            p,
            args...;
            alias_u0 = alias_u0,
            originator = SciMLBase.ChainRulesOriginator(),
            kwargs...)
    end
end

function solve_up(prob::AbstractNonlinearProblem, sensealg, u0, p,
        args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...)
    alg = extract_alg(args, kwargs, has_kwargs(prob) ? prob.kwargs : kwargs)
    if isnothing(alg) || !(alg isa AbstractNonlinearSolveAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob, true; u0 = u0,
            p = p, kwargs...)
        solve_call(_prob, args...; kwargs...)
    else
        _prob = get_concrete_problem(prob, true; u0 = u0, p = p, kwargs...)
        #check_prob_alg_pairing(_prob, alg) # use alg for improved inference
        if length(args) > 1
            solve_call(_prob, alg, Base.tail(args)...; kwargs...)
        else
            solve_call(_prob, alg; kwargs...)
        end
    end
end

function solve_call(_prob, args...; merge_callbacks = true, kwargshandle = nothing,
        kwargs...)
    kwargshandle = kwargshandle === nothing ? KeywordArgError : kwargshandle
    kwargshandle = has_kwargs(_prob) && haskey(_prob.kwargs, :kwargshandle) ?
                   _prob.kwargs[:kwargshandle] : kwargshandle

    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    checkkwargs(kwargshandle; kwargs...)
    if isdefined(_prob, :u0)
        if _prob.u0 isa Array
            if !isconcretetype(RecursiveArrayTools.recursive_unitless_eltype(_prob.u0))
                throw(NonConcreteEltypeError(RecursiveArrayTools.recursive_unitless_eltype(_prob.u0)))
            end

            if !(eltype(_prob.u0) <: Number) && !(eltype(_prob.u0) <: Enum) &&
               !(_prob.u0 isa AbstractVector{<:AbstractArray} && _prob isa BVProblem)
                # Allow Enums for FunctionMaps, make into a trait in the future
                # BVPs use Vector of Arrays for initial guesses
                throw(NonNumberEltypeError(eltype(_prob.u0)))
            end
        end

        if _prob.u0 === nothing
            return build_null_solution(_prob, args...; kwargs...)
        end
    end

    if hasfield(typeof(_prob), :f) && hasfield(typeof(_prob.f), :f) &&
       _prob.f.f isa EvalFunc
        Base.invokelatest(__solve, _prob, args...; kwargs...)#::T
    else
        __solve(_prob, args...; kwargs...)#::T
    end
end

function solve_call(prob::SteadyStateProblem,
        alg::AbstractNonlinearAlgorithm, args...;
        kwargs...)
    solve_call(NonlinearProblem(prob),
        alg, args...;
        kwargs...)
end

function init(
        prob::AbstractNonlinearProblem, args...; sensealg = nothing,
        u0 = nothing, p = nothing, kwargs...)
    if sensealg === nothing && has_kwargs(prob) && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end

    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p

    init_up(prob, sensealg, u0, p, args...; kwargs...)
end

function init_up(prob::AbstractNonlinearProblem,
        sensealg, u0, p, args...; kwargs...)
    alg = extract_alg(args, kwargs, has_kwargs(prob) ? prob.kwargs : kwargs)
    if isnothing(alg) || !(alg isa AbstractNonlinearAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob, true; u0 = u0,
            p = p, kwargs...)
        init_call(_prob, args...; kwargs...)
    else
        tstops = get(kwargs, :tstops, nothing)
        if tstops === nothing && has_kwargs(prob)
            tstops = get(prob.kwargs, :tstops, nothing)
        end
        if !(tstops isa Union{Nothing, AbstractArray, Tuple, Real}) &&
           !SciMLBase.allows_late_binding_tstops(alg)
            throw(LateBindingTstopsNotSupportedError())
        end
        _prob = get_concrete_problem(prob, true; u0 = u0, p = p, kwargs...)
        #check_prob_alg_pairing(_prob, alg) # alg for improved inference
        if length(args) > 1
            init_call(_prob, alg, Base.tail(args)...; kwargs...)
        else
            init_call(_prob, alg; kwargs...)
        end
    end
end

function init_call(_prob, args...; merge_callbacks=true, kwargshandle=nothing,
    kwargs...)
    kwargshandle = kwargshandle === nothing ? KeywordArgError : kwargshandle
    kwargshandle = has_kwargs(_prob) && haskey(_prob.kwargs, :kwargshandle) ?
                   _prob.kwargs[:kwargshandle] : kwargshandle
    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    checkkwargs(kwargshandle; kwargs...)
    if hasfield(typeof(_prob), :f) && hasfield(typeof(_prob.f), :f) &&
           _prob.f.f isa EvalFunc
        Base.invokelatest(__init, _prob, args...; kwargs...)#::T
    else
        __init(_prob, args...; kwargs...)#::T
    end
end

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...; kwargs...)
    cache = SciMLBase.__init(prob, alg, args...; kwargs...)
    sol = CommonSolve.solve!(cache)

    return sol
end

function CommonSolve.solve!(cache::AbstractNonlinearSolveCache)
    if cache.retcode == ReturnCode.InitialFailure
        return SciMLBase.build_solution(
            cache.prob, cache.alg, get_u(cache), get_fu(cache);
            cache.retcode, cache.stats, cache.trace
        )
    end

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

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, args...; default_set = false, second_time = false,
        kwargs...)
    if second_time
        throw(NoDefaultAlgorithmError())
    elseif length(args) > 0 && !(first(args) isa AbstractNonlinearAlgorithm)
        throw(NonSolverError())
    else
        __solve(prob, nothing, args...; default_set = false, second_time = true, kwargs...)
    end
end

function __init(prob::AbstractNonlinearProblem, args...; default_set = false, second_time = false,
        kwargs...)
    if second_time
        throw(NoDefaultAlgorithmError())
    elseif length(args) > 0 && !(first(args) isa
             Union{Nothing, AbstractDEAlgorithm, AbstractNonlinearAlgorithm})
        throw(NonSolverError())
    else
        __init(prob, nothing, args...; default_set = false, second_time = true, kwargs...)
    end
end

@generated function __generated_polysolve(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm{Val{N}}, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias_u0 = false, verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveDefaultInit(), kwargs...
) where {N}
    sol_syms = [gensym("sol") for _ in 1:N]
    prob_syms = [gensym("prob") for _ in 1:N]
    u_result_syms = [gensym("u_result") for _ in 1:N]
    calls = [quote
        current = alg.start_index
        if alias_u0 && !ArrayInterface.ismutable(prob.u0)
            @SciMLMessage("`alias_u0` has been set to `true`, but `u0` is
            immutable (checked using `ArrayInterface.ismutable``).", verbose, :alias_u0_immutable, :error_control)
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
                    if SciMLBase.successful_retcode($(cur_sol)) &&
                       $(cur_sol).retcode !== ReturnCode.StalledSuccess
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

    verbose
end

function get_abstol(cache::NonlinearSolveNoInitCache)
    get(cache.kwargs, :abstol, get_tolerance(nothing, eltype(cache.prob.u0)))
end
function get_reltol(cache::NonlinearSolveNoInitCache)
    get(cache.kwargs, :reltol, get_tolerance(nothing, eltype(cache.prob.u0)))
end

SII.parameter_values(cache::NonlinearSolveNoInitCache) = SII.parameter_values(cache.prob)
SII.state_values(cache::NonlinearSolveNoInitCache) = SII.state_values(cache.prob)

get_u(cache::NonlinearSolveNoInitCache) = SII.state_values(cache.prob)

# has_kwargs(_prob::AbstractNonlinearProblem) = has_kwargs(typeof(_prob))
# Base.@pure __has_kwargs(::Type{T}) where {T} = :kwargs ∈ fieldnames(T)
# has_kwargs(::Type{T}) where {T} = __has_kwargs(T)

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
        initializealg = NonlinearSolveDefaultInit(), verbose = NonlinearVerbosity(),
        kwargs...
)
    cache = NonlinearSolveNoInitCache(
        prob, alg, args, kwargs, initializealg, ReturnCode.Default, verbose)
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

function _solve_adjoint(prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...)
    alg = extract_alg(args, kwargs, prob.kwargs)
    if isnothing(alg) || !(alg isa AbstractDEAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob, true; u0 = u0,
            p = p, kwargs...)
    else
        _prob = get_concrete_problem(prob, isadaptive(alg); u0 = u0, p = p, kwargs...)
    end

    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    if length(args) > 1
        _concrete_solve_adjoint(_prob, alg, sensealg, u0, p, originator,
            Base.tail(args)...; kwargs...)
    else
        _concrete_solve_adjoint(_prob, alg, sensealg, u0, p, originator; kwargs...)
    end
end

function _solve_forward(prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...)
    alg = extract_alg(args, kwargs, prob.kwargs)
    if isnothing(alg) || !(alg isa AbstractDEAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob, true; u0 = u0,
            p = p, kwargs...)
    else
        _prob = get_concrete_problem(prob, isadaptive(alg); u0 = u0, p = p, kwargs...)
    end

    if has_kwargs(_prob)
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs), kwargs)
    end

    if length(args) > 1
        _concrete_solve_forward(_prob, alg, sensealg, u0, p, originator,
            Base.tail(args)...; kwargs...)
    else
        _concrete_solve_forward(_prob, alg, sensealg, u0, p, originator; kwargs...)
    end
end

function get_concrete_problem(prob::NonlinearProblem, isadapt; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SII.state_values(prob), p = SII.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs) 
    u0 = get_concrete_u0(prob, isadapt, nothing, kwargs)
    u0 = promote_u0(u0, p, nothing)
    remake(prob; u0 = u0, p = p)
end

function get_concrete_problem(prob::NonlinearLeastSquaresProblem, isadapt; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SII.state_values(prob), p = SII.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs)
    u0 = get_concrete_u0(prob, isadapt, nothing, kwargs)
    u0 = promote_u0(u0, p, nothing)
    remake(prob; u0 = u0, p = p)
end

function get_concrete_problem(
    prob::ImmutableNonlinearProblem, isadapt; kwargs...)
    u0 = get_concrete_u0(prob, isadapt, nothing, kwargs)
    u0 = promote_u0(u0, prob.p, nothing)
    p = get_concrete_p(prob, kwargs)
    return remake(prob; u0 = u0, p = p)
end

function get_concrete_problem(prob::SteadyStateProblem, isadapt; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(SciMLBase.get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SII.state_values(prob), p = SII.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs)
    u0 = get_concrete_u0(prob, isadapt, Inf, kwargs)
    u0 = promote_u0(u0, p, nothing)
    remake(prob; u0 = u0, p = p)
end


"""
Given the index provider `indp` used to construct the problem `prob` being solved, return
an updated `prob` to be used for solving. All implementations should accept arbitrary
keyword arguments.

Should be called before the problem is solved, after performing type-promotion on the
problem. If the returned problem is not `===` the provided `prob`, it is assumed to
contain the `u0` and `p` passed as keyword arguments.

# Keyword Arguments

- `u0`, `p`: Override values for `state_values(prob)` and `parameter_values(prob)` which
  should be used instead of the ones in `prob`.
"""
function get_updated_symbolic_problem(indp, prob; kw...)
    return prob
end

function build_null_solution(
        prob::Union{NonlinearProblem, SteadyStateProblem},
        args...;
        saveat = (),
        save_everystep = true,
        save_on = true,
        save_start = save_everystep || isempty(saveat) ||
                         saveat isa Number || prob.tspan[1] in saveat,
        save_end = true,
        kwargs...)
    prob, success = hack_null_solution_init(prob)
    retcode = success ? ReturnCode.Success : ReturnCode.InitialFailure
    SciMLBase.build_solution(prob, nothing, Float64[], nothing; retcode)
end

function build_null_solution(
        prob::NonlinearLeastSquaresProblem,
        args...; abstol = 1e-6, kwargs...)
    prob, success = hack_null_solution_init(prob)
    retcode = success ? ReturnCode.Success : ReturnCode.InitialFailure

    if isinplace(prob)
        resid = isnothing(prob.f.resid_prototype) ? Float64[] : copy(prob.f.resid_prototype)
        prob.f(resid, prob.u0, prob.p)
    else
        resid = prob.f(prob.f.resid_prototype, prob.p)
    end

    if success
        retcode = norm(resid) < abstol ? ReturnCode.Success : ReturnCode.Failure
    end

    SciMLBase.build_solution(prob, nothing, Float64[], resid; retcode)
end

function hack_null_solution_init(prob::Union{NonlinearProblem, NonlinearLeastSquaresProblem, SteadyStateProblem})
    if SciMLBase.has_initialization_data(prob.f)
        initializeprob = prob.f.initialization_data.initializeprob
        nlsol = solve(initializeprob)
        success = SciMLBase.successful_retcode(nlsol)
        if prob.f.initialization_data.initializeprobpmap !== nothing
            @set! prob.p = prob.f.initializeprobpmap(prob, nlsol)
        end
    else
        success = true
    end
    return prob, success
end
