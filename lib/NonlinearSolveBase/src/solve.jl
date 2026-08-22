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

These tolerances are interpreted by the termination condition.

### Nonlinear Preconditioning

* `precondition`: a left preconditioner `G` applied to the residual, giving the
  root-equivalent system `G(f(u, p), u, p) = 0`. Out-of-place problems return the
  transformed residual, `Gfu = precondition(fu, u, p)`; in-place problems overwrite the
  first argument, `precondition(fu, u, p) -> nothing`. The composition is what the solver
  evaluates and differentiates, so termination and `sol.resid` are measured on it. `G`
  must be root-preserving: `G(r, u, p) = 0` if and only if `r = 0`.
* `postcondition`: an iterate corrector `H` applied to every accepted iterate before the
  residual is evaluated or convergence tested there, and once to the initial guess.
  Out-of-place problems return the corrected iterate,
  `u_new = postcondition(u_proposed, u_prev, p, cache)`; in-place problems overwrite the
  first argument, `postcondition(u_proposed, u_prev, p, cache) -> nothing`. The fourth
  argument is the solver cache — `nothing` for the initial-guess correction, since that
  runs before a cache exists — and correctors that do not need solver state simply ignore
  it. `H` must satisfy `H(u, u, p, cache) = u` at solutions so that roots are unchanged.

  On a problem with `lb`/`ub` bounds the solver iterates on an unconstrained
  reparameterization of `u`, and `H` is applied in the original bounded variable by
  default. Wrap it in a [`PostconditionSpecifier`](@ref) to say otherwise:
  `postcondition = PostconditionSpecifier(H; space = PostconditionSpace.Transformed)`
  applies it to the unconstrained iterate instead.

Both are ordinary solver options: pass them to `solve`/`init`, or carry them on the
problem and have them forwarded like any other keyword.

### Miscellaneous

* `maxiters`: Maximum number of iterations before stopping. Defaults to 1000.
* `verbose`: Toggles whether warnings are thrown when the solver exits early.
  Defaults to true.

### Sensitivity Algorithms (`sensealg`)

`sensealg` is used for choosing the way the automatic differentiation is performed. For
 more information, see the documentation for
[SciMLSensitivity](https://docs.sciml.ai/SciMLSensitivity/stable/)

"""
function solve(
        prob::AbstractNonlinearProblem, args...; sensealg = nothing,
        u0 = nothing, p = nothing, wrap = Val(true), verbose = NonlinearVerbosity(), kwargs...
    )
    if sensealg === nothing && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end

    if verbose isa Bool
        # @warn "Using `true` or `false` for `verbose` is being deprecated. Please use a `NonlinearVerbosity` type to specify verbosity settings.
        # For details see the verbosity section of the common solver options documentation page."
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(None())
        end
    elseif verbose isa AbstractVerbosityPreset
        verbose = NonlinearVerbosity(verbose)
    end

    alias_spec = if haskey(kwargs, :alias) && kwargs[:alias] isa NonlinearAliasSpecifier
        kwargs[:alias]
    elseif haskey(prob.kwargs, :alias) && prob.kwargs[:alias] isa NonlinearAliasSpecifier
        prob.kwargs[:alias]
    elseif haskey(kwargs, :alias) && kwargs[:alias] isa Bool
        NonlinearAliasSpecifier(alias = kwargs[:alias])
    elseif haskey(prob.kwargs, :alias) && prob.kwargs[:alias] isa Bool
        NonlinearAliasSpecifier(alias = prob.kwargs[:alias])
    elseif haskey(kwargs, :alias_u0)
        @warn "The `alias_u0` keyword argument is deprecated. Please use a NonlinearAliasSpecifier, e.g. `alias = NonlinearAliasSpecifier(alias_u0 = true)`."
        NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    elseif haskey(prob.kwargs, :alias_u0)
        @warn "The `alias_u0` keyword argument is deprecated. Please use a NonlinearAliasSpecifier, e.g. `alias = NonlinearAliasSpecifier(alias_u0 = true)`."
        NonlinearAliasSpecifier(alias_u0 = prob.kwargs[:alias_u0])
    else
        NonlinearAliasSpecifier(alias_u0 = false)
    end

    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p

    return if wrap isa Val{true}
        wrap_sol(
            solve_up(
                prob,
                sensealg,
                u0,
                p,
                args...;
                alias = alias_spec,
                originator = SciMLBase.set_mooncakeoriginator_if_mooncake(SciMLBase.ChainRulesOriginator()),
                verbose,
                kwargs...
            )
        )
    else
        solve_up(
            prob,
            sensealg,
            u0,
            p,
            args...;
            alias = alias_spec,
            originator = SciMLBase.set_mooncakeoriginator_if_mooncake(SciMLBase.ChainRulesOriginator()),
            verbose,
            kwargs...
        )
    end
end

function solve_up(
        prob::AbstractNonlinearProblem, sensealg, u0, p,
        args...; originator = SciMLBase.ChainRulesOriginator(),
        kwargs...
    )
    alg = extract_alg(args, kwargs, has_kwargs(prob) ? prob.kwargs : kwargs)
    return if isnothing(alg) || !(alg isa AbstractNonlinearSolveAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)
        solve_call(_prob, args...; kwargs...)
    else
        _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)
        #check_prob_alg_pairing(_prob, alg) # use alg for improved inference
        if length(args) > 1
            solve_call(_prob, alg, Base.tail(args)...; kwargs...)
        else
            solve_call(_prob, alg; kwargs...)
        end
    end
end

function solve_call(
        _prob, args...; merge_callbacks = true, kwargshandle = nothing,
        kwargs...
    )
    kwargshandle = kwargshandle === nothing ? KeywordArgError : kwargshandle
    kwargshandle = has_kwargs(_prob) && haskey(_prob.kwargs, :kwargshandle) ?
        _prob.kwargs[:kwargshandle] : kwargshandle

    if has_kwargs(_prob)
        # `::NamedTuple` assert keeps dispatch off the invalidation-prone `merge(::Any, ::Pairs)` path
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs)::NamedTuple, kwargs)
    end

    checkkwargs(kwargshandle; kwargs...)

    # Compose the nonlinear preconditioning options. Done here (in addition to the
    # `__solve`/`init_call` funnels) so that algorithms with their own `__solve` methods
    # (SimpleNonlinearSolve, extension wrappers) see the composed residual and unsupported
    # `postcondition` usage errors instead of being silently ignored.
    if needs_conditioning(_prob, kwargs)
        _prob = transform_conditioned_problem(
            _prob, length(args) > 0 ? args[1] : nothing, kwargs
        )
    end

    if isdefined(_prob, :u0)
        if _prob.u0 isa Array
            if !isconcretetype(RecursiveArrayTools.recursive_unitless_eltype(_prob.u0))
                throw(NonConcreteEltypeError(RecursiveArrayTools.recursive_unitless_eltype(_prob.u0)))
            end

            if !(eltype(_prob.u0) <: Number) && !(eltype(_prob.u0) <: Enum)
                throw(SciMLBase.NonNumberEltypeError(eltype(_prob.u0)))
            end
        end

        if _prob.u0 === nothing
            return build_null_solution(_prob, args...; kwargs...)
        end
    end

    sol = if hasfield(typeof(_prob), :f) && hasfield(typeof(_prob.f), :f) &&
            _prob.f.f isa EvalFunc
        Base.invokelatest(__solve, _prob, args...; kwargs...) #::T
    else
        __solve(_prob, args...; kwargs...) #::T
    end

    return sol
end

function solve_call(
        prob::SteadyStateProblem,
        alg::AbstractNonlinearAlgorithm, args...;
        kwargs...
    )
    return solve_call(
        NonlinearProblem(prob),
        alg, args...;
        kwargs...
    )
end

function init(
        prob::AbstractNonlinearProblem, args...; sensealg = nothing,
        u0 = nothing, p = nothing, verbose = NonlinearVerbosity(), kwargs...
    )
    if sensealg === nothing && has_kwargs(prob) && haskey(prob.kwargs, :sensealg)
        sensealg = prob.kwargs[:sensealg]
    end

    alias_spec = if haskey(kwargs, :alias) && kwargs[:alias] isa NonlinearAliasSpecifier
        kwargs[:alias]
    elseif haskey(prob.kwargs, :alias) && prob.kwargs[:alias] isa NonlinearAliasSpecifier
        prob.kwargs[:alias]
    elseif haskey(kwargs, :alias) && kwargs[:alias] isa Bool
        NonlinearAliasSpecifier(alias = kwargs[:alias])
    elseif haskey(prob.kwargs, :alias) && prob.kwargs[:alias] isa Bool
        NonlinearAliasSpecifier(alias = prob.kwargs[:alias])
    elseif haskey(kwargs, :alias_u0)
        @warn "The `alias_u0` keyword argument is deprecated. Please use a NonlinearAliasSpecifier, e.g. `alias = NonlinearAliasSpecifier(alias_u0 = true)`."
        NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    elseif haskey(prob.kwargs, :alias_u0)
        @warn "The `alias_u0` keyword argument is deprecated. Please use a NonlinearAliasSpecifier, e.g. `alias = NonlinearAliasSpecifier(alias_u0 = true)`."
        NonlinearAliasSpecifier(alias_u0 = prob.kwargs[:alias_u0])
    else
        NonlinearAliasSpecifier(alias_u0 = false)
    end

    if verbose isa Bool
        # @warn "Using `true` or `false` for `verbose` is being deprecated. Please use a `NonlinearVerbosity` type to specify verbosity settings.
        # For details see the verbosity section of the common solver options documentation page."
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(None())
        end
    elseif verbose isa AbstractVerbosityPreset
        verbose = NonlinearVerbosity(verbose)
    end

    u0 = u0 !== nothing ? u0 : prob.u0
    p = p !== nothing ? p : prob.p

    return init_up(prob, sensealg, u0, p, args...; alias = alias_spec, verbose, kwargs...)
end

function init_up(
        prob::AbstractNonlinearProblem,
        sensealg, u0, p, args...; kwargs...
    )
    alg = extract_alg(args, kwargs, has_kwargs(prob) ? prob.kwargs : kwargs)
    return if isnothing(alg) || !(alg isa AbstractNonlinearAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)
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
        _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)
        #check_prob_alg_pairing(_prob, alg) # alg for improved inference
        if length(args) > 1
            init_call(_prob, alg, Base.tail(args)...; kwargs...)
        else
            init_call(_prob, alg; kwargs...)
        end
    end
end

function init_call(
        _prob, args...; merge_callbacks = true, kwargshandle = nothing,
        kwargs...
    )
    kwargshandle = kwargshandle === nothing ? KeywordArgError : kwargshandle
    kwargshandle = has_kwargs(_prob) && haskey(_prob.kwargs, :kwargshandle) ?
        _prob.kwargs[:kwargshandle] : kwargshandle
    if has_kwargs(_prob)
        # `::NamedTuple` assert keeps dispatch off the invalidation-prone `merge(::Any, ::Pairs)` path
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs)::NamedTuple, kwargs)
    end

    checkkwargs(kwargshandle; kwargs...)

    alg = length(args) > 0 ? args[1] : nothing

    # Compose the nonlinear preconditioning options before any bounds transform, so the
    # composition acts in the original iterate coordinates.
    if needs_conditioning(_prob, kwargs)
        _prob = transform_conditioned_problem(_prob, alg, kwargs)
    end

    # Forward bounds transform: if the algorithm doesn't natively support bounds,
    # apply a variable transformation so the solver operates in unconstrained space.
    if needs_bounds_transform(_prob, alg)
        _prob = transform_bounded_problem(_prob, alg)
    end

    return if hasfield(typeof(_prob), :f) && hasfield(typeof(_prob.f), :f) &&
            _prob.f.f isa EvalFunc
        Base.invokelatest(__init, _prob, args...; kwargs...) #::T
    else
        __init(_prob, args...; kwargs...) #::T
    end
end

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...; kwargs...
    )
    _prob = if needs_conditioning(prob, kwargs)
        transform_conditioned_problem(prob, alg, kwargs)
    else
        prob
    end
    _prob = if needs_bounds_transform(_prob, alg)
        transform_bounded_problem(_prob, alg)
    else
        _prob
    end
    cache = SciMLBase.__init(_prob, alg, args...; kwargs...)
    sol = CommonSolve.solve!(cache)

    return sol
end

@inline _observe_nonlinear_step!(::Nothing, cache) = nothing
@inline function _observe_nonlinear_step!(step_observer, cache)
    return step_observer(get_u(cache), get_fu(cache), cache.nsteps)
end

function _run_cache_to_completion!(
        cache::AbstractNonlinearSolveCache, step_observer = nothing
    )
    cache.retcode == ReturnCode.InitialFailure && return cache
    while not_terminated(cache)
        CommonSolve.step!(cache)
        _observe_nonlinear_step!(step_observer, cache)
    end

    # The solver might have set a different `retcode`
    if cache.retcode == ReturnCode.Default
        cache.retcode = ifelse(
            cache.nsteps ≥ cache.maxiters, ReturnCode.MaxIters, ReturnCode.Success
        )
    end

    # A driver may have stepped with `evaluate_residual = false`; the residual has to be
    # brought forward before it is reported, since nothing downstream re-evaluates it.
    refresh_residual!(cache)
    update_from_termination_cache!(get_termination_cache(cache), cache)

    update_trace!(
        get_trace(cache), cache.nsteps, get_u(cache), get_fu(cache), nothing, nothing, nothing;
        last = Val(true)
    )

    return cache
end

function solve_cache!(cache::AbstractNonlinearSolveCache; step_observer = nothing)
    applicable(InternalAPI.step!, cache) || throw(
        ArgumentError(
            "`solve_cache!` requires an algorithm that supports the nonlinear " *
                "solver iterator interface."
        )
    )
    _run_cache_to_completion!(cache, step_observer)
    return cache.retcode
end

@inline function _has_bounded_wrapper(cache::AbstractNonlinearSolveCache)
    return bounded_wrapper(cache) !== nothing
end

function _solution_from_cache(cache::AbstractNonlinearSolveCache; transform_bounds::Bool)
    sol = SciMLBase.build_solution(
        cache.prob, cache.alg, get_u(cache), get_fu(cache);
        cache.retcode, cache.stats, cache.trace
    )

    # Inverse bounds transform: if the problem function was wrapped with a
    # BoundedWrapper, map the solution back from unbounded to bounded space.
    if transform_bounds && _has_bounded_wrapper(cache)
        bw = cache.prob.f.f
        sol.u .= _from_unbounded.(sol.u, bw.lb, bw.ub)

        # Reset the problem to the original fields that were overwritten
        @set! sol.prob = remake(sol.prob; f = bw.f, lb = bw.lb, ub = bw.ub)
        sol.prob.u0 .= _from_unbounded.(sol.prob.u0, bw.lb, bw.ub)
    end

    return sol
end

function _solve_without_solution!(cache::AbstractNonlinearSolveCache)
    if applicable(InternalAPI.step!, cache) && hasfield(typeof(cache), :termination_cache) &&
            hasfield(typeof(cache), :trace)
        cache.retcode == ReturnCode.InitialFailure && return cache
        _run_cache_to_completion!(cache)
        return _has_bounded_wrapper(cache) ?
            _solution_from_cache(cache; transform_bounds = true) : cache
    end
    return CommonSolve.solve!(cache)
end

function CommonSolve.solve!(cache::AbstractNonlinearSolveCache)
    if cache.retcode == ReturnCode.InitialFailure
        return _solution_from_cache(cache; transform_bounds = false)
    end

    _run_cache_to_completion!(cache)
    return _solution_from_cache(cache; transform_bounds = true)
end


@inline _solve_result_u(result) = result.u
@inline _solve_result_resid(result) = result.resid
@inline _solve_result_retcode(result) = result.retcode
@inline _solve_result_stats(result) = result.stats
@inline _solve_result_original(result) = result

@inline _solve_result_u(cache::AbstractNonlinearSolveCache) = get_u(cache)
@inline _solve_result_resid(cache::AbstractNonlinearSolveCache) = get_fu(cache)
@inline _solve_result_retcode(cache::AbstractNonlinearSolveCache) = cache.retcode
@inline _solve_result_stats(cache::AbstractNonlinearSolveCache) = cache.stats
@inline function _solve_result_original(cache::AbstractNonlinearSolveCache)
    return _solution_from_cache(
        cache; transform_bounds = cache.retcode != ReturnCode.InitialFailure
    )
end

@inline function _solve_result_successful(result)
    return SciMLBase.successful_retcode(_solve_result_retcode(result))
end

@generated function CommonSolve.solve!(cache::NonlinearSolvePolyAlgorithmCache{Val{N}}) where {N}
    calls = [
        quote
            1 ≤ cache.current ≤ $(N) || error("Current choices shouldn't get here!")
            # Compute concrete types from the cache to help inference on Julia 1.10
            # where the compiler can't track them across branches.
            _uType = typeof(cache.u0)
            _fuType = typeof(NonlinearSolveBase.get_fu(cache.caches[1]))
            _traceType = typeof(cache.caches[1].trace)
        end,
    ]

    cache_syms = [gensym("cache") for i in 1:N]
    sol_syms = [gensym("sol") for i in 1:N]
    u_result_syms = [gensym("u_result") for i in 1:N]

    push!(
        calls,
        quote
            if cache.retcode == ReturnCode.InitialFailure
                u = $(SII.state_values)(cache)::_uType
                return build_solution_less_specialize(
                    cache.prob, cache.alg, u,
                    $(Utils.evaluate_f)(cache.prob, u)::_fuType;
                    retcode = cache.retcode, stats = cache.stats,
                    trace = (cache.caches[1].trace::_traceType),
                    store_original = cache.alg.store_original
                )
            end
        end
    )

    # The per-subalgorithm attempt body, shared by the normal in-order pass and the
    # retention wrap-around pass below (`cache_syms[i]` is assigned unconditionally in
    # the first pass, and a given `i` runs in at most one of the two passes, so the
    # `sol`/`u_result` symbols can be shared).
    attempt_block = i -> quote
        if cache.retain_best && $(i) != cache.start_current
            # a `retain_best` reinit! only reinitialized the starting subcache;
            # escalation freshens each further subcache right before its attempt
            deferred_subcache_reinit!(cache, $(cache_syms[i]))
        end
        cache.alias_u0 && copyto!(cache.u0_aliased, cache.u0)
        $(sol_syms[i]) = CommonSolve.solve!($(cache_syms[i]))
        if SciMLBase.successful_retcode($(sol_syms[i]))
            stats = $(sol_syms[i]).stats
            cache.best = $(i)
            if cache.alias_u0
                copyto!(cache.u0, $(sol_syms[i]).u)
                $(u_result_syms[i]) = cache.u0::_uType
            else
                $(u_result_syms[i]) = $(sol_syms[i]).u::_uType
            end
            fu = NonlinearSolveBase.get_fu($(cache_syms[i]))::_fuType
            return build_solution_less_specialize(
                cache.prob, cache.alg, $(u_result_syms[i]), fu;
                retcode = $(sol_syms[i]).retcode, stats,
                original = $(sol_syms[i]), trace = ($(sol_syms[i]).trace::_traceType),
                store_original = cache.alg.store_original
            )
        elseif cache.alias_u0
            # For safety we need to maintain a copy of the solution
            $(u_result_syms[i]) = copy($(sol_syms[i]).u)
        end
        cache.current = $(i + 1)
    end

    for i in 1:N
        push!(
            calls,
            quote
                $(cache_syms[i]) = cache.caches[$(i)]
                if $(i) == cache.current
                    $(attempt_block(i))
                end
            end
        )
    end

    # Retention wrap-around (see the `NonlinearSolvePolyAlgorithm` docstring): when
    # `reinit!` armed `retain_best` and the retained subalgorithm's ladder started
    # past the algorithm's own `start_index`, a fully-failed pass continues with the
    # skipped cheaper subalgorithms before falling through to the lowest-residual
    # selection. The wrapped pass starts at `start_index` (never below it — retention
    # must not attempt subalgorithms the algorithm itself excludes) and stops where
    # the retained start already ran, so every subalgorithm of a full ladder run is
    # attempted exactly once.
    push!(
        calls,
        quote
            if cache.retain_best && !cache.wrapped &&
                    cache.start_current > cache.alg.start_index
                cache.wrapped = true
                cache.current = cache.alg.start_index
            end
        end
    )
    for i in 1:(N - 1)
        push!(
            calls,
            quote
                if cache.wrapped && $(i) == cache.current && $(i) < cache.start_current
                    $(attempt_block(i))
                end
            end
        )
    end

    resids = map(Base.Fix2(Symbol, :resid), cache_syms)
    for (sym, resid) in zip(cache_syms, resids)
        # Use get_fu instead of accessing .resid directly since caches have `fu`, not `resid`
        push!(calls, :($(resid) = @isdefined($(sym)) ? NonlinearSolveBase.get_fu($(sym)) : nothing))
    end
    push!(
        calls, quote
            fus = tuple($(Tuple(resids)...))
            # Use findmin_resids directly since fus already contains residual vectors from get_fu
            minfu, idx = findmin_resids(cache.prob, fus)
        end
    )
    for i in 1:N
        push!(
            calls,
            quote
                if idx == $(i)
                    u = cache.alias_u0 ? $(u_result_syms[i]) :
                        NonlinearSolveBase.get_u(cache.caches[$(i)])
                end
            end
        )
    end
    push!(
        calls,
        quote
            retcode = cache.caches[idx].retcode
            if cache.alias_u0
                copyto!(cache.u0, u)
                u = cache.u0
            end
            _trace = cache.caches[idx].trace::_traceType
            return build_solution_less_specialize(
                cache.prob, cache.alg, u::_uType, fus[idx]::_fuType;
                retcode, cache.stats, trace = _trace,
                store_original = cache.alg.store_original
            )
        end
    )

    return Expr(:block, calls...)
end

function _solve_without_solution!(cache::NonlinearSolvePolyAlgorithmCache)
    return CommonSolve.solve!(cache)
end

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm,
        args...; kwargs...
    )
    return __generated_polysolve(prob, alg, args...; kwargs...)
end

function SciMLBase.__solve(
        prob::AbstractNonlinearProblem, args...; default_set = false, second_time = false,
        kwargs...
    )
    return if second_time
        throw(NoDefaultAlgorithmError())
    elseif length(args) > 0 && !(first(args) isa AbstractNonlinearAlgorithm)
        throw(NonSolverError())
    else
        __solve(prob, nothing, args...; default_set = false, second_time = true, kwargs...)
    end
end

function __init(
        prob::AbstractNonlinearProblem, args...; default_set = false, second_time = false,
        kwargs...
    )
    return if second_time
        throw(NoDefaultAlgorithmError())
    elseif length(args) > 0 && !(
            first(args) isa
                Union{Nothing, AbstractDEAlgorithm, AbstractNonlinearAlgorithm}
        )
        throw(NonSolverError())
    else
        __init(prob, nothing, args...; default_set = false, second_time = true, kwargs...)
    end
end

@generated function __generated_polysolve(
        prob::AbstractNonlinearProblem, alg::NonlinearSolvePolyAlgorithm{Val{N}}, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias = NonlinearAliasSpecifier(alias_u0 = false), verbose = NonlinearVerbosity(),
        initializealg = NonlinearSolveDefaultInit(), kwargs...
    ) where {N}

    if verbose isa Bool
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(None())
        end
    elseif verbose isa AbstractVerbosityPreset
        verbose = NonlinearVerbosity(verbose)
    end

    sol_syms = [gensym("sol") for _ in 1:N]
    prob_syms = [gensym("prob") for _ in 1:N]
    u_result_syms = [gensym("u_result") for _ in 1:N]
    calls = [
        quote
            alias_u0 = alias.alias_u0
            current = alg.start_index
            if alias_u0 && !ArrayInterface.ismutable(prob.u0)
                @SciMLMessage("`alias_u0` has been set to `true`, but `u0` is
            immutable (checked using `ArrayInterface.ismutable``).", verbose, :alias_u0_immutable)
                alias_u0 = false  # If immutable don't care about aliasing
            end
        end,
    ]

    push!(
        calls,
        quote
            prob, success = $(run_initialization!)(prob, initializealg, prob)
            if !success
                u = $(SII.state_values)(prob)
                return build_solution_less_specialize(
                    prob, alg, u, $(Utils.evaluate_f)(prob, u);
                    retcode = $(ReturnCode.InitialFailure),
                    store_original = alg.store_original
                )
            end
        end
    )

    push!(
        calls, quote
            u0 = prob.u0
            u0_aliased = alias_u0 ? zero(u0) : u0
        end
    )
    for i in 1:N
        cur_sol = sol_syms[i]
        push!(
            calls,
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
                            $(cur_sol).trace, original = $(cur_sol),
                            store_original = alg.store_original
                        )
                    elseif alias_u0
                        # For safety we need to maintain a copy of the solution
                        $(u_result_syms[i]) = copy($(cur_sol).u)
                    end
                    current = $(i + 1)
                end
            end
        )
    end

    resids = map(Base.Fix2(Symbol, :resid), sol_syms)
    for (sym, resid) in zip(sol_syms, resids)
        push!(calls, :($(resid) = @isdefined($(sym)) ? $(sym).resid : nothing))
    end

    push!(
        calls, quote
            resids = tuple($(Tuple(resids)...))
            minfu, idx = findmin_resids(prob, resids)
        end
    )

    for i in 1:N
        push!(
            calls,
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
                        $(sol_syms[i]).trace, original = $(sol_syms[i]),
                        store_original = alg.store_original
                    )
                end
            end
        )
    end
    push!(calls, :(error("Current choices shouldn't get here!")))

    return Expr(:block, calls...)
end

"""
    step!(cache::AbstractNonlinearSolveCache, args...; kwargs...)

Perform one step of a nonlinear solver and mutate `cache` in place.

The public wrapper first checks whether the cache is still active, then calls
`NonlinearSolveBase.InternalAPI.step!`, updates the step counters, and enforces a time
limit when one is configured. It returns the value produced by the algorithm-specific
implementation, which is commonly `nothing`.

# Arguments

- `cache::AbstractNonlinearSolveCache`: the initialized stepping cache to advance.
- `args...`: positional arguments forwarded to the algorithm-specific implementation.

# Keywords

- `recompute_jacobian::Union{Nothing, Bool} = nothing`: whether to recompute a Jacobian
  when the algorithm uses one. `nothing` delegates the choice to the algorithm. This
  keyword is ignored or rejected by algorithms according to their own interface.
- `evaluate_residual::Bool = true`: a hint that the algorithm may skip the residual
  evaluation at the newly accepted iterate. It is honored only when
  [`supports_deferred_residual`](@ref) returns `true`; call [`refresh_residual!`](@ref)
  before reading a deferred residual.
- `kwargs...`: additional algorithm-specific keyword arguments.

# Returns

The value returned by `InternalAPI.step!`. The cache is mutated in place. Calling `step!`
on a terminated cache does nothing and returns `nothing`.

# Extension Rules

Solver packages implement `InternalAPI.step!`, not this `CommonSolve.step!` method. The
implementation must leave the cache state consistent with [`get_u`](@ref),
[`get_fu`](@ref), and the termination cache. The wrapper owns the top-level `nsteps` and
`stats.nsteps` increments, so an implementation should not increment those counters for
the same step.

# Examples

```julia
import NonlinearSolve

prob = NonlinearSolve.NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
cache = NonlinearSolve.init(prob, NonlinearSolve.NewtonRaphson())
NonlinearSolve.step!(cache)
```
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

"""
    NonlinearSolveNoInitCache <: AbstractNonlinearSolveCache

Cache returned by `init(prob, alg; kwargs...)` when `alg` has no algorithm-specific
`SciMLBase.__init` method. Every `SimpleNonlinearSolve` algorithm uses this form, for
example. It stores the problem and solve options so generic code can call `init` on any
nonlinear algorithm.

Unlike a stepping cache, it holds no iteration state and implements only part of the
[`AbstractNonlinearSolveCache`](@ref) interface. `solve!(cache)` runs the complete solve
and returns a `SciMLBase.NonlinearSolution`; that solution is the only record of the
iterations, and no iteration state is written back into the cache. `get_u(cache)` reads
the problem's initial state, while `SciMLBase.reinit!`, `get_abstol`, `get_reltol`, and the
`SymbolicIndexingInterface` accessors operate on the stored problem as usual.

`CommonSolve.step!`, `get_fu`, `get_nsteps`, and `cache.stats` are not available for this
cache. Generic code that drives caches one step at a time must detect this type and call
`solve!(cache)` instead.

# Fields

- `prob::AbstractNonlinearProblem`: the problem passed to `init`.
- `alg::AbstractNonlinearSolveAlgorithm`: the algorithm passed to `init`.
- `args::Tuple`: positional arguments forwarded to the eventual solve.
- `kwargs::Any`: keyword options forwarded to the eventual solve, including tolerances.
- `initializealg`: the initialization algorithm used before the solve.
- `retcode::SciMLBase.ReturnCode.T`: the initialization status, if initialization ran.
- `verbose`: the verbosity specification forwarded to the solve.

# Extension Rules

This cache is the fallback produced by the package's generic initialization method; solver
packages should not add `step!` methods to it to imitate a stepping cache. Use `isa
NonlinearSolveNoInitCache` only to select the complete `solve!` path, and use the generic
accessors for all other supported operations.

# Examples

```julia
import NonlinearSolve
import NonlinearSolveBase

prob = NonlinearSolve.NonlinearProblem((u, p) -> u^2 - p, 1.0, 2.0)
cache = NonlinearSolve.init(prob, NonlinearSolve.SimpleNewtonRaphson())
cache isa NonlinearSolveBase.NonlinearSolveNoInitCache
sol = NonlinearSolve.solve!(cache)
```
"""
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
    return get(cache.kwargs, :abstol, get_tolerance(nothing, eltype(cache.prob.u0)))
end
function get_reltol(cache::NonlinearSolveNoInitCache)
    return get(cache.kwargs, :reltol, get_tolerance(nothing, eltype(cache.prob.u0)))
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
    return print(io, "NonlinearSolveNoInitCache(alg = $(cache.alg))")
end

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...;
        initializealg = NonlinearSolveDefaultInit(), verbose = NonlinearVerbosity(),
        kwargs...
    )
    cache = NonlinearSolveNoInitCache(
        prob, alg, args, kwargs, initializealg, ReturnCode.Default, verbose
    )
    run_initialization!(cache)
    return cache
end

function CommonSolve.solve!(cache::NonlinearSolveNoInitCache)
    if cache.retcode == ReturnCode.InitialFailure
        u = SII.state_values(cache)
        return SciMLBase.build_solution(
            cache.prob, cache.alg, u, Utils.evaluate_f(cache.prob, u); cache.retcode
        )
    end
    return CommonSolve.solve(cache.prob, cache.alg, cache.args...; cache.kwargs...)
end


function _solve_without_solution!(cache::NonlinearSolveNoInitCache)
    return CommonSolve.solve!(cache)
end

function _solve_adjoint(
        prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...
    )
    alg = extract_alg(args, kwargs, prob.kwargs)
    _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)

    # Enzyme cannot differentiate through FunctionWrappers' `llvmcall`, so its
    # traced forward solve runs on the unwrapped function (see
    # `maybe_unwrap_prob_for_enzyme`, #940). That unwrap is keyed off the solver's
    # own autodiff, which for an MTK DAE initialization is ForwardDiff even when
    # the *outer* differentiation is Enzyme — so it does not fire here. Key off
    # the originator instead: the EnzymeOriginator adjoint primal must have the
    # same (unwrapped) type as Enzyme's traced forward, otherwise the custom
    # `solve_up` rule's returned primal type mismatches the inferred return type
    # (`EnzymeRuntimeException: Expected return type of primal to be ...`).
    if originator isa SciMLBase.EnzymeOriginator
        if _prob.p isa SciMLBase.DespecializedParameters
            _prob = _unwrap_despecialized_problem(_prob)
        elseif is_fw_wrapped(_prob.f.f)
            @set! _prob.f.f = get_raw_f(_prob.f.f)
        end
    end

    if has_kwargs(_prob)
        # `::NamedTuple` assert keeps dispatch off the invalidation-prone `merge(::Any, ::Pairs)` path
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs)::NamedTuple, kwargs)
    end

    return if length(args) > 1
        _concrete_solve_adjoint(
            _prob, alg, sensealg, u0, p, originator,
            Base.tail(args)...; kwargs...
        )
    else
        _concrete_solve_adjoint(_prob, alg, sensealg, u0, p, originator; kwargs...)
    end
end

function _solve_forward(
        prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...
    )
    alg = extract_alg(args, kwargs, prob.kwargs)
    _prob = get_concrete_problem(prob; u0 = u0, p = p, kwargs...)

    if has_kwargs(_prob)
        # `::NamedTuple` assert keeps dispatch off the invalidation-prone `merge(::Any, ::Pairs)` path
        kwargs = isempty(_prob.kwargs) ? kwargs : merge(values(_prob.kwargs)::NamedTuple, kwargs)
    end

    return if length(args) > 1
        _concrete_solve_forward(
            _prob, alg, sensealg, u0, p, originator,
            Base.tail(args)...; kwargs...
        )
    else
        _concrete_solve_forward(_prob, alg, sensealg, u0, p, originator; kwargs...)
    end
end

function maybe_wrap_f(prob::AbstractNonlinearProblem)
    # AutoDePSpecialize opaque-`p` path (packs `p` + wraps `f` together).
    opaque = maybe_opaque_wrap(prob)
    opaque === nothing || return opaque
    prob = _despecialize_parameters(prob)
    wrapped_f = maybe_wrap_nonlinear_f(prob)
    wrapped_f === prob.f.f && return prob
    f = SciMLBase.unwrapped_f(prob.f, wrapped_f)
    return SciMLBase.remake(prob; f)
end


"""
    get_concrete_problem(prob; kwargs...)

Return the concrete nonlinear problem used by solver initialization after applying state
and parameter overrides, numeric promotion, symbolic updates, and the problem function's
specialization policy. Solver packages can use this developer API before constructing a
cache or composing nonlinear subproblems.
"""
function get_concrete_problem(prob::NonlinearProblem; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SII.state_values(prob), p = SII.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs)
    u0 = get_concrete_u0(prob, true, nothing, kwargs)
    u0 = promote_u0(u0, p, nothing)
    prob = remake(prob; u0 = u0, p = p, lb = prob.lb, ub = prob.ub)
    return maybe_wrap_f(prob)
end

function get_concrete_problem(prob::NonlinearLeastSquaresProblem; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SII.state_values(prob), p = SII.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs)
    u0 = get_concrete_u0(prob, true, nothing, kwargs)
    u0 = promote_u0(u0, p, nothing)
    prob = remake(prob; u0 = u0, p = p, lb = prob.lb, ub = prob.ub)
    return maybe_wrap_f(prob)
end

function get_concrete_problem(prob::ImmutableNonlinearProblem; kwargs...)
    u0 = get_concrete_u0(prob, true, nothing, kwargs)
    u0 = promote_u0(u0, prob.p, nothing)
    p = get_concrete_p(prob, kwargs)
    prob = remake(prob; u0 = u0, p = p)
    return maybe_wrap_f(prob)
end

function get_concrete_problem(prob::SteadyStateProblem; kwargs...)
    oldprob = prob
    prob = get_updated_symbolic_problem(SciMLBase.get_root_indp(prob), prob; kwargs...)
    if prob !== oldprob
        kwargs = (; kwargs..., u0 = SII.state_values(prob), p = SII.parameter_values(prob))
    end
    p = get_concrete_p(prob, kwargs)
    u0 = get_concrete_u0(prob, true, Inf, kwargs)
    u0 = promote_u0(u0, p, nothing)
    return remake(prob; u0 = u0, p = p)
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
        kwargs...
    )
    prob, success = hack_null_solution_init(prob)
    retcode = success ? ReturnCode.Success : ReturnCode.InitialFailure
    return SciMLBase.build_solution(prob, nothing, Float64[], nothing; retcode)
end

function build_null_solution(
        prob::NonlinearLeastSquaresProblem,
        args...; abstol = 1.0e-6, kwargs...
    )
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

    return SciMLBase.build_solution(prob, nothing, Float64[], resid; retcode)
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
