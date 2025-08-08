const allowedkeywords = (:dense,
    :saveat,
    :save_idxs,
    :tstops,
    :tspan,
    :d_discontinuities,
    :save_everystep,
    :save_on,
    :save_start,
    :save_end,
    :initialize_save,
    :adaptive,
    :abstol,
    :reltol,
    :dt,
    :dtmax,
    :dtmin,
    :force_dtmin,
    :internalnorm,
    :controller,
    :gamma,
    :beta1,
    :beta2,
    :qmax,
    :qmin,
    :qsteady_min,
    :qsteady_max,
    :qoldinit,
    :failfactor,
    :calck,
    :alias_u0,
    :maxiters,
    :maxtime,
    :callback,
    :isoutofdomain,
    :unstable_check,
    :verbose,
    :merge_callbacks,
    :progress,
    :progress_steps,
    :progress_name,
    :progress_message,
    :progress_id,
    :timeseries_errors,
    :dense_errors,
    :weak_timeseries_errors,
    :weak_dense_errors,
    :wrap,
    :calculate_error,
    :initializealg,
    :alg,
    :save_noise,
    :delta,
    :seed,
    :alg_hints,
    :kwargshandle,
    :trajectories,
    :batch_size,
    :sensealg,
    :advance_to_tstop,
    :stop_at_next_tstop,
    :u0,
    :p,
    # These two are from the default algorithm handling
    :default_set,
    :second_time,
    # This is for DiffEqDevTools
    :prob_choice,
    # Jump problems
    :alias_jump,
    # This is for copying/deepcopying noise in StochasticDiffEq
    :alias_noise,
    # This is for SimpleNonlinearSolve handling for batched Nonlinear Solves
    :batch,
    # Shooting method in BVP needs to differentiate between these two categories
    :nlsolve_kwargs,
    :odesolve_kwargs,
    # If Solvers which internally use linsolve
    :linsolve_kwargs,
    # Solvers internally using EnsembleProblem
    :ensemblealg,
    # Fine Grained Control of Tracing (Storing and Logging) during Solve
    :show_trace,
    :trace_level,
    :store_trace,
    # Termination condition for solvers
    :termination_condition,
    # For AbstractAliasSpecifier
    :alias,
    # Parameter estimation with BVP
    :fit_parameters)

const KWARGWARN_MESSAGE = """
Unrecognized keyword arguments found.
The only allowed keyword arguments to `solve` are:
$allowedkeywords

See https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/ for more details.

Set kwargshandle=KeywordArgError for an error message.
Set kwargshandle=KeywordArgSilent to ignore this message.
"""

const KWARGERROR_MESSAGE = """
     Unrecognized keyword arguments found.
     The only allowed keyword arguments to `solve` are:
     $allowedkeywords

     See https://docs.sciml.ai/NonlinearSolve/stable/basics/solve/ for more details.
     """

struct CommonKwargError <: Exception
    kwargs::Any
end

function Base.showerror(io::IO, e::CommonKwargError)
    println(io, KWARGERROR_MESSAGE)
    notin = collect(map(x -> x ∉ allowedkeywords, keys(e.kwargs)))
    unrecognized = collect(keys(e.kwargs))[notin]
    print(io, "Unrecognized keyword arguments: ")
    printstyled(io, unrecognized; bold = true, color = :red)
    print(io, "\n\n")
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

@enum KeywordArgError KeywordArgWarn KeywordArgSilent

const INCOMPATIBLE_U0_MESSAGE = """
                                Initial condition incompatible with functional form.
                                Detected an in-place function with an initial condition of type Number or SArray.
                                This is incompatible because Numbers cannot be mutated, i.e.
                                `x = 2.0; y = 2.0; x .= y` will error.

                                If using a immutable initial condition type, please use the out-of-place form.
                                I.e. define the function `du=f(u,p,t)` instead of attempting to "mutate" the immutable `du`.

                                If your differential equation function was defined with multiple dispatches and one is
                                in-place, then the automatic detection will choose in-place. In this case, override the
                                choice in the problem constructor, i.e. `ODEProblem{false}(f,u0,tspan,p,kwargs...)`.

                                For a longer discussion on mutability vs immutability and in-place vs out-of-place, see:
                                https://diffeq.sciml.ai/stable/tutorials/faster_ode_example/#Example-Accelerating-a-Non-Stiff-Equation:-The-Lorenz-Equation
                                """

struct IncompatibleInitialConditionError <: Exception end

function Base.showerror(io::IO, e::IncompatibleInitialConditionError)
    print(io, INCOMPATIBLE_U0_MESSAGE)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

const NO_DEFAULT_ALGORITHM_MESSAGE = """
                                     Default algorithm choices require NonlinearSolve.jl.
                                     Please specify an algorithm (e.g., `solve(prob, NewtonRaphson())` or
                                     init(prob, NewtonRaphson()) or 
                                     import NonlinearSolve.jl directly.

                                     You can find the list of available solvers at https://diffeq.sciml.ai/stable/solvers/ode_solve/
                                     and its associated pages.
                                     """

struct NoDefaultAlgorithmError <: Exception end

function Base.showerror(io::IO, e::NoDefaultAlgorithmError)
    print(io, NO_DEFAULT_ALGORITHM_MESSAGE)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end 

const NON_SOLVER_MESSAGE = """
                           The arguments to solve are incorrect.
                           The second argument must be a solver choice, `solve(prob,alg)`
                           where `alg` is a `<: AbstractDEAlgorithm`, e.g. `Tsit5()`.

                           Please double check the arguments being sent to the solver.

                           You can find the list of available solvers at https://diffeq.sciml.ai/stable/solvers/ode_solve/
                           and its associated pages.
                           """

struct NonSolverError <: Exception end

function Base.showerror(io::IO, e::NonSolverError)
    print(io, NON_SOLVER_MESSAGE)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

const DIRECT_AUTODIFF_INCOMPATABILITY_MESSAGE = """
                                                Incompatible solver + automatic differentiation pairing.
                                                The chosen automatic differentiation algorithm requires the ability
                                                for compiler transforms on the code which is only possible on pure-Julia
                                                solvers such as those from OrdinaryDiffEq.jl. Direct differentiation methods
                                                which require this ability include:

                                                - Direct use of ForwardDiff.jl on the solver
                                                - `ForwardDiffSensitivity`, `ReverseDiffAdjoint`, `TrackerAdjoint`, and `ZygoteAdjoint`
                                                  sensealg choices for adjoint differentiation.

                                                Either switch the choice of solver to a pure Julia method, or change the automatic
                                                differentiation method to one that does not require such transformations.

                                                For more details on automatic differentiation, adjoint, and sensitivity analysis
                                                of differential equations, see the documentation page:

                                                https://diffeq.sciml.ai/stable/analysis/sensitivity/
                                                """

struct DirectAutodiffError <: Exception end

function Base.showerror(io::IO, e::DirectAutodiffError)
    println(io, DIRECT_AUTODIFF_INCOMPATABILITY_MESSAGE)
    println(io, TruncatedStacktraces.VERBOSE_MSG)
end

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
        if merge_callbacks && haskey(_prob.kwargs, :callback) && haskey(kwargs, :callback)
            kwargs_temp = NamedTuple{
                Base.diff_names(Base._nt_names(values(kwargs)),
                (:callback,))}(values(kwargs))
            callbacks = NamedTuple{(:callback,)}((DiffEqBase.CallbackSet(
                _prob.kwargs[:callback],
                values(kwargs).callback),))
            kwargs = merge(kwargs_temp, callbacks)
        end
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

function init_up(prob::AbstractNonlinearProblem, sensealg, u0, p, args...; kwargs...)
    alg = extract_alg(args, kwargs, has_kwargs(prob) ? prob.kwargs : kwargs)
    if isnothing(alg) || !(alg isa AbstractNonlinearAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob, !(prob isa DiscreteProblem); u0 = u0,
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
        if merge_callbacks && haskey(_prob.kwargs, :callback) && haskey(kwargs, :callback)
            kwargs_temp = NamedTuple{
                Base.diff_names(Base._nt_names(values(kwargs)),
                    (:callback,))}(values(kwargs))
            callbacks = NamedTuple{(:callback,)}((DiffEqBase.CallbackSet(
                _prob.kwargs[:callback],
                values(kwargs).callback),))
            kwargs = merge(kwargs_temp, callbacks)
        end
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
        prob::AbstractNonlinearProblem, alg::AbstractNonlinearSolveAlgorithm, args...;
        kwargs...
)
    cache = SciMLBase.__init(prob, alg, args...; kwargs...)
    return CommonSolve.solve!(cache)
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

has_kwargs(_prob::AbstractNonlinearProblem) = has_kwargs(typeof(_prob))
Base.@pure __has_kwargs(::Type{T}) where {T} = :kwargs ∈ fieldnames(T)
has_kwargs(::Type{T}) where {T} = __has_kwargs(T)

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

function _solve_adjoint(prob, sensealg, u0, p, originator, args...; merge_callbacks = true,
        kwargs...)
    alg = extract_alg(args, kwargs, prob.kwargs)
    if isnothing(alg) || !(alg isa AbstractDEAlgorithm) # Default algorithm handling
        _prob = get_concrete_problem(prob, !(prob isa DiscreteProblem); u0 = u0,
            p = p, kwargs...)
    else
        _prob = get_concrete_problem(prob, isadaptive(alg); u0 = u0, p = p, kwargs...)
    end

    if has_kwargs(_prob)
        if merge_callbacks && haskey(_prob.kwargs, :callback) && haskey(kwargs, :callback)
            kwargs_temp = NamedTuple{
                Base.diff_names(Base._nt_names(values(kwargs)),
                (:callback,))}(values(kwargs))
            callbacks = NamedTuple{(:callback,)}((DiffEqBase.CallbackSet(
                _prob.kwargs[:callback],
                values(kwargs).callback),))
            kwargs = merge(kwargs_temp, callbacks)
        end
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
        _prob = get_concrete_problem(prob, !(prob isa DiscreteProblem); u0 = u0,
            p = p, kwargs...)
    else
        _prob = get_concrete_problem(prob, isadaptive(alg); u0 = u0, p = p, kwargs...)
    end

    if has_kwargs(_prob)
        if merge_callbacks && haskey(_prob.kwargs, :callback) && haskey(kwargs, :callback)
            kwargs_temp = NamedTuple{
                Base.diff_names(Base._nt_names(values(kwargs)),
                (:callback,))}(values(kwargs))
            callbacks = NamedTuple{(:callback,)}((DiffEqBase.CallbackSet(
                _prob.kwargs[:callback],
                values(kwargs).callback),))
            kwargs = merge(kwargs_temp, callbacks)
        end
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
        prob::NonlinearProblem,
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

@inline function extract_alg(solve_args, solve_kwargs, prob_kwargs)
    if isempty(solve_args) || isnothing(first(solve_args))
        if haskey(solve_kwargs, :alg)
            solve_kwargs[:alg]
        elseif haskey(prob_kwargs, :alg)
            prob_kwargs[:alg]
        else
            nothing
        end
    elseif first(solve_args) isa SciMLBase.AbstractSciMLAlgorithm &&
           !(first(solve_args) isa SciMLBase.EnsembleAlgorithm)
        first(solve_args)
    else
        nothing
    end
end

function get_concrete_u0(prob, isadapt, t0, kwargs)
    if eval_u0(prob.u0)
        u0 = prob.u0(prob.p, t0)
    elseif haskey(kwargs, :u0)
        u0 = kwargs[:u0]
    else
        u0 = prob.u0
    end

    isadapt && eltype(u0) <: Integer && (u0 = float.(u0))

    _u0 = handle_distribution_u0(u0)

    if isinplace(prob) && (_u0 isa Number || _u0 isa SArray)
        throw(IncompatibleInitialConditionError())
    end

    if _u0 isa Tuple
        throw(TupleStateError())
    end

    _u0
end

function get_concrete_p(prob, kwargs)
    if haskey(kwargs, :p)
        p = kwargs[:p]
    else
        p = prob.p
    end
end

eval_u0(u0::Function) = true
eval_u0(u0) = false

handle_distribution_u0(_u0) = _u0

anyeltypedual(x) = anyeltypedual(x, Val{0})
anyeltypedual(x, counter) = Any

function promote_u0(u0, p, t0)
    if SciMLStructures.isscimlstructure(p)
        _p = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1]
        if !isequal(_p, p)
            return promote_u0(u0, _p, t0)
        end
    end
    Tu = eltype(u0)
    if isdualtype(Tu)
        return u0
    end
    Tp = anyeltypedual(p, Val{0})
    if Tp == Any
        Tp = Tu
    end
    Tt = anyeltypedual(t0, Val{0})
    if Tt == Any
        Tt = Tu
    end
    Tcommon = promote_type(Tu, Tp, Tt)
    return if isdualtype(Tcommon)
        Tcommon.(u0)
    else
        u0
    end
end

function promote_u0(u0::AbstractArray{<:Complex}, p, t0)
    if SciMLStructures.isscimlstructure(p)
        _p = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)[1]
        if !isequal(_p, p)
            return promote_u0(u0, _p, t0)
        end
    end
    Tu = real(eltype(u0))
    if isdualtype(Tu)
        return u0
    end
    Tp = anyeltypedual(p, Val{0})
    if Tp == Any
        Tp = Tu
    end
    Tt = anyeltypedual(t0, Val{0})
    if Tt == Any
        Tt = Tu
    end
    Tcommon = promote_type(eltype(u0), Tp, Tt)
    return if isdualtype(real(Tcommon))
        Tcommon.(u0)
    else
        u0
    end
end

function checkkwargs(kwargshandle; kwargs...)
    if any(x -> x ∉ allowedkeywords, keys(kwargs))
        if kwargshandle == KeywordArgError
            throw(CommonKwargError(kwargs))
        elseif kwargshandle == KeywordArgWarn
            @warn KWARGWARN_MESSAGE
            unrecognized = setdiff(keys(kwargs), allowedkeywords)
            print("Unrecognized keyword arguments: ")
            printstyled(unrecognized; bold = true, color = :red)
            print("\n\n")
        else
            @assert kwargshandle == KeywordArgSilent
        end
    end
end