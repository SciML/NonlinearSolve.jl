"""
    GeneralizedFirstOrderAlgorithm(;
        descent, linesearch = missing,
        trustregion = missing, autodiff = nothing, vjp_autodiff = nothing,
        jvp_autodiff = nothing, max_shrink_times::Int = typemax(Int),
        concrete_jac = Val(false), name::Symbol = :unknown
    )

This is a Generalization of First-Order (uses Jacobian) Nonlinear Solve Algorithms. The most
common example of this is Newton-Raphson Method.

First Order here refers to the order of differentiation, and should not be confused with the
order of convergence.

### Keyword Arguments

  - `trustregion`: Globalization using a Trust Region Method. This needs to follow the
    [`NonlinearSolveBase.AbstractTrustRegionMethod`](@ref) interface.
  - `descent`: The descent method to use to compute the step. This needs to follow the
    [`NonlinearSolveBase.AbstractDescentDirection`](@ref) interface.
  - `max_shrink_times`: The maximum number of times the trust region radius can be shrunk
    before the algorithm terminates.
"""
@concrete struct GeneralizedFirstOrderAlgorithm <: AbstractNonlinearSolveAlgorithm
    linesearch
    trustregion
    descent
    forcing
    max_shrink_times::Int

    autodiff
    vjp_autodiff
    jvp_autodiff

    concrete_jac <: Union{Val{false}, Val{true}}
    name::Symbol
end

function GeneralizedFirstOrderAlgorithm(;
        descent, linesearch = missing, trustregion = missing, autodiff = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing, max_shrink_times::Int = typemax(Int),
        concrete_jac = Val(false), forcing = nothing, name::Symbol = :unknown
    )
    concrete_jac = concrete_jac isa Bool ? Val(concrete_jac) :
        (concrete_jac isa Val ? concrete_jac : Val(concrete_jac !== nothing))
    return GeneralizedFirstOrderAlgorithm(
        linesearch, trustregion, descent, forcing, max_shrink_times,
        autodiff, vjp_autodiff, jvp_autodiff,
        concrete_jac, name
    )
end

@concrete mutable struct GeneralizedFirstOrderAlgorithmCache <: AbstractNonlinearSolveCache
    # Basic Requirements
    fu
    u
    u_cache
    p
    alg <: GeneralizedFirstOrderAlgorithm
    prob <: AbstractNonlinearProblem
    globalization <: Union{Val{:LineSearch}, Val{:TrustRegion}, Val{:None}}

    # Internal Caches
    jac_cache
    descent_cache
    forcing_cache
    linesearch_cache
    trustregion_cache

    # Counters
    stats::NLStats
    nsteps::Int
    maxiters::Int
    maxtime
    max_shrink_times::Int

    # Timer
    timer
    total_time::Float64

    # State Affect
    make_new_jacobian::Bool

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    kwargs

    initializealg

    verbose
end

function SciMLBase.get_du(cache::GeneralizedFirstOrderAlgorithmCache)
    return SciMLBase.get_du(cache.descent_cache)
end
function NonlinearSolveBase.set_du!(cache::GeneralizedFirstOrderAlgorithmCache, δu)
    return NonlinearSolveBase.set_du!(cache.descent_cache, δu)
end

function InternalAPI.reinit_self!(
        cache::GeneralizedFirstOrderAlgorithmCache, args...; p = cache.p, u0 = cache.u,
        alias_u0::Bool = hasproperty(cache, :alias_u0) ? cache.alias_u0 : false,
        maxiters = hasproperty(cache, :maxiters) ? cache.maxiters : 1000,
        maxtime = hasproperty(cache, :maxtime) ? cache.maxtime : nothing, kwargs...
    )
    Utils.reinit_common!(cache, u0, p, alias_u0)

    InternalAPI.reinit!(cache.stats)
    cache.nsteps = 0
    cache.maxiters = maxiters
    cache.maxtime = maxtime
    cache.total_time = 0.0
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    cache.make_new_jacobian = true

    NonlinearSolveBase.reset!(cache.trace)
    SciMLBase.reinit!(
        cache.termination_cache, NonlinearSolveBase.get_fu(cache),
        NonlinearSolveBase.get_u(cache); kwargs...
    )
    NonlinearSolveBase.reset_timer!(cache.timer)
    return
end

NonlinearSolveBase.@internal_caches(
    GeneralizedFirstOrderAlgorithmCache,
    :jac_cache, :descent_cache, :linesearch_cache, :trustregion_cache, :forcing_cache
)

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::GeneralizedFirstOrderAlgorithm, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false), maxiters = 1000,
        abstol = nothing, reltol = nothing, maxtime = nothing,
        termination_condition = nothing, internalnorm::IN = L2_NORM, verbose = NonlinearVerbosity(),
        linsolve_kwargs = (;), initializealg = NonlinearSolveBase.NonlinearSolveDefaultInit(), kwargs...
    ) where {IN}
    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    @set! alg.autodiff = NonlinearSolveBase.select_jacobian_autodiff(prob, alg.autodiff)
    provided_jvp_autodiff = alg.jvp_autodiff !== nothing
    @set! alg.jvp_autodiff = if !provided_jvp_autodiff && alg.autodiff !== nothing &&
            (
            ADTypes.mode(alg.autodiff) isa ADTypes.ForwardMode ||
                ADTypes.mode(alg.autodiff) isa
                ADTypes.ForwardOrReverseMode
        )
        NonlinearSolveBase.select_forward_mode_autodiff(prob, alg.autodiff)
    else
        NonlinearSolveBase.select_forward_mode_autodiff(prob, alg.jvp_autodiff)
    end
    provided_vjp_autodiff = alg.vjp_autodiff !== nothing
    @set! alg.vjp_autodiff = if !provided_vjp_autodiff && alg.autodiff !== nothing &&
            (
            ADTypes.mode(alg.autodiff) isa ADTypes.ReverseMode ||
                ADTypes.mode(alg.autodiff) isa
                ADTypes.ForwardOrReverseMode
        )
        NonlinearSolveBase.select_reverse_mode_autodiff(prob, alg.autodiff)
    else
        NonlinearSolveBase.select_reverse_mode_autodiff(prob, alg.vjp_autodiff)
    end

    if verbose isa Bool
        if verbose
            verbose = NonlinearVerbosity()
        else
            verbose = NonlinearVerbosity(None())
        end
    elseif verbose isa AbstractVerbosityPreset
        verbose = NonlinearVerbosity(verbose)
    end

    timer = get_timer_output()
    @static_timeit timer "cache construction" begin
        u = Utils.maybe_unaliased(prob.u0, alias_u0)
        fu = Utils.evaluate_f(prob, u)
        @bb u_cache = copy(u)

        linsolve = NonlinearSolveBase.get_linear_solver(alg.descent)

        abstol, reltol,
            termination_cache = NonlinearSolveBase.init_termination_cache(
            prob, abstol, reltol, fu, u, termination_condition, Val(:regular)
        )
        linsolve_kwargs = merge((; verbose = verbose.linear_verbosity, abstol, reltol), linsolve_kwargs)

        jac_cache = NonlinearSolveBase.construct_jacobian_cache(
            prob, alg, prob.f, fu, u, prob.p;
            stats, alg.autodiff, linsolve, alg.jvp_autodiff, alg.vjp_autodiff
        )
        J = reused_jacobian(jac_cache, u)

        descent_cache = InternalAPI.init(
            prob, alg.descent, J, fu, u; stats, abstol, reltol, internalnorm,
            linsolve_kwargs, timer
        )
        du = SciMLBase.get_du(descent_cache)

        has_linesearch = alg.linesearch !== missing && alg.linesearch !== nothing
        has_trustregion = alg.trustregion !== missing && alg.trustregion !== nothing
        has_forcing = alg.forcing !== missing && alg.forcing !== nothing && !(u isa Number) && !(J isa Diagonal)

        if has_trustregion && has_linesearch
            error("TrustRegion and LineSearch methods are algorithmically incompatible.")
        end

        globalization = Val(:None)
        linesearch_cache = nothing
        trustregion_cache = nothing
        forcing_cache = nothing

        if has_trustregion
            NonlinearSolveBase.supports_trust_region(alg.descent) ||
                error("Trust Region not supported by $(alg.descent).")
            trustregion_cache = InternalAPI.init(
                prob, alg.trustregion, prob.f, fu, u, prob.p;
                alg.vjp_autodiff, alg.jvp_autodiff, stats, internalnorm, kwargs...
            )
            globalization = Val(:TrustRegion)
        end

        if has_linesearch
            NonlinearSolveBase.supports_line_search(alg.descent) ||
                error("Line Search not supported by $(alg.descent).")
            linesearch_cache = CommonSolve.init(
                prob, alg.linesearch, fu, u; stats, internalnorm,
                autodiff = ifelse(
                    provided_jvp_autodiff, alg.jvp_autodiff, alg.vjp_autodiff
                ),
                kwargs...
            )
            globalization = Val(:LineSearch)
        end

        if has_forcing
            forcing_cache = InternalAPI.init(
                prob, alg.forcing, fu, u, u, prob.p; stats, internalnorm,
                autodiff = ifelse(
                    provided_jvp_autodiff, alg.jvp_autodiff, alg.vjp_autodiff
                ),
                verbose,
                kwargs...
            )
        end

        trace = NonlinearSolveBase.init_nonlinearsolve_trace(
            prob, alg, u, fu, J, du; kwargs...
        )

        cache = GeneralizedFirstOrderAlgorithmCache(
            fu, u, u_cache, prob.p, alg, prob, globalization,
            jac_cache, descent_cache, forcing_cache, linesearch_cache, trustregion_cache,
            stats, 0, maxiters, maxtime, alg.max_shrink_times, timer,
            0.0, true, termination_cache, trace, ReturnCode.Default, false, kwargs,
            initializealg, verbose
        )
        NonlinearSolveBase.run_initialization!(cache)
    end

    return cache
end

function InternalAPI.step!(
        cache::GeneralizedFirstOrderAlgorithmCache;
        recompute_jacobian::Union{Nothing, Bool} = nothing
    )
    @static_timeit cache.timer "jacobian" begin
        if (recompute_jacobian === nothing || recompute_jacobian) && cache.make_new_jacobian
            J = cache.jac_cache(cache.u)
            new_jacobian = true
        else
            J = reused_jacobian(cache.jac_cache, cache.u)
            new_jacobian = false
        end
    end

    has_forcing = cache.forcing_cache !== nothing && cache.forcing_cache !== missing && !(cache.u isa Number) && !(J isa Diagonal)

    if has_forcing
        pre_step_forcing!(cache.forcing_cache, cache.descent_cache, J, cache.u, cache.fu, cache.nsteps)
    end

    @static_timeit cache.timer "descent" begin
        if cache.trustregion_cache !== nothing &&
                hasfield(typeof(cache.trustregion_cache), :trust_region)
            descent_result = InternalAPI.solve!(
                cache.descent_cache, J, cache.fu, cache.u;
                new_jacobian, cache.trustregion_cache.trust_region, cache.kwargs...
            )
        else
            descent_result = InternalAPI.solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian, cache.kwargs...
            )
        end
    end

    if !descent_result.linsolve_success
        if new_jacobian
            # Jacobian Information is current and linear solve failed terminate the solve
            cache.retcode = ReturnCode.InternalLinearSolveFailed
            cache.force_stop = true
            return
        else
            # Jacobian Information is not current and linear solve failed, recompute it
            @SciMLMessage("Linear Solve Failed but Jacobian information is not current. Retrying with updated Jacobian. \
                Retrying with updated Jacobian.", cache.verbose, :linsolve_failed_noncurrent)
            # In the 2nd call the `new_jacobian` is guaranteed to be `true`.
            cache.make_new_jacobian = true
            InternalAPI.step!(cache; recompute_jacobian = true, cache.kwargs...)
            return
        end
    end

    δu, descent_intermediates = descent_result.δu, descent_result.extras

    if descent_result.success
        if has_forcing
            post_step_forcing!(cache.forcing_cache, J, cache.u, cache.fu, δu, cache.nsteps)
        end

        cache.make_new_jacobian = true
        if cache.globalization isa Val{:LineSearch}
            @static_timeit cache.timer "linesearch" begin
                linesearch_sol = CommonSolve.solve!(cache.linesearch_cache, cache.u, δu)
                linesearch_failed = !SciMLBase.successful_retcode(linesearch_sol.retcode)
                α = linesearch_sol.step_size
            end
            if linesearch_failed
                cache.retcode = ReturnCode.InternalLineSearchFailed
                cache.force_stop = true
            end
            @static_timeit cache.timer "step" begin
                @bb axpy!(α, δu, cache.u)
                Utils.evaluate_f!(cache, cache.u, cache.p)
            end
        elseif cache.globalization isa Val{:TrustRegion}
            @static_timeit cache.timer "trustregion" begin
                tr_accepted, u_new,
                    fu_new = InternalAPI.solve!(
                    cache.trustregion_cache, J, cache.fu, cache.u, δu, descent_intermediates
                )
                if tr_accepted
                    @bb copyto!(cache.u, u_new)
                    @bb copyto!(cache.fu, fu_new)
                    α = true
                else
                    α = false
                    cache.make_new_jacobian = false
                end
                if hasfield(typeof(cache.trustregion_cache), :shrink_counter) &&
                        cache.trustregion_cache.shrink_counter > cache.max_shrink_times
                    cache.retcode = ReturnCode.ShrinkThresholdExceeded
                    cache.force_stop = true
                end
            end
        elseif cache.globalization isa Val{:None}
            @static_timeit cache.timer "step" begin
                @bb axpy!(1, δu, cache.u)
                Utils.evaluate_f!(cache, cache.u, cache.p)
            end
            α = true
        else
            error("Unknown Globalization Strategy: $(cache.globalization). Allowed values \
                   are (:LineSearch, :TrustRegion, :None)")
        end
        NonlinearSolveBase.check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
    else
        α = false
        cache.make_new_jacobian = false
    end

    update_trace!(cache, α)
    @bb copyto!(cache.u_cache, cache.u)

    NonlinearSolveBase.callback_into_cache!(cache)

    return nothing
end
