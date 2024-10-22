"""
    GeneralizedFirstOrderAlgorithm{concrete_jac, name}(; descent, linesearch = missing,
        trustregion = missing, jacobian_ad = nothing, forward_ad = nothing,
        reverse_ad = nothing, max_shrink_times::Int = typemax(Int))
    GeneralizedFirstOrderAlgorithm(; concrete_jac = nothing, name::Symbol = :unknown,
        kwargs...)

This is a Generalization of First-Order (uses Jacobian) Nonlinear Solve Algorithms. The most
common example of this is Newton-Raphson Method.

First Order here refers to the order of differentiation, and should not be confused with the
order of convergence.

`trustregion` and `linesearch` cannot be specified together.

### Keyword Arguments

  - `trustregion`: Globalization using a Trust Region Method. This needs to follow the
    [`NonlinearSolve.AbstractTrustRegionMethod`](@ref) interface.
  - `descent`: The descent method to use to compute the step. This needs to follow the
    [`NonlinearSolve.AbstractDescentAlgorithm`](@ref) interface.
  - `max_shrink_times`: The maximum number of times the trust region radius can be shrunk
    before the algorithm terminates.
"""
@concrete struct GeneralizedFirstOrderAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    trustregion
    descent
    max_shrink_times::Int
    jacobian_ad
    forward_ad
    reverse_ad
end

function __show_algorithm(io::IO, alg::GeneralizedFirstOrderAlgorithm, name, indent)
    modifiers = String[]
    __is_present(alg.linesearch) && push!(modifiers, "linesearch = $(alg.linesearch)")
    __is_present(alg.trustregion) && push!(modifiers, "trustregion = $(alg.trustregion)")
    push!(modifiers, "descent = $(alg.descent)")
    __is_present(alg.jacobian_ad) && push!(modifiers, "jacobian_ad = $(alg.jacobian_ad)")
    __is_present(alg.forward_ad) && push!(modifiers, "forward_ad = $(alg.forward_ad)")
    __is_present(alg.reverse_ad) && push!(modifiers, "reverse_ad = $(alg.reverse_ad)")
    spacing = " "^indent * "   "
    spacing_last = " "^indent
    print(io, "$(name)(\n$(spacing)$(join(modifiers, ",\n$(spacing)"))\n$(spacing_last))")
end

function GeneralizedFirstOrderAlgorithm(;
        concrete_jac = nothing, name::Symbol = :unknown, kwargs...)
    return GeneralizedFirstOrderAlgorithm{concrete_jac, name}(; kwargs...)
end

function GeneralizedFirstOrderAlgorithm{concrete_jac, name}(;
        descent, linesearch = missing, trustregion = missing,
        jacobian_ad = nothing, forward_ad = nothing, reverse_ad = nothing,
        max_shrink_times::Int = typemax(Int)) where {concrete_jac, name}
    forward_ad = ifelse(forward_ad !== nothing,
        forward_ad,
        ifelse(
            jacobian_ad !== nothing && ADTypes.mode(jacobian_ad) isa ADTypes.ForwardMode,
            jacobian_ad, nothing))
    reverse_ad = ifelse(reverse_ad !== nothing,
        reverse_ad,
        ifelse(
            jacobian_ad !== nothing && ADTypes.mode(jacobian_ad) isa ADTypes.ReverseMode,
            jacobian_ad, nothing))

    return GeneralizedFirstOrderAlgorithm{concrete_jac, name}(
        linesearch, trustregion, descent, max_shrink_times,
        jacobian_ad, forward_ad, reverse_ad)
end

concrete_jac(::GeneralizedFirstOrderAlgorithm{CJ}) where {CJ} = CJ

@concrete mutable struct GeneralizedFirstOrderAlgorithmCache{iip, GB, timeit} <:
                         AbstractNonlinearSolveCache{iip, timeit}
    # Basic Requirements
    fu
    u
    u_cache
    p
    du  # Aliased to `get_du(descent_cache)`
    J   # Aliased to `jac_cache.J`
    alg
    prob

    # Internal Caches
    jac_cache
    descent_cache
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
    total_time::Float64   # Simple Counter which works even if TimerOutput is disabled

    # State Affect
    make_new_jacobian::Bool

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    kwargs
end

SymbolicIndexingInterface.state_values(cache::GeneralizedFirstOrderAlgorithmCache) = cache.u
function SymbolicIndexingInterface.parameter_values(cache::GeneralizedFirstOrderAlgorithmCache)
    cache.p
end

function __reinit_internal!(
        cache::GeneralizedFirstOrderAlgorithmCache{iip}, args...; p = cache.p, u0 = cache.u,
        alias_u0::Bool = false, maxiters = 1000, maxtime = nothing, kwargs...) where {iip}
    if iip
        recursivecopy!(cache.u, u0)
        cache.prob.f(cache.fu, cache.u, p)
    else
        cache.u = __maybe_unaliased(u0, alias_u0)
        set_fu!(cache, cache.prob.f(cache.u, p))
    end
    cache.p = p

    __reinit_internal!(cache.stats)
    cache.nsteps = 0
    cache.maxiters = maxiters
    cache.maxtime = maxtime
    cache.total_time = 0.0
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    cache.make_new_jacobian = true

    reset!(cache.trace)
    reinit!(cache.termination_cache, get_fu(cache), get_u(cache); kwargs...)
    reset_timer!(cache.timer)
end

@internal_caches GeneralizedFirstOrderAlgorithmCache :jac_cache :descent_cache :linesearch_cache :trustregion_cache

function SciMLBase.__init(
        prob::AbstractNonlinearProblem{uType, iip}, alg::GeneralizedFirstOrderAlgorithm,
        args...; stats = empty_nlstats(), alias_u0 = false, maxiters = 1000,
        abstol = nothing, reltol = nothing, maxtime = nothing,
        termination_condition = nothing, internalnorm = L2_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip}
    timer = get_timer_output()
    @static_timeit timer "cache construction" begin
        (; f, u0, p) = prob
        u = __maybe_unaliased(u0, alias_u0)
        fu = evaluate_f(prob, u)
        @bb u_cache = copy(u)

        linsolve = get_linear_solver(alg.descent)

        abstol, reltol, termination_cache = NonlinearSolveBase.init_termination_cache(
            prob, abstol, reltol, fu, u, termination_condition, Val(:regular))
        linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

        jac_cache = JacobianCache(
            prob, alg, f, fu, u, p; stats, autodiff = alg.jacobian_ad, linsolve,
            jvp_autodiff = alg.forward_ad, vjp_autodiff = alg.reverse_ad)
        J = jac_cache(nothing)
        descent_cache = __internal_init(prob, alg.descent, J, fu, u; stats, abstol,
            reltol, internalnorm, linsolve_kwargs, timer)
        du = get_du(descent_cache)

        if alg.trustregion !== missing && alg.linesearch !== missing
            error("TrustRegion and LineSearch methods are algorithmically incompatible.")
        end

        GB = :None
        linesearch_cache = nothing
        trustregion_cache = nothing

        if alg.trustregion !== missing
            supports_trust_region(alg.descent) || error("Trust Region not supported by \
                                                        $(alg.descent).")
            trustregion_cache = __internal_init(
                prob, alg.trustregion, f, fu, u, p; stats, internalnorm, kwargs...)
            GB = :TrustRegion
        end

        if alg.linesearch !== missing
            supports_line_search(alg.descent) || error("Line Search not supported by \
                                                        $(alg.descent).")
            linesearch_ad = alg.forward_ad === nothing ?
                            (alg.reverse_ad === nothing ? alg.jacobian_ad :
                             alg.reverse_ad) : alg.forward_ad
            if linesearch_ad !== nothing && iip && !DI.check_inplace(linesearch_ad)
                @warn "$(linesearch_ad) doesn't support in-place problems."
                linesearch_ad = nothing
            end
            linesearch_ad = get_concrete_forward_ad(
                linesearch_ad, prob, False; check_forward_mode = false)
            linesearch_cache = init(
                prob, alg.linesearch, fu, u; stats, autodiff = linesearch_ad, kwargs...)
            GB = :LineSearch
        end

        trace = init_nonlinearsolve_trace(
            prob, alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

        return GeneralizedFirstOrderAlgorithmCache{iip, GB, maxtime !== nothing}(
            fu, u, u_cache, p, du, J, alg, prob, jac_cache, descent_cache, linesearch_cache,
            trustregion_cache, stats, 0, maxiters, maxtime, alg.max_shrink_times, timer,
            0.0, true, termination_cache, trace, ReturnCode.Default, false, kwargs)
    end
end

function __step!(cache::GeneralizedFirstOrderAlgorithmCache{iip, GB};
        recompute_jacobian::Union{Nothing, Bool} = nothing, kwargs...) where {iip, GB}
    @static_timeit cache.timer "jacobian" begin
        if (recompute_jacobian === nothing || recompute_jacobian) && cache.make_new_jacobian
            J = cache.jac_cache(cache.u)
            new_jacobian = true
        else
            J = cache.jac_cache(nothing)
            new_jacobian = false
        end
    end

    @static_timeit cache.timer "descent" begin
        if cache.trustregion_cache !== nothing &&
           hasfield(typeof(cache.trustregion_cache), :trust_region)
            descent_result = __internal_solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian,
                trust_region = cache.trustregion_cache.trust_region, cache.kwargs...)
        else
            descent_result = __internal_solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian, cache.kwargs...)
        end
    end

    if !descent_result.linsolve_success
        if new_jacobian
            # Jacobian Information is current and linear solve failed terminate the solve
            cache.retcode = LinearSolveFailureCode
            cache.force_stop = true
            return
        else
            # Jacobian Information is not current and linear solve failed, recompute
            # Jacobian
            if !haskey(cache.kwargs, :verbose) || cache.kwargs[:verbose]
                @warn "Linear Solve Failed but Jacobian Information is not current. \
                       Retrying with updated Jacobian."
            end
            # In the 2nd call the `new_jacobian` is guaranteed to be `true`.
            cache.make_new_jacobian = true
            __step!(cache; recompute_jacobian = true, kwargs...)
            return
        end
    end

    δu, descent_intermediates = descent_result.δu, descent_result.extras

    if descent_result.success
        cache.make_new_jacobian = true
        if GB === :LineSearch
            @static_timeit cache.timer "linesearch" begin
                linesearch_sol = solve!(cache.linesearch_cache, cache.u, δu)
                linesearch_failed = !SciMLBase.successful_retcode(linesearch_sol.retcode)
                α = linesearch_sol.step_size
            end
            if linesearch_failed
                cache.retcode = ReturnCode.InternalLineSearchFailed
                cache.force_stop = true
            end
            @static_timeit cache.timer "step" begin
                @bb axpy!(α, δu, cache.u)
                evaluate_f!(cache, cache.u, cache.p)
            end
        elseif GB === :TrustRegion
            @static_timeit cache.timer "trustregion" begin
                tr_accepted, u_new, fu_new = __internal_solve!(
                    cache.trustregion_cache, J, cache.fu,
                    cache.u, δu, descent_intermediates)
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
        elseif GB === :None
            @static_timeit cache.timer "step" begin
                @bb axpy!(1, δu, cache.u)
                evaluate_f!(cache, cache.u, cache.p)
            end
            α = true
        else
            error("Unknown Globalization Strategy: $(GB). Allowed values are (:LineSearch, \
                  :TrustRegion, :None)")
        end
        check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
    else
        α = false
        cache.make_new_jacobian = false
    end

    update_trace!(cache, α)
    @bb copyto!(cache.u_cache, cache.u)

    callback_into_cache!(cache)

    return nothing
end
