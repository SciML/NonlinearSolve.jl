"""
    ApproximateJacobianSolveAlgorithm{concrete_jac, name}(; linesearch = missing,
        trustregion = missing, descent, update_rule, reinit_rule, initialization,
        max_resets::Int = typemax(Int), max_shrink_times::Int = typemax(Int))
    ApproximateJacobianSolveAlgorithm(; concrete_jac = nothing,
        name::Symbol = :unknown, kwargs...)

Nonlinear Solve Algorithms using an Iterative Approximation of the Jacobian. Most common
examples include [`Broyden`](@ref)'s Method.

### Keyword Arguments

  - `trustregion`: Globalization using a Trust Region Method. This needs to follow the
    [`NonlinearSolve.AbstractTrustRegionMethod`](@ref) interface.
  - `descent`: The descent method to use to compute the step. This needs to follow the
    [`NonlinearSolve.AbstractDescentAlgorithm`](@ref) interface.
  - `max_shrink_times`: The maximum number of times the trust region radius can be shrunk
    before the algorithm terminates.
  - `update_rule`: The update rule to use to update the Jacobian. This needs to follow the
    [`NonlinearSolve.AbstractApproximateJacobianUpdateRule`](@ref) interface.
  - `reinit_rule`: The reinitialization rule to use to reinitialize the Jacobian. This
    needs to follow the [`NonlinearSolve.AbstractResetCondition`](@ref) interface.
  - `initialization`: The initialization method to use to initialize the Jacobian. This
    needs to follow the [`NonlinearSolve.AbstractJacobianInitialization`](@ref) interface.
"""
@concrete struct ApproximateJacobianSolveAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    trustregion
    descent
    update_rule
    reinit_rule
    max_resets::Int
    max_shrink_times::Int
    initialization
end

function __show_algorithm(io::IO, alg::ApproximateJacobianSolveAlgorithm, name, indent)
    modifiers = String[]
    __is_present(alg.linesearch) && push!(modifiers, "linesearch = $(alg.linesearch)")
    __is_present(alg.trustregion) && push!(modifiers, "trustregion = $(alg.trustregion)")
    push!(modifiers, "descent = $(alg.descent)")
    push!(modifiers, "update_rule = $(alg.update_rule)")
    push!(modifiers, "reinit_rule = $(alg.reinit_rule)")
    push!(modifiers, "max_resets = $(alg.max_resets)")
    push!(modifiers, "initialization = $(alg.initialization)")
    store_inverse_jacobian(alg.update_rule) && push!(modifiers, "inverse_jacobian = true")
    spacing = " "^indent * "    "
    spacing_last = " "^indent
    print(io, "$(name)(\n$(spacing)$(join(modifiers, ",\n$(spacing)"))\n$(spacing_last))")
end

function ApproximateJacobianSolveAlgorithm(;
        concrete_jac = nothing, name::Symbol = :unknown, kwargs...)
    return ApproximateJacobianSolveAlgorithm{concrete_jac, name}(; kwargs...)
end

function ApproximateJacobianSolveAlgorithm{concrete_jac, name}(;
        linesearch = missing, trustregion = missing, descent, update_rule,
        reinit_rule, initialization, max_resets::Int = typemax(Int),
        max_shrink_times::Int = typemax(Int)) where {concrete_jac, name}
    if linesearch !== missing && !(linesearch isa AbstractNonlinearSolveLineSearchAlgorithm)
        Base.depwarn("Passing in a `LineSearches.jl` algorithm directly is deprecated. \
                      Please use `LineSearchesJL` instead.",
            :GeneralizedFirstOrderAlgorithm)
        linesearch = LineSearchesJL(; method = linesearch)
    end
    return ApproximateJacobianSolveAlgorithm{concrete_jac, name}(
        linesearch, trustregion, descent, update_rule,
        reinit_rule, max_resets, max_shrink_times, initialization)
end

@inline concrete_jac(::ApproximateJacobianSolveAlgorithm{CJ}) where {CJ} = CJ

@concrete mutable struct ApproximateJacobianSolveCache{INV, GB, iip, timeit} <:
                         AbstractNonlinearSolveCache{iip, timeit}
    # Basic Requirements
    fu
    u
    u_cache
    p
    du  # Aliased to `get_du(descent_cache)`
    J   # Aliased to `initialization_cache.J` if !INV
    alg
    prob

    # Internal Caches
    initialization_cache
    descent_cache
    linesearch_cache
    trustregion_cache
    update_rule_cache
    reinit_rule_cache

    inv_workspace

    # Counters
    nf::Int
    nsteps::Int
    nresets::Int
    max_resets::Int
    maxiters::Int
    maxtime
    max_shrink_times::Int
    steps_since_last_reset::Int

    # Timer
    timer
    total_time::Float64   # Simple Counter which works even if TimerOutput is disabled

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    force_reinit::Bool
end

store_inverse_jacobian(::ApproximateJacobianSolveCache{INV}) where {INV} = INV

function __reinit_internal!(cache::ApproximateJacobianSolveCache{INV, GB, iip},
        args...; p = cache.p, u0 = cache.u, alias_u0::Bool = false,
        maxiters = 1000, maxtime = nothing, kwargs...) where {INV, GB, iip}
    if iip
        recursivecopy!(cache.u, u0)
        cache.prob.f(cache.fu, cache.u, p)
    else
        cache.u = __maybe_unaliased(u0, alias_u0)
        set_fu!(cache, cache.prob.f(cache.u, p))
    end
    cache.p = p

    cache.nf = 1
    cache.nsteps = 0
    cache.nresets = 0
    cache.steps_since_last_reset = 0
    cache.maxiters = maxiters
    cache.maxtime = maxtime
    cache.total_time = 0.0
    cache.force_stop = false
    cache.force_reinit = false
    cache.retcode = ReturnCode.Default

    reset!(cache.trace)
    reinit!(cache.termination_cache, get_fu(cache), get_u(cache); kwargs...)
    reset_timer!(cache.timer)
end

@internal_caches ApproximateJacobianSolveCache :initialization_cache :descent_cache :linesearch_cache :trustregion_cache :update_rule_cache :reinit_rule_cache

function SciMLBase.__init(
        prob::AbstractNonlinearProblem{uType, iip}, alg::ApproximateJacobianSolveAlgorithm,
        args...; alias_u0 = false, maxtime = nothing, maxiters = 1000, abstol = nothing,
        reltol = nothing, linsolve_kwargs = (;), termination_condition = nothing,
        internalnorm::F = DEFAULT_NORM, kwargs...) where {uType, iip, F}
    timer = get_timer_output()
    @static_timeit timer "cache construction" begin
        (; f, u0, p) = prob
        u = __maybe_unaliased(u0, alias_u0)
        fu = evaluate_f(prob, u)
        @bb u_cache = copy(u)

        INV = store_inverse_jacobian(alg.update_rule)

        linsolve = get_linear_solver(alg.descent)
        initialization_cache = __internal_init(
            prob, alg.initialization, alg, f, fu, u, p; linsolve, maxiters, internalnorm)

        abstol, reltol, termination_cache = init_termination_cache(
            abstol, reltol, fu, u, termination_condition)
        linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

        J = initialization_cache(nothing)
        inv_workspace, J = INV ? __safe_inv_workspace(J) : (nothing, J)
        descent_cache = __internal_init(
            prob, alg.descent, J, fu, u; abstol, reltol, internalnorm,
            linsolve_kwargs, pre_inverted = Val(INV), timer)
        du = get_du(descent_cache)

        reinit_rule_cache = __internal_init(alg.reinit_rule, J, fu, u, du)

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
                prob, alg.trustregion, f, fu, u, p; internalnorm, kwargs...)
            GB = :TrustRegion
        end

        if alg.linesearch !== missing
            supports_line_search(alg.descent) || error("Line Search not supported by \
                                                        $(alg.descent).")
            linesearch_cache = __internal_init(
                prob, alg.linesearch, f, fu, u, p; internalnorm, kwargs...)
            GB = :LineSearch
        end

        update_rule_cache = __internal_init(
            prob, alg.update_rule, J, fu, u, du; internalnorm)

        trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du;
            uses_jacobian_inverse = Val(INV), kwargs...)

        return ApproximateJacobianSolveCache{INV, GB, iip, maxtime !== nothing}(
            fu, u, u_cache, p, du, J, alg, prob, initialization_cache,
            descent_cache, linesearch_cache, trustregion_cache,
            update_rule_cache, reinit_rule_cache, inv_workspace, 0, 0, 0,
            alg.max_resets, maxiters, maxtime, alg.max_shrink_times, 0, timer,
            0.0, termination_cache, trace, ReturnCode.Default, false, false)
    end
end

function __step!(cache::ApproximateJacobianSolveCache{INV, GB, iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing) where {INV, GB, iip}
    new_jacobian = true
    @static_timeit cache.timer "jacobian init/reinit" begin
        if get_nsteps(cache) == 0  # First Step is special ignore kwargs
            J_init = __internal_solve!(
                cache.initialization_cache, cache.fu, cache.u, Val(false))
            if INV
                if jacobian_initialized_preinverted(cache.initialization_cache.alg)
                    cache.J = J_init
                else
                    cache.J = __safe_inv!!(cache.inv_workspace, J_init)
                end
            else
                if jacobian_initialized_preinverted(cache.initialization_cache.alg)
                    cache.J = __safe_inv!!(cache.inv_workspace, J_init)
                else
                    cache.J = J_init
                end
            end
            J = cache.J
            cache.steps_since_last_reset += 1
        else
            countable_reinit = false
            if cache.force_reinit
                reinit, countable_reinit = true, true
                cache.force_reinit = false
            elseif recompute_jacobian === nothing
                # Standard Step
                reinit = __internal_solve!(
                    cache.reinit_rule_cache, cache.J, cache.fu, cache.u, cache.du)
                reinit && (countable_reinit = true)
            elseif recompute_jacobian
                reinit = true  # Force ReInitialization: Don't count towards resetting
            else
                new_jacobian = false # Jacobian won't be updated in this step
                reinit = false # Override Checks: Unsafe operation
            end

            if countable_reinit
                cache.nresets += 1
                if cache.nresets ≥ cache.max_resets
                    cache.retcode = ReturnCode.ConvergenceFailure
                    cache.force_stop = true
                    return
                end
            end

            if reinit
                J_init = __internal_solve!(
                    cache.initialization_cache, cache.fu, cache.u, Val(true))
                cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
                J = cache.J
                cache.steps_since_last_reset = 0
            else
                J = cache.J
                cache.steps_since_last_reset += 1
            end
        end
    end

    @static_timeit cache.timer "descent" begin
        if cache.trustregion_cache !== nothing &&
           hasfield(typeof(cache.trustregion_cache), :trust_region)
            δu, descent_success, descent_intermediates = __internal_solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian,
                trust_region = cache.trustregion_cache.trust_region)
        else
            δu, descent_success, descent_intermediates = __internal_solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian)
        end
    end

    if descent_success
        if GB === :LineSearch
            @static_timeit cache.timer "linesearch" begin
                needs_reset, α = __internal_solve!(cache.linesearch_cache, cache.u, δu)
            end
            if needs_reset && cache.steps_since_last_reset > 5 # Reset after a burn-in period
                cache.force_reinit = true
            else
                @static_timeit cache.timer "step" begin
                    @bb axpy!(α, δu, cache.u)
                    evaluate_f!(cache, cache.u, cache.p)
                end
            end
        elseif GB === :TrustRegion
            @static_timeit cache.timer "trustregion" begin
                tr_accepted, u_new, fu_new = __internal_solve!(
                    cache.trustregion_cache, J, cache.fu,
                    cache.u, δu, descent_intermediates)
                if tr_accepted
                    @bb copyto!(cache.u, u_new)
                    @bb copyto!(cache.fu, fu_new)
                end
                if hasfield(typeof(cache.trustregion_cache), :shrink_counter) &&
                   cache.trustregion_cache.shrink_counter > cache.max_shrink_times
                    cache.retcode = ReturnCode.ShrinkThresholdExceeded
                    cache.force_stop = true
                end
            end
            α = true
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
        cache.force_reinit = true
    end

    update_trace!(cache, α)
    @bb copyto!(cache.u_cache, cache.u)

    if (cache.force_stop ||
        cache.force_reinit ||
        (recompute_jacobian !== nothing && !recompute_jacobian))
        callback_into_cache!(cache)
        return nothing
    end

    @static_timeit cache.timer "jacobian update" begin
        cache.J = __internal_solve!(cache.update_rule_cache, cache.J, cache.fu, cache.u, δu)
        callback_into_cache!(cache)
    end

    return nothing
end
