"""
    QuasiNewtonAlgorithm(;
        linesearch = missing, trustregion = missing, descent, update_rule, reinit_rule,
        initialization, max_resets::Int = typemax(Int), name::Symbol = :unknown,
        max_shrink_times::Int = typemax(Int), concrete_jac = Val(false)
    )

Nonlinear Solve Algorithms using an Iterative Approximation of the Jacobian. Most common
examples include [`Broyden`](@ref)'s Method.

### Keyword Arguments

  - `trustregion`: Globalization using a Trust Region Method. This needs to follow the
    [`NonlinearSolveBase.AbstractTrustRegionMethod`](@ref) interface.
  - `descent`: The descent method to use to compute the step. This needs to follow the
    [`NonlinearSolveBase.AbstractDescentDirection`](@ref) interface.
  - `max_shrink_times`: The maximum number of times the trust region radius can be shrunk
    before the algorithm terminates.
  - `update_rule`: The update rule to use to update the Jacobian. This needs to follow the
    [`NonlinearSolveBase.AbstractApproximateJacobianUpdateRule`](@ref) interface.
  - `reinit_rule`: The reinitialization rule to use to reinitialize the Jacobian. This
    needs to follow the [`NonlinearSolveBase.AbstractResetCondition`](@ref) interface.
  - `initialization`: The initialization method to use to initialize the Jacobian. This
    needs to follow the [`NonlinearSolveBase.AbstractJacobianInitialization`](@ref)
    interface.
"""
@concrete struct QuasiNewtonAlgorithm <: AbstractNonlinearSolveAlgorithm
    linesearch
    trustregion
    descent <: AbstractDescentDirection
    update_rule <: AbstractApproximateJacobianUpdateRule
    reinit_rule <: AbstractResetCondition
    max_resets::Int
    max_shrink_times::Int
    initialization
    concrete_jac <: Union{Val{false}, Val{true}}
    name::Symbol
end

function QuasiNewtonAlgorithm(;
        linesearch = missing, trustregion = missing, descent, update_rule, reinit_rule,
        initialization, max_resets::Int = typemax(Int), name::Symbol = :unknown,
        max_shrink_times::Int = typemax(Int), concrete_jac = Val(false)
)
    return QuasiNewtonAlgorithm(
        linesearch, trustregion, descent, update_rule, reinit_rule,
        max_resets, max_shrink_times, initialization, concrete_jac, name
    )
end

@concrete mutable struct QuasiNewtonCache <: AbstractNonlinearSolveCache
    # Basic Requirements
    fu
    u
    u_cache
    p
    du  # Aliased to `get_du(descent_cache)`
    J   # Aliased to `initialization_cache.J` if !inverted_jac
    alg <: QuasiNewtonAlgorithm
    prob <: AbstractNonlinearProblem
    globalization <: Union{Val{:LineSearch}, Val{:TrustRegion}, Val{:None}}

    # Internal Caches
    initialization_cache
    descent_cache
    linesearch_cache
    trustregion_cache
    update_rule_cache
    reinit_rule_cache

    inv_workspace

    # Counters
    stats::NLStats
    nsteps::Int
    nresets::Int
    max_resets::Int
    maxiters::Int
    maxtime
    max_shrink_times::Int
    steps_since_last_reset::Int

    # Timer
    timer
    total_time::Float64

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    force_reinit::Bool
    kwargs
end

# XXX: Implement
# function __reinit_internal!(cache::ApproximateJacobianSolveCache{INV, GB, iip},
#         args...; p = cache.p, u0 = cache.u, alias_u0::Bool = false,
#         maxiters = 1000, maxtime = nothing, kwargs...) where {INV, GB, iip}
#     if iip
#         recursivecopy!(cache.u, u0)
#         cache.prob.f(cache.fu, cache.u, p)
#     else
#         cache.u = __maybe_unaliased(u0, alias_u0)
#         set_fu!(cache, cache.prob.f(cache.u, p))
#     end
#     cache.p = p

#     __reinit_internal!(cache.stats)
#     cache.nsteps = 0
#     cache.nresets = 0
#     cache.steps_since_last_reset = 0
#     cache.maxiters = maxiters
#     cache.maxtime = maxtime
#     cache.total_time = 0.0
#     cache.force_stop = false
#     cache.force_reinit = false
#     cache.retcode = ReturnCode.Default

#     reset!(cache.trace)
#     reinit!(cache.termination_cache, get_fu(cache), get_u(cache); kwargs...)
#     reset_timer!(cache.timer)
# end

# @internal_caches ApproximateJacobianSolveCache :initialization_cache :descent_cache :linesearch_cache :trustregion_cache :update_rule_cache :reinit_rule_cache

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::QuasiNewtonAlgorithm, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias_u0 = false, maxtime = nothing,
        maxiters = 1000, abstol = nothing, reltol = nothing,
        linsolve_kwargs = (;), termination_condition = nothing,
        internalnorm::F = L2_NORM, kwargs...
) where {F}
    timer = get_timer_output()
    @static_timeit timer "cache construction" begin
        u = Utils.maybe_unaliased(prob.u0, alias_u0)
        fu = Utils.evaluate_f(prob, u)
        @bb u_cache = copy(u)

        inverted_jac = NonlinearSolveBase.store_inverse_jacobian(alg.update_rule)

        linsolve = NonlinearSolveBase.get_linear_solver(alg.descent)

        initialization_cache = InternalAPI.init(
            prob, alg.initialization, alg, prob.f, fu, u, prob.p;
            stats, linsolve, maxiters, internalnorm
        )

        abstol, reltol, termination_cache = NonlinearSolveBase.init_termination_cache(
            prob, abstol, reltol, fu, u, termination_condition, Val(:regular)
        )
        linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

        J = initialization_cache(nothing)

        inv_workspace, J = Utils.unwrap_val(inverted_jac) ?
                           Utils.maybe_pinv!!_workspace(J) : (nothing, J)

        descent_cache = InternalAPI.init(
            prob, alg.descent, J, fu, u;
            stats, abstol, reltol, internalnorm,
            linsolve_kwargs, pre_inverted = inverted_jac, timer
        )
        du = SciMLBase.get_du(descent_cache)

        reinit_rule_cache = InternalAPI.init(alg.reinit_rule, J, fu, u, du)

        has_linesearch = alg.linesearch !== missing && alg.linesearch !== nothing
        has_trustregion = alg.trustregion !== missing && alg.trustregion !== nothing

        if has_trustregion && has_linesearch
            error("TrustRegion and LineSearch methods are algorithmically incompatible.")
        end

        globalization = Val(:None)
        linesearch_cache = nothing
        trustregion_cache = nothing

        if has_trustregion
            NonlinearSolveBase.supports_trust_region(alg.descent) ||
                error("Trust Region not supported by $(alg.descent).")
            trustregion_cache = InternalAPI.init(
                prob, alg.trustregion, fu, u, p; stats, internalnorm, kwargs...
            )
            globalization = Val(:TrustRegion)
        end

        if has_linesearch
            NonlinearSolveBase.supports_line_search(alg.descent) ||
                error("Line Search not supported by $(alg.descent).")
            linesearch_cache = CommonSolve.init(
                prob, alg.linesearch, fu, u; stats, internalnorm, kwargs...
            )
            globalization = Val(:LineSearch)
        end

        update_rule_cache = InternalAPI.init(
            prob, alg.update_rule, J, fu, u, du; stats, internalnorm
        )

        trace = NonlinearSolveBase.init_nonlinearsolve_trace(
            prob, alg, u, fu, J, du;
            uses_jacobian_inverse = inverted_jac, kwargs...
        )

        return QuasiNewtonCache(
            fu, u, u_cache, prob.p, du, J, alg, prob, globalization,
            initialization_cache, descent_cache, linesearch_cache,
            trustregion_cache, update_rule_cache, reinit_rule_cache,
            inv_workspace, stats, 0, 0, alg.max_resets, maxiters, maxtime,
            alg.max_shrink_times, 0, timer, 0.0, termination_cache, trace,
            ReturnCode.Default, false, false, kwargs
        )
    end
end

function InternalAPI.step!(
        cache::QuasiNewtonCache; recompute_jacobian::Union{Nothing, Bool} = nothing
)
    new_jacobian = true
    @static_timeit cache.timer "jacobian init/reinit" begin
        if cache.nsteps == 0  # First Step is special ignore kwargs
            J_init = InternalAPI.solve!(
                cache.initialization_cache, cache.fu, cache.u, Val(false)
            )
            if Utils.unwrap_val(NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache))
                if NonlinearSolveBase.jacobian_initialized_preinverted(
                    cache.initialization_cache.alg
                )
                    cache.J = J_init
                else
                    cache.J = Utils.maybe_pinv!!(cache.inv_workspace, J_init)
                end
            else
                if NonlinearSolveBase.jacobian_initialized_preinverted(
                    cache.initialization_cache.alg
                )
                    cache.J = Utils.maybe_pinv!!(cache.inv_workspace, J_init)
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
                reinit = InternalAPI.solve!(
                    cache.reinit_rule_cache, cache.J, cache.fu, cache.u, cache.du
                )
                reinit && (countable_reinit = true)
            elseif recompute_jacobian
                reinit = true  # Force ReInitialization: Don't count towards resetting
            else
                new_jacobian = false # Jacobian won't be updated in this step
                reinit = false       # Override Checks: Unsafe operation
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
                J_init = InternalAPI.solve!(
                    cache.initialization_cache, cache.fu, cache.u, Val(true)
                )
                cache.J = Utils.unwrap_val(NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache)) ?
                          Utils.maybe_pinv!!(cache.inv_workspace, J_init) : J_init
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
            descent_result = InternalAPI.solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian,
                cache.trustregion_cache.trust_region, cache.kwargs...
            )
        else
            descent_result = InternalAPI.solve!(
                cache.descent_cache, J, cache.fu, cache.u; new_jacobian, cache.kwargs...
            )
        end
    end

    if !descent_result.linsolve_success
        if new_jacobian && cache.steps_since_last_reset == 0
            # Extremely pathological case. Jacobian was just reset and linear solve
            # failed. Should ideally never happen in practice unless true jacobian init
            # is used.
            cache.retcode = ReturnCode.InternalLinearSolveFailed
            cache.force_stop = true
            return
        else
            # Force a reinit because the problem is currently un-solvable
            if !haskey(cache.kwargs, :verbose) || cache.kwargs[:verbose]
                @warn "Linear Solve Failed but Jacobian Information is not current. \
                       Retrying with reinitialized Approximate Jacobian."
            end
            cache.force_reinit = true
            InternalAPI.step!(cache; recompute_jacobian = true)
            return
        end
    end

    δu, descent_intermediates = descent_result.δu, descent_result.extras

    if descent_result.success
        if cache.globalization isa Val{:LineSearch}
            @static_timeit cache.timer "linesearch" begin
                linesearch_sol = CommonSolve.solve!(cache.linesearch_cache, cache.u, δu)
                needs_reset = !SciMLBase.successful_retcode(linesearch_sol.retcode)
                α = linesearch_sol.step_size
            end
            if needs_reset && cache.steps_since_last_reset > 5 # Reset after a burn-in period
                cache.force_reinit = true
            else
                @static_timeit cache.timer "step" begin
                    @bb axpy!(α, δu, cache.u)
                    Utils.evaluate_f!(cache, cache.u, cache.p)
                end
            end
        elseif cache.globalization isa Val{:TrustRegion}
            @static_timeit cache.timer "trustregion" begin
                tr_accepted, u_new, fu_new = InternalAPI.solve!(
                    cache.trustregion_cache, J, cache.fu, cache.u, δu, descent_intermediates
                )
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
        # XXX: Implement
        # check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
    else
        α = false
        cache.force_reinit = true
    end

    update_trace!(
        cache, α;
        uses_jac_inverse = NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache)
    )
    @bb copyto!(cache.u_cache, cache.u)

    if (cache.force_stop || cache.force_reinit ||
        (recompute_jacobian !== nothing && !recompute_jacobian))
        # XXX: Implement
        # callback_into_cache!(cache)
        return nothing
    end

    @static_timeit cache.timer "jacobian update" begin
        cache.J = InternalAPI.solve!(
            cache.update_rule_cache, cache.J, cache.fu, cache.u, δu
        )
        # XXX: Implement
        # callback_into_cache!(cache)
    end

    return nothing
end
