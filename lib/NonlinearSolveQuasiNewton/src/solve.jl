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
    initialization <: AbstractJacobianInitialization

    max_resets::Int
    max_shrink_times::Int

    concrete_jac <: Union{Val{false}, Val{true}}
    name::Symbol
end

NonlinearSolveBase.supports_postcondition(::QuasiNewtonAlgorithm) = true

function QuasiNewtonAlgorithm(;
        linesearch = missing, trustregion = missing, descent, update_rule, reinit_rule,
        initialization, max_resets::Int = typemax(Int), name::Symbol = :unknown,
        max_shrink_times::Int = typemax(Int), concrete_jac = Val(false)
    )
    return QuasiNewtonAlgorithm(
        linesearch, trustregion, descent, update_rule, reinit_rule, initialization,
        max_resets, max_shrink_times, concrete_jac, name
    )
end

@concrete mutable struct QuasiNewtonCache <: AbstractNonlinearSolveCache
    # Basic Requirements
    fu
    u
    u_cache
    p
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

    linsolve_workspace

    # Counters
    stats::NLStats
    nsteps
    nresets
    max_resets::Int
    maxiters::Int
    maxtime
    max_shrink_times::Int
    steps_since_last_reset

    # Timer
    timer
    total_time::Float64

    # Termination & Tracking
    termination_cache
    trace
    retcode
    force_stop
    force_reinit
    kwargs

    # Initialization
    initializealg

    verbose
end

function SciMLBase.get_du(cache::QuasiNewtonCache)
    return SciMLBase.get_du(cache.descent_cache)
end
function NonlinearSolveBase.set_du!(cache::QuasiNewtonCache, δu)
    return NonlinearSolveBase.set_du!(cache.descent_cache, δu)
end

function NonlinearSolveBase.get_abstol(cache::QuasiNewtonCache)
    return NonlinearSolveBase.get_abstol(cache.termination_cache)
end
function NonlinearSolveBase.get_reltol(cache::QuasiNewtonCache)
    return NonlinearSolveBase.get_reltol(cache.termination_cache)
end

function InternalAPI.reinit_self!(
        cache::QuasiNewtonCache, args...; p = cache.p, u0 = cache.u,
        alias_u0::Bool = hasproperty(cache, :alias_u0) ? cache.alias_u0 : false,
        maxiters = hasproperty(cache, :maxiters) ? cache.maxiters : 1000,
        maxtime = hasproperty(cache, :maxtime) ? cache.maxtime : nothing, kwargs...
    )
    Utils.reinit_common!(cache, u0, p, alias_u0)

    NonlinearSolveBase.reset_update_rule_state!(
        cache.update_rule_cache, NonlinearSolveBase.get_fu(cache)
    )

    InternalAPI.reinit!(cache.stats)
    cache.nsteps = NonlinearSolveBase.maybe_traced(0)
    cache.nresets = NonlinearSolveBase.maybe_traced(0)
    cache.steps_since_last_reset = NonlinearSolveBase.maybe_traced(0)
    cache.maxiters = maxiters
    cache.maxtime = maxtime
    cache.total_time = 0.0
    cache.force_stop = NonlinearSolveBase.maybe_traced(false)
    cache.force_reinit = NonlinearSolveBase.maybe_traced(false)
    cache.retcode = NonlinearSolveBase.maybe_traced(ReturnCode.Default)

    NonlinearSolveBase.reset!(cache.trace)
    SciMLBase.reinit!(
        cache.termination_cache, NonlinearSolveBase.get_fu(cache),
        NonlinearSolveBase.get_u(cache); kwargs...
    )
    NonlinearSolveBase.reset_timer!(cache.timer)
    return
end

NonlinearSolveBase.@internal_caches(
    QuasiNewtonCache,
    :initialization_cache, :descent_cache, :linesearch_cache, :trustregion_cache,
    :update_rule_cache, :reinit_rule_cache
)

function SciMLBase.__init(
        prob::AbstractNonlinearProblem, alg::QuasiNewtonAlgorithm, args...;
        stats = NLStats(0, 0, 0, 0, 0), alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = false), maxtime = nothing,
        maxiters = 1000, abstol = nothing, reltol = nothing,
        linsolve_kwargs = (;), termination_condition = nothing,
        internalnorm::F = L2_NORM, initializealg = NonlinearSolveBase.NonlinearSolveDefaultInit(),
        verbose = NonlinearVerbosity(),
        kwargs...
    ) where {F}
    if haskey(kwargs, :alias_u0)
        alias = SciMLBase.NonlinearAliasSpecifier(alias_u0 = kwargs[:alias_u0])
    end
    alias_u0 = alias.alias_u0
    # Enzyme cannot differentiate through FunctionWrappers' llvmcall.
    # QuasiNewton doesn't have alg.autodiff fields; autodiff may come through kwargs
    # or from the linesearch/trustregion algorithm's own autodiff field.
    _ls_ad = if alg.linesearch !== missing && alg.linesearch !== nothing &&
            hasfield(typeof(alg.linesearch), :autodiff)
        alg.linesearch.autodiff
    else
        nothing
    end
    _tr_ad = if alg.trustregion !== missing && alg.trustregion !== nothing &&
            hasfield(typeof(alg.trustregion), :autodiff)
        alg.trustregion.autodiff
    else
        nothing
    end
    _ad_prob = NonlinearSolveBase.maybe_unwrap_prob_for_enzyme(
        prob,
        get(kwargs, :autodiff, nothing),
        get(kwargs, :jvp_autodiff, nothing),
        get(kwargs, :vjp_autodiff, nothing),
        _ls_ad,
        _tr_ad,
    )

    timer = get_timer_output()
    @static_timeit timer "cache construction" begin

        if verbose isa Bool
            if verbose
                verbose = NonlinearVerbosity()
            else
                verbose = NonlinearVerbosity(None())
            end
        elseif verbose isa AbstractVerbosityPreset
            verbose = NonlinearVerbosity(verbose)
        end

        u = Utils.maybe_unaliased(prob.u0, alias_u0)
        fu = Utils.evaluate_f(prob, u)
        @bb u_cache = copy(u)

        inverted_jac = NonlinearSolveBase.store_inverse_jacobian(alg.update_rule)

        linsolve = NonlinearSolveBase.get_linear_solver(alg.descent)

        initialization_cache = InternalAPI.init(
            prob, alg.initialization, alg, prob.f, fu, u, prob.p;
            stats, linsolve, maxiters, internalnorm
        )

        abstol, reltol,
            termination_cache = NonlinearSolveBase.init_termination_cache(
            prob, abstol, reltol, fu, u, termination_condition, Val(:regular)
        )
        linsolve_kwargs = merge((; verbose = verbose.linear_verbosity, abstol, reltol), linsolve_kwargs)

        J = initialization_cache(nothing)

        linsolve_workspace,
            J = Utils.unwrap_val(inverted_jac) ?
            Utils.linsolve_workspace(J) : (nothing, J)

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
                _ad_prob, alg.trustregion, fu, u, _ad_prob.p; stats, internalnorm, kwargs...
            )
            globalization = Val(:TrustRegion)
        end

        if has_linesearch
            NonlinearSolveBase.supports_line_search(alg.descent) ||
                error("Line Search not supported by $(alg.descent).")
            _ls_ad = NonlinearSolveBase.standardize_forwarddiff_tag(
                _ls_ad, _ad_prob
            )
            linesearch_cache = CommonSolve.init(
                _ad_prob, alg.linesearch, fu, u;
                stats, internalnorm, autodiff = _ls_ad, kwargs...
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

        cache = QuasiNewtonCache(
            fu, u, u_cache, prob.p, J, alg, prob, globalization,
            initialization_cache, descent_cache, linesearch_cache,
            trustregion_cache, update_rule_cache, reinit_rule_cache,
            linsolve_workspace, stats, NonlinearSolveBase.maybe_traced(0),
            NonlinearSolveBase.maybe_traced(0),
            alg.max_resets, maxiters, maxtime, alg.max_shrink_times,
            NonlinearSolveBase.maybe_traced(0), timer, 0.0,
            termination_cache, trace, NonlinearSolveBase.maybe_traced(ReturnCode.Default),
            NonlinearSolveBase.maybe_traced(false), NonlinearSolveBase.maybe_traced(false),
            kwargs, initializealg, verbose
        )
        NonlinearSolveBase.run_initialization!(cache)
    end

    return cache
end

function InternalAPI.step!(
        cache::QuasiNewtonCache; recompute_jacobian::Union{Nothing, Bool} = nothing
    )
    new_jacobian = true
    @static_timeit cache.timer "jacobian init/reinit" begin
        # `nsteps` is traced under Reactant; keep host early-returns and isolate
        # the compile path so Trim/JET does not see `@trace if` boxing.
        J = if ReactantCore.within_compile()
            _quasi_newton_jacobian_traced!(cache, recompute_jacobian)
        elseif cache.nsteps == 0  # First Step is special ignore kwargs
            J_init = InternalAPI.solve!(
                cache.initialization_cache, cache.fu, cache.u, Val(false)
            )
            if Utils.unwrap_val(NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache))
                if NonlinearSolveBase.jacobian_initialized_preinverted(
                        cache.initialization_cache.alg
                    )
                    cache.J = J_init
                else
                    cache.J = Utils.linsolve_identity!!(cache.linsolve_workspace, J_init)
                end
            else
                if NonlinearSolveBase.jacobian_initialized_preinverted(
                        cache.initialization_cache.alg
                    )
                    cache.J = Utils.linsolve_identity!!(cache.linsolve_workspace, J_init)
                else
                    cache.J = J_init
                end
            end
            cache.steps_since_last_reset += 1
            cache.J
        else
            countable_reinit = false
            if cache.force_reinit
                reinit, countable_reinit = true, true
                cache.force_reinit = false
            elseif recompute_jacobian === nothing
                # Standard Step
                reinit = InternalAPI.solve!(
                    cache.reinit_rule_cache, cache.J, cache.fu, cache.u, SciMLBase.get_du(cache)
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
                    Utils.linsolve_identity!!(cache.linsolve_workspace, J_init) : J_init
                cache.steps_since_last_reset = 0
                cache.J
            else
                cache.steps_since_last_reset += 1
                cache.J
            end
        end
    end

    if ReactantCore.within_compile()
        # Do not wrap descent/update in `@trace if`: Reactant would try to
        # `set_mlir_data!` on `Diagonal` / `BroydenLowRankJacobian` wrappers.
        _quasi_newton_descent_and_update_traced!(cache, J, new_jacobian)
        return nothing
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

            @SciMLMessage("Linear Solve Failed but Jacobian information is not current. Retrying with updated Jacobian. \
                Retrying with updated Jacobian.", cache.verbose, :linsolve_failed_noncurrent)

            cache.force_reinit = true
            InternalAPI.step!(cache; recompute_jacobian = true)
            return
        end
    end

    δu, descent_intermediates = descent_result.δu, descent_result.extras

    α = if descent_result.success
        α_host = _quasi_newton_apply_descent!(cache, J, δu, descent_intermediates)
        NonlinearSolveBase.check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
        α_host
    else
        cache.force_reinit = true
        false
    end

    update_trace!(
        cache, α;
        uses_jac_inverse = NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache)
    )
    @bb copyto!(cache.u_cache, cache.u)

    if (
            cache.force_stop || cache.force_reinit ||
                (recompute_jacobian !== nothing && !recompute_jacobian)
        )
        NonlinearSolveBase.callback_into_cache!(cache)
        return nothing
    end

    @static_timeit cache.timer "jacobian update" begin
        cache.J = InternalAPI.solve!(
            cache.update_rule_cache, cache.J, cache.fu, cache.u, δu
        )
        NonlinearSolveBase.callback_into_cache!(cache)
    end

    return nothing
end

function _quasi_newton_prepared_jacobian(cache, J_init)
    return if Utils.unwrap_val(NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache))
        if NonlinearSolveBase.jacobian_initialized_preinverted(
                cache.initialization_cache.alg
            )
            return J_init
        else
            return Utils.linsolve_identity!!(cache.linsolve_workspace, J_init)
        end
    else
        if NonlinearSolveBase.jacobian_initialized_preinverted(
                cache.initialization_cache.alg
            )
            return Utils.linsolve_identity!!(cache.linsolve_workspace, J_init)
        else
            return J_init
        end
    end
end

function _quasi_newton_set_initialized_jacobian!(cache, J_init)
    cache.J = _quasi_newton_prepared_jacobian(cache, J_init)
    return cache.J
end

function _quasi_newton_unwrap_diag_jacobian(J)
    return J isa Diagonal ? J.diag : J
end

function _quasi_newton_install_jacobian!(cache, J_set)
    if cache.J isa Diagonal && J_set isa Diagonal && cache.J !== J_set &&
            ArrayInterface.can_setindex(cache.J.diag)
        copyto!(cache.J.diag, J_set.diag)
    elseif cache.J isa BroydenLowRankJacobian
        nothing  # Low-rank reinit mutates cache.J in place via initialization
    elseif cache.J === J_set
        nothing
    elseif ArrayInterface.can_setindex(cache.J)
        copyto!(cache.J, J_set)
    else
        cache.J = J_set
    end
    return cache.J
end

function _quasi_newton_jacobian_traced!(cache, recompute_jacobian)
    is_first = cache.nsteps == 0
    force = cache.force_reinit
    cache.force_reinit = NonlinearSolveBase.select(force, false, force)

    rule_reinit = NonlinearSolveBase.maybe_traced(false)
    ReactantCore.@trace track_numbers = false if !is_first
        rule_reinit = InternalAPI.solve!(
            cache.reinit_rule_cache, cache.J, cache.fu, cache.u, SciMLBase.get_du(cache)
        )
    end
    forced_recompute = recompute_jacobian === true
    countable = (!is_first) & (force | rule_reinit)
    should_reinit = (!is_first) & (force | rule_reinit | forced_recompute)

    nresets_next = cache.nresets + 1
    exceeded = countable & (nresets_next ≥ cache.max_resets)
    cache.nresets = NonlinearSolveBase.select(countable, nresets_next, cache.nresets)
    cache.retcode = NonlinearSolveBase.select(
        exceeded, ReturnCode.ConvergenceFailure, cache.retcode
    )
    cache.force_stop = cache.force_stop | exceeded

    do_reinit = should_reinit & !exceeded
    _quasi_newton_maybe_reinit_traced!(cache, do_reinit)
    return cache.J
end

function _quasi_newton_maybe_reinit_traced!(cache, do_reinit)
    if cache.J isa BroydenLowRankJacobian
        _quasi_newton_reinit_lbroyden_traced!(cache, do_reinit)
    elseif cache.J isa Diagonal
        _quasi_newton_reinit_diagonal_traced!(cache, do_reinit)
    else
        ReactantCore.@trace track_numbers = false if do_reinit
            J_init = InternalAPI.solve!(
                cache.initialization_cache, cache.fu, cache.u, Val(true)
            )
            J_set = _quasi_newton_prepared_jacobian(cache, J_init)
            _quasi_newton_install_jacobian!(cache, J_set)
            cache.steps_since_last_reset = 0
        else
            cache.steps_since_last_reset = cache.steps_since_last_reset + 1
        end
    end
    return nothing
end

function _quasi_newton_reinit_lbroyden_traced!(cache, do_reinit)
    J = cache.J
    α = Utils.initial_jacobian_scaling_alpha(
        cache.initialization_cache.alg.alpha, cache.u, cache.fu,
        cache.initialization_cache.internalnorm
    )
    inv_α = oftype(J.alpha, inv(α))
    zero_idx = zero(J.idx)
    J.idx = ifelse(do_reinit, zero_idx, J.idx)
    J.alpha = ifelse(do_reinit, inv_α, J.alpha)
    zU = zero(eltype(J.U))
    zV = zero(eltype(J.Vᵀ))
    @. J.U = ifelse(do_reinit, zU, J.U)
    @. J.Vᵀ = ifelse(do_reinit, zV, J.Vᵀ)
    cache.steps_since_last_reset = ifelse(
        do_reinit, zero(cache.steps_since_last_reset), cache.steps_since_last_reset + 1
    )
    return nothing
end

function _quasi_newton_reinit_diagonal_traced!(cache, do_reinit)
    # Compute seed outside `@trace if` to avoid MissingTracedValue from Bool branches.
    α = Utils.initial_jacobian_scaling_alpha(
        cache.initialization_cache.alg.alpha, cache.u, cache.fu,
        cache.initialization_cache.internalnorm
    )
    store_inv = Utils.unwrap_val(
        NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache)
    )
    preinverted = NonlinearSolveBase.jacobian_initialized_preinverted(
        cache.initialization_cache.alg
    )
    α_val = (store_inv && !preinverted) ? inv(α) : α
    α_diag = convert(eltype(cache.J.diag), α_val)
    ReactantCore.@trace track_numbers = false if do_reinit
        @. cache.J.diag = α_diag
        cache.steps_since_last_reset = 0
    else
        cache.steps_since_last_reset = cache.steps_since_last_reset + 1
    end
    return nothing
end

function _quasi_newton_descent_and_update_traced!(cache, J, new_jacobian)
    already_stopped = cache.force_stop
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
    linsolve_fail = !descent_result.linsolve_success
    pathology = (!already_stopped) & linsolve_fail & new_jacobian &
        (cache.steps_since_last_reset == 0)
    cache.retcode = NonlinearSolveBase.select(
        pathology, ReturnCode.InternalLinearSolveFailed, cache.retcode
    )
    cache.force_stop = cache.force_stop | pathology
    cache.force_reinit = cache.force_reinit |
        NonlinearSolveBase.select(
        (!already_stopped) & linsolve_fail & !pathology, true, false
    )
    δu, descent_intermediates = descent_result.δu, descent_result.extras
    can_apply = (!cache.force_stop) & descent_result.success & !linsolve_fail
    α = _quasi_newton_after_descent_traced!(
        cache, J, δu, descent_intermediates, can_apply
    )
    cache.force_reinit = cache.force_reinit |
        NonlinearSolveBase.select(
        !can_apply & !cache.force_stop & !linsolve_fail, true, false
    )
    update_trace!(
        cache, α;
        uses_jac_inverse = NonlinearSolveBase.store_inverse_jacobian(cache.update_rule_cache)
    )
    @bb copyto!(cache.u_cache, cache.u)
    NonlinearSolveBase.callback_into_cache!(cache)
    @static_timeit cache.timer "jacobian update" begin
        J_new = InternalAPI.solve!(
            cache.update_rule_cache, cache.J, cache.fu, cache.u, δu
        )
        _quasi_newton_install_jacobian!(cache, J_new)
        NonlinearSolveBase.callback_into_cache!(cache)
    end
    return nothing
end

function _quasi_newton_apply_descent!(cache, J, δu, descent_intermediates)
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
                cache.u = NonlinearSolveBase.apply_postcondition!!(
                    cache.u, cache.u_cache, cache
                )
                Utils.evaluate_f!(cache, cache.u, cache.p)
            end
        end
        return α
    elseif cache.globalization isa Val{:TrustRegion}
        @static_timeit cache.timer "trustregion" begin
            tr_accepted, u_new,
                fu_new = InternalAPI.solve!(
                cache.trustregion_cache, J, cache.fu, cache.u, δu, descent_intermediates
            )
            if tr_accepted
                @bb copyto!(cache.u, u_new)
                if NonlinearSolveBase.get_postcondition(cache) === nothing
                    @bb copyto!(cache.fu, fu_new)
                else
                    cache.u = NonlinearSolveBase.apply_postcondition!!(
                        cache.u, cache.u_cache, cache
                    )
                    Utils.evaluate_f!(cache, cache.u, cache.p)
                end
            end
            if hasfield(typeof(cache.trustregion_cache), :shrink_counter) &&
                    cache.trustregion_cache.shrink_counter > cache.max_shrink_times
                cache.retcode = ReturnCode.ShrinkThresholdExceeded
                cache.force_stop = true
            end
        end
        return true
    elseif cache.globalization isa Val{:None}
        @static_timeit cache.timer "step" begin
            @bb axpy!(1, δu, cache.u)
            cache.u = NonlinearSolveBase.apply_postcondition!!(
                cache.u, cache.u_cache, cache
            )
            Utils.evaluate_f!(cache, cache.u, cache.p)
        end
        return true
    else
        error("Unknown Globalization Strategy: $(cache.globalization). Allowed values \
               are (:LineSearch, :TrustRegion, :None)")
    end
end

function _quasi_newton_after_descent_traced!(cache, J, δu, descent_intermediates, success)
    # Val{:None}: gate unit step with ifelse (no nested isa inside @trace if).
    if cache.globalization isa Val{:None}
        T = eltype(δu)
        α_step = ifelse(success, one(T), zero(T))
        @bb axpy!(α_step, δu, cache.u)
        cache.u = NonlinearSolveBase.apply_postcondition!!(
            cache.u, cache.u_cache, cache
        )
        Utils.evaluate_f!(cache, cache.u, cache.p)
        ReactantCore.@trace track_numbers = false if success
            NonlinearSolveBase.check_and_update!(
                cache, cache.fu, cache.u, cache.u_cache
            )
        end
        return NonlinearSolveBase.select(
            success,
            NonlinearSolveBase.maybe_traced(true),
            NonlinearSolveBase.maybe_traced(false)
        )
    end
    α = NonlinearSolveBase.maybe_traced(false)
    ReactantCore.@trace track_numbers = false if success
        α = _quasi_newton_apply_descent!(cache, J, δu, descent_intermediates)
        NonlinearSolveBase.check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
    else
        α = NonlinearSolveBase.maybe_traced(false)
    end
    return α
end
