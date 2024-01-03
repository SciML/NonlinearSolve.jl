@concrete struct ApproximateJacobianSolveAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    trustregion
    descent
    update_rule
    reinit_rule
    max_resets::UInt
    initialization
end

function ApproximateJacobianSolveAlgorithm(; concrete_jac = nothing,
        name::Symbol = :unknown, kwargs...)
    return ApproximateJacobianSolveAlgorithm{concrete_jac, name}(; kwargs...)
end

function ApproximateJacobianSolveAlgorithm{concrete_jac, name}(; linesearch = missing,
        trustregion = missing, descent, update_rule, reinit_rule, initialization,
        max_resets = typemax(UInt)) where {concrete_jac, name}
    return ApproximateJacobianSolveAlgorithm{concrete_jac, name}(linesearch, trustregion,
        descent, update_rule, reinit_rule, UInt(max_resets), initialization)
end

@inline concrete_jac(::ApproximateJacobianSolveAlgorithm{CJ}) where {CJ} = CJ

@concrete mutable struct ApproximateJacobianSolveCache{INV, GB, iip} <:
                         AbstractNonlinearSolveCache{iip}
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
    nf::UInt
    nsteps::UInt
    nresets::UInt
    max_resets::UInt
    maxiters::UInt
    total_time::Float64
    cache_initialization_time::Float64

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
    force_reinit::Bool
end

# Accessors Interface
get_fu(cache::ApproximateJacobianSolveCache) = cache.fu
get_u(cache::ApproximateJacobianSolveCache) = cache.u
set_fu!(cache::ApproximateJacobianSolveCache, fu) = (cache.fu = fu)
set_u!(cache::ApproximateJacobianSolveCache, u) = (cache.u = u)

# NLStats interface
# @inline get_nf(cache::ApproximateJacobianSolveCache) = cache.nf +
#                                                        get_nf(cache.linesearch_cache)
# @inline get_njacs(cache::ApproximateJacobianSolveCache) = get_njacs(cache.initialization_cache)
@inline get_nsteps(cache::ApproximateJacobianSolveCache) = cache.nsteps
# @inline increment_nsteps!(cache::ApproximateJacobianSolveCache) = (cache.nsteps += 1)
# @inline function get_nsolve(cache::ApproximateJacobianSolveCache)
#     cache.linsolve_cache === nothing && return 0
#     return get_nsolve(cache.linsolve_cache)
# end
# @inline function get_nfactors(cache::ApproximateJacobianSolveCache)
#     cache.linsolve_cache === nothing && return 0
#     return get_nfactors(cache.linsolve_cache)
# end

function SciMLBase.__init(prob::AbstractNonlinearProblem{uType, iip},
        alg::ApproximateJacobianSolveAlgorithm, args...; alias_u0 = false,
        maxiters = 1000, abstol = nothing, reltol = nothing, linsolve_kwargs = (;),
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F}
    time_start = time()
    (; f, u0, p) = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    @bb u_cache = copy(u)

    INV = store_inverse_jacobian(alg.update_rule)
    # TODO: alpha = __initial_alpha(alg_.alpha, u, fu, internalnorm)

    linsolve = __getproperty(alg.descent, Val(:linsolve))
    initialization_cache = init(prob, alg.initialization, alg, f, fu, u, p; linsolve,
        maxiters)

    abstol, reltol, termination_cache = init_termination_cache(abstol, reltol, u, u,
        termination_condition)
    linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

    J = initialization_cache(nothing)
    inv_workspace, J = INV ? __safe_inv_workspace(J) : (nothing, J)
    descent_cache = init(prob, alg.descent, J, fu, u; abstol, reltol, internalnorm,
        linsolve_kwargs, pre_inverted = Val(INV))
    du = get_du(descent_cache)

    reinit_rule_cache = init(alg.reinit_rule, J, fu, u, du)

    if alg.trustregion !== missing && alg.linesearch !== missing
        error("TrustRegion and LineSearch methods are algorithmically incompatible.")
    end

    GB = :None
    linesearch_cache = nothing
    trustregion_cache = nothing

    if alg.trustregion !== missing
        supports_trust_region(alg.descent) || error("Trust Region not supported by \
                                                     $(alg.descent).")
        trustregion_cache = init(prob, alg.trustregion, f, fu, u, p; internalnorm,
            kwargs...)
        GB = :TrustRegion
    end

    if alg.linesearch !== missing
        supports_line_search(alg.descent) || error("Line Search not supported by \
                                                    $(alg.descent).")
        linesearch_cache = init(prob, alg.linesearch, f, fu, u, p; internalnorm, kwargs...)
        GB = :LineSearch
    end

    update_rule_cache = init(prob, alg.update_rule, J, fu, u, du; internalnorm)

    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du;
        uses_jacobian_inverse = Val(INV), kwargs...)

    cache = ApproximateJacobianSolveCache{INV, GB, iip}(fu, u, u_cache, p, du, J, alg, prob,
        initialization_cache, descent_cache, linesearch_cache, trustregion_cache,
        update_rule_cache, reinit_rule_cache, inv_workspace, UInt(0), UInt(0), UInt(0),
        UInt(alg.max_resets), UInt(maxiters), 0.0, 0.0, termination_cache, trace,
        ReturnCode.Default, false, false)

    cache.cache_initialization_time = time() - time_start
    return cache
end

function SciMLBase.step!(cache::ApproximateJacobianSolveCache{INV, GB, iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing) where {INV, GB, iip}
    new_jacobian = true
    if get_nsteps(cache) == 0
        # First Step is special ignore kwargs
        J_init = solve!(cache.initialization_cache, cache.u, Val(false))
        # TODO: trait to check if init was pre inverted
        cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
        J = cache.J
    else
        countable_reinit = false
        if cache.force_reinit
            reinit, countable_reinit = true, true
            cache.force_reinit = false
        elseif recompute_jacobian === nothing
            # Standard Step
            reinit = solve!(cache.reinit_rule_cache, cache.J, cache.fu, cache.u, cache.du)
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
            J_init = solve!(cache.initialization_cache, cache.u, Val(true))
            cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
            J = cache.J
        else
            J = cache.J
        end
    end

    if cache.trustregion_cache !== nothing &&
       hasfield(typeof(cache.trustregion_cache), :trust_region)
        δu, descent_success, descent_intermediates = solve!(cache.descent_cache,
            ifelse(new_jacobian, J, nothing), cache.fu, cache.u;
            trust_region = cache.trustregion_cache.trust_region)
    else
        δu, descent_success, descent_intermediates = solve!(cache.descent_cache,
            ifelse(new_jacobian, J, nothing), cache.fu, cache.u)
    end

    # TODO: Shrink counter termination for trust region methods
    if descent_success
        if GB === :LineSearch
            needs_reset, α = solve!(cache.linesearch_cache, cache.u, δu)
            if needs_reset
                cache.force_reinit = true
            else
                @bb axpy!(α, δu, cache.u)
                evaluate_f!(cache, cache.u, cache.p)
            end
        elseif GB === :TrustRegion
            tr_accepted, u_new, fu_new = solve!(cache.trustregion_cache, J, cache.fu,
                cache.u, δu, descent_intermediates)
            if tr_accepted
                @bb copyto!(cache.u, u_new)
                @bb copyto!(cache.fu, fu_new)
            end
        elseif GB === :None
            @bb axpy!(1, δu, cache.u)
            evaluate_f!(cache, cache.u, cache.p)
        else
            error("Unknown Globalization Strategy: $(GB). Allowed values are (:LineSearch, \
                  :TrustRegion, :None)")
        end
        check_and_update!(cache, cache.fu, cache.u, cache.u_cache)
    else
        cache.force_reinit = true
    end

    # TODO: update_trace!(cache, α)

    @bb copyto!(cache.u_cache, cache.u)

    if (cache.force_stop || cache.force_reinit ||
        (recompute_jacobian !== nothing && !recompute_jacobian))
        callback_into_cache!(cache)
        return nothing
    end

    cache.J = solve!(cache.update_rule_cache, cache.J, cache.fu, cache.u, δu)
    callback_into_cache!(cache)

    return nothing
end

function callback_into_cache!(cache::ApproximateJacobianSolveCache)
    callback_into_cache!(cache, cache.initialization_cache)
    callback_into_cache!(cache, cache.descent_cache)
    callback_into_cache!(cache, cache.linesearch_cache)
    callback_into_cache!(cache, cache.trustregion_cache)
    callback_into_cache!(cache, cache.update_rule_cache)
    callback_into_cache!(cache, cache.reinit_rule_cache)
end
