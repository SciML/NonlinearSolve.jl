# TODO: Trust Region
# TODO: alpha_scaling
@concrete struct ApproximateJacobianSolveAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    descent
    update_rule
    reinit_rule
    max_resets::UInt
    initialization
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
    initialization_cache = init(prob, alg.initialization, alg, f, fu, u, p; linsolve)

    abstol, reltol, termination_cache = init_termination_cache(abstol, reltol, u, u,
        termination_condition)
    linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

    J = initialization_cache(nothing)
    inv_workspace, J = INV ? __safe_inv_workspace(J) : (nothing, J)
    descent_cache = init(prob, alg.descent, J, fu, u; abstol, reltol, internalnorm,
        linsolve_kwargs, pre_inverted = Val(INV))
    du = get_du(descent_cache)

    reinit_rule_cache = init(alg.reinit_rule, J, fu, u, du)

    # if alg.trust_region !== missing && alg.linesearch !== missing
    #     error("TrustRegion and LineSearch methods are algorithmically incompatible.")
    # end

    # if alg.trust_region !== missing
    #     supports_trust_region(alg.descent) || error("Trust Region not supported by \
    #                                                  $(alg.descent).")
    #     trustregion_cache = nothing
    #     linesearch_cache = nothing
    #     GB = :TrustRegion
    #     error("Trust Region not implemented yet!")
    # end

    if alg.linesearch !== missing
        supports_line_search(alg.descent) || error("Line Search not supported by \
                                                    $(alg.descent).")
        linesearch_cache = init(prob, alg.linesearch, f, fu, u, p)
        trustregion_cache = nothing
        GB = :LineSearch
    end

    update_rule_cache = init(prob, alg.update_rule, J, fu, u, du; internalnorm)

    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du;
        uses_jacobian_inverse = Val(INV), kwargs...)

    cache = ApproximateJacobianSolveCache{INV, GB, iip}(fu, u, u_cache, p, du, J, alg, prob,
        initialization_cache, descent_cache, linesearch_cache, trustregion_cache,
        update_rule_cache, reinit_rule_cache, inv_workspace, UInt(0), UInt(0), UInt(0),
        UInt(alg.max_resets), UInt(maxiters), 0.0, 0.0, termination_cache, trace,
        ReturnCode.Default, false)

    cache.cache_initialization_time = time() - time_start
    return cache
end

function SciMLBase.step!(cache::ApproximateJacobianSolveCache{INV, GB, iip};
        recompute_jacobian::Union{Nothing, Bool} = nothing) where {INV, GB, iip}
    new_jacobian = true
    if get_nsteps(cache) == 0
        # First Step is special ignore kwargs
        J_init = solve!(cache.initialization_cache, cache.u, Val(false))
        cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
        J = cache.J
    else
        if recompute_jacobian === nothing
            # Standard Step
            reinit = solve!(cache.reinit_rule_cache, cache.J, cache.fu, cache.u, cache.du)
            if reinit
                cache.nresets += 1
                if cache.nresets ≥ cache.max_resets
                    cache.retcode = ReturnCode.ConvergenceFailure
                    cache.force_stop = true
                    return
                end
            end
        elseif recompute_jacobian
            reinit = true  # Force ReInitialization: Don't count towards resetting
        else
            new_jacobian = false # Jacobian won't be updated in this step
            reinit = false # Override Checks: Unsafe operation
        end

        if reinit
            J_init = solve!(cache.initialization_cache, cache.u, Val(true))
            cache.J = INV ? __safe_inv!!(cache.inv_workspace, J_init) : J_init
            J = cache.J
        else
            J = cache.J
        end
    end

    if GB === :LineSearch
        δu, descent_success, descent_intermediates = solve!(cache.descent_cache,
            ifelse(new_jacobian, J, nothing), cache.fu, cache.u)
        needs_reset, α = solve!(cache.linesearch_cache, cache.u, δu)  # TODO: use `needs_reset`
        @bb axpy!(α, δu, cache.u)
    elseif GB === :TrustRegion
        error("Trust Region not implemented yet!")
    else
        error("Unknown Globalization Strategy: $(GB). Allowed values are (:LineSearch, \
               :TrustRegion)")
    end

    evaluate_f!(cache, cache.u, cache.p)

    # TODO: update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    @bb copyto!(cache.u_cache, cache.u)

    (cache.force_stop || (recompute_jacobian !== nothing && !recompute_jacobian)) &&
        return nothing

    cache.J = solve!(cache.update_rule_cache, cache.J, cache.fu, cache.u, δu)

    return nothing
end
