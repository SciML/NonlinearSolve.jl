# TODO: Trust Region
@concrete struct GeneralizedFirstOrderRootFindingAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    descent
    jacobian_ad
    forward_ad
    reverse_ad
end

concrete_jac(::GeneralizedFirstOrderRootFindingAlgorithm{CJ}) where {CJ} = CJ

@concrete mutable struct GeneralizedFirstOrderRootFindingCache{iip, GB} <:
                         AbstractNonlinearSolveCache{iip}
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
    nf::UInt
    nsteps::UInt
    maxiters::UInt

    # State Affect
    make_new_jacobian::Bool

    # Termination & Tracking
    termination_cache
    trace
    retcode::ReturnCode.T
    force_stop::Bool
end

get_u(cache::GeneralizedFirstOrderRootFindingCache) = cache.u
set_u!(cache::GeneralizedFirstOrderRootFindingCache, u) = (cache.u = u)
get_fu(cache::GeneralizedFirstOrderRootFindingCache) = cache.fu
set_fu!(cache::GeneralizedFirstOrderRootFindingCache, fu) = (cache.fu = fu)

get_nsteps(cache::GeneralizedFirstOrderRootFindingCache) = cache.nsteps

function SciMLBase.__init(prob::AbstractNonlinearProblem{uType, iip},
        alg::GeneralizedFirstOrderRootFindingAlgorithm, args...; alias_u0 = false,
        maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm = DEFAULT_NORM, linsolve_kwargs = (;),
        kwargs...) where {uType, iip}
    # General Setup
    (; f, u0, p) = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    @bb u_cache = copy(u)

    # Concretize the AD types
    jacobian_ad = get_concrete_forward_ad(alg.jacobian_ad, prob, args...;
        check_reverse_mode = false, kwargs...)
    forward_ad = get_concrete_forward_ad(alg.forward_ad, prob, args...;
        check_reverse_mode = true, kwargs...)
    reverse_ad = get_concrete_reverse_ad(alg.reverse_ad, prob, args...;
        check_forward_mode = true, kwargs...)

    linsolve = __getproperty(alg.descent, Val(:linsolve))

    abstol, reltol, termination_cache = init_termination_cache(abstol, reltol, u, u,
        termination_condition)
    linsolve_kwargs = merge((; abstol, reltol), linsolve_kwargs)

    jac_cache = JacobianCache(prob, alg, f, fu, u, p, jacobian_ad, linsolve)
    J = jac_cache.J
    descent_cache = SciMLBase.init(prob, alg.descent, J, fu, u; abstol, reltol,
        internalnorm, linsolve_kwargs)
    du = get_du(descent_cache)

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
        linesearch_cache = SciMLBase.init(prob, alg.linesearch, f, fu, u, p)
        trustregion_cache = nothing
        GB = :LineSearch
    end

    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

    return GeneralizedFirstOrderRootFindingCache{iip, GB}(fu, u, u_cache, p,
        du, J, alg, prob, jac_cache, descent_cache, linesearch_cache,
        trustregion_cache, UInt(0), UInt(0), UInt(maxiters), true, termination_cache, trace,
        ReturnCode.Default, false)
end

function SciMLBase.step!(cache::GeneralizedFirstOrderRootFindingCache{iip, GB};
        recompute_jacobian::Union{Nothing, Bool} = nothing, kwargs...) where {iip, GB}
    # TODO: Use `make_new_jacobian`
    if recompute_jacobian === nothing || recompute_jacobian # Standard Step
        J = cache.jac_cache(cache.u)
        new_jacobian = true
    else # Don't recompute Jacobian
        J = cache.jac_cache(nothing)
        new_jacobian = false
    end

    if GB === :LineSearch
        δu = solve!(cache.descent_cache, ifelse(new_jacobian, J, nothing), cache.fu)
        α = solve!(cache.linesearch_cache, cache.u, δu)
        @bb axpy!(α, δu, cache.u)
    elseif GB === :TrustRegion
        error("Trust Region not implemented yet!")
        α = true
    else
        error("Unknown Globalization Strategy: $(GB). Possible values are (:LineSearch, \
               :TrustRegion)")
    end

    evaluate_f!(cache, cache.u, cache.p)

    # TODO: update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    @bb copyto!(cache.u_cache, cache.u)
    return nothing
end
