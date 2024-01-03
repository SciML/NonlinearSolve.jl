@concrete struct GeneralizedFirstOrderRootFindingAlgorithm{concrete_jac, name} <:
                 AbstractNonlinearSolveAlgorithm{name}
    linesearch
    trustregion
    descent
    jacobian_ad
    forward_ad
    reverse_ad
end

function GeneralizedFirstOrderRootFindingAlgorithm(; concrete_jac = nothing,
        name::Symbol = :unknown, kwargs...)
    return GeneralizedFirstOrderRootFindingAlgorithm{concrete_jac, name}(; kwargs...)
end

function GeneralizedFirstOrderRootFindingAlgorithm{concrete_jac, name}(; descent,
        linesearch = missing, trustregion = missing, jacobian_ad = nothing,
        forward_ad = nothing, reverse_ad = nothing) where {concrete_jac, name}
    forward_ad = ifelse(forward_ad !== nothing, forward_ad,
        ifelse(jacobian_ad isa ADTypes.AbstractForwardMode, jacobian_ad, nothing))
    reverse_ad = ifelse(reverse_ad !== nothing, reverse_ad,
        ifelse(jacobian_ad isa ADTypes.AbstractReverseMode, jacobian_ad, nothing))

    if linesearch !== missing && !(linesearch isa AbstractNonlinearSolveLineSearchAlgorithm)
        Base.depwarn("Passing in a `LineSearches.jl` algorithm directly is deprecated. \
                      Please use `LineSearchesJL` instead.",
            :GeneralizedFirstOrderRootFindingAlgorithm)
        linesearch = LineSearchesJL(; method = linesearch)
    end

    return GeneralizedFirstOrderRootFindingAlgorithm{concrete_jac, name}(linesearch,
        trustregion, descent, jacobian_ad, forward_ad, reverse_ad)
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

    jac_cache = JacobianCache(prob, alg, f, fu, u, p; autodiff = jacobian_ad, linsolve,
        jvp_autodiff = forward_ad, vjp_autodiff = reverse_ad)
    J = jac_cache(nothing)
    descent_cache = SciMLBase.init(prob, alg.descent, J, fu, u; abstol, reltol,
        internalnorm, linsolve_kwargs)
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

    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

    return GeneralizedFirstOrderRootFindingCache{iip, GB}(fu, u, u_cache, p,
        du, J, alg, prob, jac_cache, descent_cache, linesearch_cache,
        trustregion_cache, UInt(0), UInt(0), UInt(maxiters), true, termination_cache, trace,
        ReturnCode.Default, false)
end

function SciMLBase.step!(cache::GeneralizedFirstOrderRootFindingCache{iip, GB};
        recompute_jacobian::Union{Nothing, Bool} = nothing, kwargs...) where {iip, GB}
    if (recompute_jacobian === nothing || recompute_jacobian) && cache.make_new_jacobian
        J = cache.jac_cache(cache.u)
        new_jacobian = true
    else
        J = cache.jac_cache(nothing)
        new_jacobian = false
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
        cache.make_new_jacobian = true
        if GB === :LineSearch
            _, α = solve!(cache.linesearch_cache, cache.u, δu)
            @bb axpy!(α, δu, cache.u)
            evaluate_f!(cache, cache.u, cache.p)
        elseif GB === :TrustRegion
            tr_accepted, u_new, fu_new = solve!(cache.trustregion_cache, J, cache.fu,
                cache.u, δu, descent_intermediates)
            if tr_accepted
                @bb copyto!(cache.u, u_new)
                @bb copyto!(cache.fu, fu_new)
            else
                cache.make_new_jacobian = false
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
        cache.make_new_jacobian = false
    end

    # TODO: update_trace!(cache, α)

    @bb copyto!(cache.u_cache, cache.u)

    callback_into_cache!(cache)

    return nothing
end

function callback_into_cache!(cache::GeneralizedFirstOrderRootFindingCache)
    callback_into_cache!(cache, cache.jac_cache)
    callback_into_cache!(cache, cache.descent_cache)
    callback_into_cache!(cache, cache.linesearch_cache)
    callback_into_cache!(cache, cache.trustregion_cache)
end
