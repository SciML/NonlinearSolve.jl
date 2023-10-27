# Sadly `Broyden` is taken up by SimpleNonlinearSolve.jl
"""
    GeneralBroyden(; max_resets = 3, linesearch = LineSearch(), reset_tolerance = nothing)

An implementation of `Broyden` with reseting and line search.

## Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(eltype(u)))`.
  - `linesearch`: the line search algorithm to use. Defaults to [`LineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` can be
    used here directly, and they will be converted to the correct `LineSearch`. It is
    recommended to use [LiFukushimaLineSearch](@ref) -- a derivative free linesearch
    specifically designed for Broyden's method.
"""
@concrete struct GeneralBroyden <: AbstractNewtonAlgorithm{false, Nothing}
    max_resets::Int
    reset_tolerance
    linesearch
end

function GeneralBroyden(; max_resets = 3, linesearch = LineSearch(),
    reset_tolerance = nothing)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return GeneralBroyden(max_resets, reset_tolerance, linesearch)
end

@concrete mutable struct GeneralBroydenCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_prev
    du
    fu
    fu2
    dfu
    p
    J⁻¹
    J⁻¹₂
    J⁻¹df
    force_stop::Bool
    resets::Int
    max_resets::Int
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    reset_tolerance
    reset_check
    prob
    stats::NLStats
    lscache
    termination_condition
    tc_storage
end

get_fu(cache::GeneralBroydenCache) = cache.fu

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::GeneralBroyden, args...;
    alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
    termination_condition = nothing, internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu = evaluate_f(prob, u)
    J⁻¹ = __init_identity_jacobian(u, fu)
    reset_tolerance = alg.reset_tolerance === nothing ? sqrt(eps(eltype(u))) :
                      alg.reset_tolerance
    reset_check = x -> abs(x) ≤ reset_tolerance

    abstol, reltol, termination_condition = _init_termination_elements(abstol,
        reltol,
        termination_condition,
        eltype(u))

    mode = DiffEqBase.get_termination_mode(termination_condition)

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
              nothing
    return GeneralBroydenCache{iip}(f, alg, u, zero(u), _mutable_zero(u), fu, zero(fu),
        zero(fu), p, J⁻¹, zero(_reshape(fu, 1, :)), _mutable_zero(u), false, 0,
        alg.max_resets, maxiters, internalnorm, ReturnCode.Default, abstol, reltol,
        reset_tolerance,
        reset_check, prob, NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), termination_condition,
        storage)
end

function perform_step!(cache::GeneralBroydenCache{true})
    @unpack f, p, du, fu, fu2, dfu, u, u_prev, J⁻¹, J⁻¹df, J⁻¹₂, tc_storage = cache

    termination_condition = cache.termination_condition(tc_storage)
    T = eltype(u)

    mul!(_vec(du), J⁻¹, -_vec(fu))
    α = perform_linesearch!(cache.lscache, u, du)
    _axpy!(α, du, u)
    f(fu2, u, p)

    termination_condition(fu2, u, u_prev, cache.abstol, cache.reltol) &&
        (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the inverse jacobian
    dfu .= fu2 .- fu

    if all(cache.reset_check, du) || all(cache.reset_check, dfu)
        if cache.resets ≥ cache.max_resets
            cache.retcode = ReturnCode.ConvergenceFailure
            cache.force_stop = true
            return nothing
        end
        fill!(J⁻¹, 0)
        J⁻¹[diagind(J⁻¹)] .= T(1)
        cache.resets += 1
    else
        mul!(_vec(J⁻¹df), J⁻¹, _vec(dfu))
        mul!(J⁻¹₂, _vec(du)', J⁻¹)
        denom = dot(du, J⁻¹df)
        du .= (du .- J⁻¹df) ./ ifelse(iszero(denom), T(1e-5), denom)
        mul!(J⁻¹, _vec(du), J⁻¹₂, 1, 1)
    end
    fu .= fu2
    @. u_prev = u

    return nothing
end

function perform_step!(cache::GeneralBroydenCache{false})
    @unpack f, p, tc_storage = cache

    termination_condition = cache.termination_condition(tc_storage)

    T = eltype(cache.u)

    cache.du = _restructure(cache.du, cache.J⁻¹ * -_vec(cache.fu))
    α = perform_linesearch!(cache.lscache, cache.u, cache.du)
    cache.u = cache.u .+ α * cache.du
    cache.fu2 = f(cache.u, p)

    termination_condition(cache.fu2, cache.u, cache.u_prev, cache.abstol, cache.reltol) &&
        (cache.force_stop = true)
    cache.stats.nf += 1

    cache.force_stop && return nothing

    # Update the inverse jacobian
    cache.dfu = cache.fu2 .- cache.fu
    if all(cache.reset_check, cache.du) || all(cache.reset_check, cache.dfu)
        if cache.resets ≥ cache.max_resets
            cache.retcode = ReturnCode.ConvergenceFailure
            cache.force_stop = true
            return nothing
        end
        cache.J⁻¹ = __init_identity_jacobian(cache.u, cache.fu)
        cache.resets += 1
    else
        cache.J⁻¹df = _restructure(cache.J⁻¹df, cache.J⁻¹ * _vec(cache.dfu))
        cache.J⁻¹₂ = _vec(cache.du)' * cache.J⁻¹
        denom = dot(cache.du, cache.J⁻¹df)
        cache.du = (cache.du .- cache.J⁻¹df) ./ ifelse(iszero(denom), T(1e-5), denom)
        cache.J⁻¹ = cache.J⁻¹ .+ _vec(cache.du) * cache.J⁻¹₂
    end
    cache.fu = cache.fu2
    cache.u_prev = @. cache.u

    return nothing
end

function SciMLBase.reinit!(cache::GeneralBroydenCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, reltol = cache.reltol,
    termination_condition = cache.termination_condition,
    maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu = cache.f(cache.u, p)
    end

    termination_condition = _get_reinit_termination_condition(cache, abstol, reltol,
        termination_condition)

    cache.abstol = abstol
    cache.reltol = reltol
    cache.termination_condition = termination_condition
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.resets = 0
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
