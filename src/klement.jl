"""
    GeneralKlement(; max_resets = 5, linsolve = nothing,
                     linesearch = LineSearch(), precs = DEFAULT_PRECS)

An implementation of `Klement` with line search, preconditioning and customizable linear
solves.

## Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `5`.
  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `linesearch`: the line search algorithm to use. Defaults to [`LineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` can be
    used here directly, and they will be converted to the correct `LineSearch`.
"""
@concrete struct GeneralKlement <: AbstractNewtonAlgorithm{false, Nothing}
    max_resets::Int
    linsolve
    precs
    linesearch
end

function set_linsolve(alg::GeneralKlement, linsolve)
    return GeneralKlement(alg.max_resets, linsolve, alg.precs, alg.linesearch)
end

function GeneralKlement(; max_resets::Int = 5, linsolve = nothing,
    linesearch = LineSearch(), precs = DEFAULT_PRECS)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return GeneralKlement(max_resets, linsolve, precs, linesearch)
end

@concrete mutable struct GeneralKlementCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    fu
    fu2
    du
    p
    linsolve
    J
    J_cache
    J_cache2
    Jᵀ²du
    Jdu
    resets
    force_stop
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    prob
    stats::NLStats
    lscache
end

get_fu(cache::GeneralKlementCache) = cache.fu

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::GeneralKlement, args...;
    alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    linsolve_kwargs = (;), kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu = evaluate_f(prob, u)
    J = __init_identity_jacobian(u, fu)

    if u isa Number
        linsolve = nothing
        alg = alg_
    else
        # For General Julia Arrays default to LU Factorization
        linsolve_alg = alg_.linsolve === nothing && u isa Array ? LUFactorization() :
                       nothing
        alg = set_linsolve(alg_, linsolve_alg)
        linsolve = __setup_linsolve(J, _vec(fu), _vec(u), p, alg)
    end

    return GeneralKlementCache{iip}(f, alg, u, fu, zero(fu), _mutable_zero(u), p, linsolve,
        J, zero(J), zero(J), _vec(zero(fu)), _vec(zero(fu)), 0, false,
        maxiters, internalnorm, ReturnCode.Default, abstol, prob, NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)))
end

function perform_step!(cache::GeneralKlementCache{true})
    @unpack u, fu, f, p, alg, J, linsolve, du = cache
    T = eltype(J)

    singular, fact_done = _try_factorize_and_check_singular!(linsolve, J)

    if singular
        if cache.resets == alg.max_resets
            cache.force_stop = true
            cache.retcode = ReturnCode.Unstable
            return nothing
        end
        fact_done = false
        fill!(J, zero(T))
        J[diagind(J)] .= T(1)
        cache.resets += 1
    end

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve; A = ifelse(fact_done, nothing, J),
        b = -_vec(fu), linu = _vec(du), p, reltol = cache.abstol)
    cache.linsolve = linres.cache

    # Line Search
    α = perform_linesearch!(cache.lscache, u, du)
    axpy!(α, du, u)
    f(cache.fu2, u, p)

    cache.internalnorm(cache.fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    cache.force_stop && return nothing

    # Update the Jacobian
    cache.J_cache .= cache.J' .^ 2
    cache.Jdu .= _vec(du) .^ 2
    mul!(cache.Jᵀ²du, cache.J_cache, cache.Jdu)
    mul!(cache.Jdu, J, _vec(du))
    cache.fu .= cache.fu2 .- cache.fu
    cache.fu .= _restructure(cache.fu,
        (_vec(cache.fu) .- cache.Jdu) ./ max.(cache.Jᵀ²du, eps(T)))
    mul!(cache.J_cache, _vec(cache.fu), _vec(du)')
    cache.J_cache .*= J
    mul!(cache.J_cache2, cache.J_cache, J)
    J .+= cache.J_cache2

    cache.fu .= cache.fu2

    return nothing
end

function perform_step!(cache::GeneralKlementCache{false})
    @unpack fu, f, p, alg, J, linsolve = cache
    T = eltype(J)

    singular, fact_done = _try_factorize_and_check_singular!(linsolve, J)

    if singular
        if cache.resets == alg.max_resets
            cache.force_stop = true
            cache.retcode = ReturnCode.Unstable
            return nothing
        end
        fact_done = false
        cache.J = __init_identity_jacobian(cache.u, fu)
        cache.resets += 1
    end

    # u = u - J \ fu
    if linsolve === nothing
        cache.du = -fu / cache.J
    else
        linres = dolinsolve(alg.precs, linsolve; A = ifelse(fact_done, nothing, J),
            b = -_vec(fu), linu = _vec(cache.du), p, reltol = cache.abstol)
        cache.linsolve = linres.cache
    end

    # Line Search
    α = perform_linesearch!(cache.lscache, cache.u, cache.du)
    cache.u = @. cache.u + α * cache.du  # `u` might not support mutation
    cache.fu2 = f(cache.u, p)

    cache.internalnorm(cache.fu2) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    cache.force_stop && return nothing

    # Update the Jacobian
    cache.J_cache = cache.J' .^ 2
    cache.Jdu = _vec(cache.du) .^ 2
    cache.Jᵀ²du = cache.J_cache * cache.Jdu
    cache.Jdu = J * _vec(cache.du)
    cache.fu = cache.fu2 .- cache.fu
    cache.fu = _restructure(cache.fu,
        (_vec(cache.fu) .- cache.Jdu) ./ max.(cache.Jᵀ²du, eps(T)))
    cache.J_cache = ((_vec(cache.fu) * _vec(cache.du)') .* J) * J
    cache.J = J .+ cache.J_cache

    cache.fu = cache.fu2

    return nothing
end

function SciMLBase.reinit!(cache::GeneralKlementCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu = cache.f(cache.u, p)
    end
    cache.abstol = abstol
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
