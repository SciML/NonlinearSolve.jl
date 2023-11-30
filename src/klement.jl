"""
    GeneralKlement(; max_resets = 5, linsolve = nothing,
                     linesearch = nothing, precs = DEFAULT_PRECS)

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
        linesearch = nothing, precs = DEFAULT_PRECS)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return GeneralKlement(max_resets, linsolve, precs, linesearch)
end

@concrete mutable struct GeneralKlementCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_prev
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
    reltol
    prob
    stats::NLStats
    ls_cache
    tc_cache
    trace
end

get_fu(cache::GeneralKlementCache) = cache.fu
set_fu!(cache::GeneralKlementCache, fu) = (cache.fu = fu)

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::GeneralKlement, args...;
        alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip, F}
    @unpack f, u0, p = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    J = __init_identity_jacobian(u, fu)
    @bb du = similar(u)

    if u isa Number
        linsolve = FakeLinearSolveJLCache(J, fu)
        alg = alg_
    else
        # For General Julia Arrays default to LU Factorization
        linsolve_alg = (alg_.linsolve === nothing && (u isa Array || u isa StaticArray)) ?
                       LUFactorization() : nothing
        alg = set_linsolve(alg_, linsolve_alg)
        linsolve = linsolve_caches(J, _vec(fu), _vec(du), p, alg; linsolve_kwargs)
    end

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, J, du; kwargs...)

    @bb u_prev = copy(u)
    @bb fu2 = similar(fu)
    @bb J_cache = similar(J)
    @bb J_cache2 = similar(J)
    @bb Jᵀ²du = similar(fu)
    @bb Jdu = similar(fu)

    return GeneralKlementCache{iip}(f, alg, u, u_prev, fu, fu2, du, p, linsolve, J, J_cache,
        J_cache2, Jᵀ²du, Jdu, 0, false, maxiters, internalnorm, ReturnCode.Default, abstol,
        reltol, prob, NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), tc_cache, trace)
end

function perform_step!(cache::GeneralKlementCache{iip}) where {iip}
    @unpack linsolve, alg = cache
    T = eltype(cache.J)
    singular, fact_done = __try_factorize_and_check_singular!(linsolve, cache.J)

    if singular
        if cache.resets == alg.max_resets
            cache.force_stop = true
            cache.retcode = ReturnCode.ConvergenceFailure
            return nothing
        end
        fact_done = false
        cache.J = __reinit_identity_jacobian!!(cache.J)
        cache.resets += 1
    end

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, cache.linsolve; A = cache.J, b = _vec(cache.fu),
        linu = _vec(cache.du), cache.p, reltol = cache.abstol)
    cache.linsolve = linres.cache

    !iip && (cache.du = linres.u)

    # Line Search
    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)

    if iip
        cache.f(cache.fu2, cache.u, cache.p)
    else
        cache.fu2 = cache.f(cache.u, cache.p)
    end

    update_trace!(cache.trace, cache.stats.nsteps + 1, get_u(cache), cache.fu2, cache.J,
        cache.du, α)

    check_and_update!(cache, cache.fu2, cache.u, cache.u_prev)
    @bb copyto!(cache.u_prev, cache.u)

    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    cache.force_stop && return nothing

    # Update the Jacobian
    @bb cache.du .*= -1
    @bb cache.J_cache .= cache.J' .^ 2
    @bb @. cache.Jdu = cache.du ^ 2
    @bb cache.Jᵀ²du = cache.J_cache × vec(cache.Jdu)
    @bb cache.Jdu = cache.J × vec(cache.du)
    @bb @. cache.fu = cache.fu2 - cache.fu

    @bb @. cache.fu = (cache.fu - cache.Jdu) / max(cache.Jᵀ²du, eps(real(T)))

    @bb cache.J_cache = vec(cache.fu) × transpose(_vec(cache.du))
    @bb @. cache.J_cache *= cache.J
    @bb cache.J_cache2 = cache.J_cache × cache.J
    @bb cache.J .+= cache.J_cache2

    @bb copyto!(cache.fu, cache.fu2)

    return nothing
end

function __reinit_internal!(cache::GeneralKlementCache)
    cache.J = __reinit_identity_jacobian!!(cache.J)
    cache.resets = 0
    return nothing
end
