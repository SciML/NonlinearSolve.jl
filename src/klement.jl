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
    u_cache
    fu
    fu_cache
    du
    p
    linsolve
    J
    J_cache
    J_cache_2
    Jdu
    Jdu_cache
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

    @bb u_cache = similar(u)
    @bb fu_cache = similar(fu)
    @bb J_cache = similar(J)
    @bb J_cache_2 = similar(J)
    @bb Jdu = similar(fu)
    @bb Jdu_cache = similar(fu)

    return GeneralKlementCache{iip}(f, alg, u, u_cache, fu, fu_cache, du, p, linsolve,
        J, J_cache, J_cache_2, Jdu, Jdu_cache, 0, false, maxiters, internalnorm,
        ReturnCode.Default, abstol, reltol, prob, NLStats(1, 0, 0, 0, 0),
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

    A = ifelse(cache.J isa SMatrix || cache.J isa Number || !fact_done, cache.J, nothing)

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, cache.linsolve; A,
        b = _vec(cache.fu), linu = _vec(cache.du), cache.p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    !iip && (cache.du = linres.u)

    # Line Search
    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)

    evaluate_f(cache, cache.u, cache.p)

    update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    @bb copyto!(cache.u_cache, cache.u)

    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    cache.force_stop && return nothing

    # Update the Jacobian
    @bb cache.du .*= -1
    @bb cache.J_cache .= cache.J' .^ 2
    @bb @. cache.Jdu = cache.du^2
    @bb cache.Jdu_cache = cache.J_cache × vec(cache.Jdu)
    @bb cache.Jdu = cache.J × vec(cache.du)
    @bb @. cache.fu_cache = (cache.fu - cache.fu_cache - cache.Jdu) /
                            max(cache.Jdu_cache, eps(real(T)))
    @bb cache.J_cache = vec(cache.fu) × transpose(_vec(cache.du))
    @bb @. cache.J_cache *= cache.J
    @bb cache.J_cache_2 = cache.J_cache × cache.J
    @bb cache.J .+= cache.J_cache_2

    @bb copyto!(cache.fu_cache, cache.fu)

    return nothing
end

function __reinit_internal!(cache::GeneralKlementCache)
    cache.J = __reinit_identity_jacobian!!(cache.J)
    cache.resets = 0
    return nothing
end
