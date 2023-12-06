"""
    GeneralKlement(; max_resets = 5, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS)

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
  - `init_jacobian`: the method to use for initializing the jacobian. Defaults to using the
    identity matrix (`Val(:identitiy)`). Alternatively, can be set to `Val(:true_jacobian)`
    to use the true jacobian as initialization. (Our tests suggest it is a good idea to
    to initialize with an identity matrix)
  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are types from ADTypes.jl. (Used if `init_jacobian = Val(:true_jacobian)`)
"""
@concrete struct GeneralKlement{IJ, CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    max_resets::Int
    linsolve
    precs
    linesearch
end

function __alg_print_modifiers(::GeneralKlement{IJ}) where {IJ}
    modifiers = String[]
    IJ !== :identity && push!(modifiers, "init_jacobian = :$(IJ)")
    return modifiers
end

function set_linsolve(alg::GeneralKlement{IJ, CS}, linsolve) where {IJ, CS}
    return GeneralKlement{IJ, CS}(alg.ad, alg.max_resets, linsolve, alg.precs,
        alg.linesearch)
end

function set_ad(alg::GeneralKlement{IJ, CS}, ad) where {IJ, CS}
    return GeneralKlement{IJ, CS}(ad, alg.max_resets, alg.linsolve, alg.precs,
        alg.linesearch)
end

function GeneralKlement(; max_resets::Int = 5, linsolve = nothing,
        linesearch = nothing, precs = DEFAULT_PRECS, init_jacobian::Val = Val(:identity),
        autodiff = nothing)
    IJ = _unwrap_val(init_jacobian)
    @assert IJ ∈ (:identity, :true_jacobian)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    CJ = IJ === :true_jacobian
    return GeneralKlement{IJ, CJ}(autodiff, max_resets, linsolve, precs, linesearch)
end

@concrete mutable struct GeneralKlementCache{iip, IJ} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    fu
    fu_cache
    fu_cache_2
    du
    p
    uf
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
    jac_cache
    stats::NLStats
    ls_cache
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::GeneralKlement{IJ},
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip, F, IJ}
    @unpack f, u0, p = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)

    if IJ === :true_jacobian
        alg = get_concrete_algorithm(alg_, prob)
        uf, _, J, fu_cache, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
            lininit = Val(false))
    elseif IJ === :identity
        alg = alg_
        @bb du = similar(u)
        uf, fu_cache, jac_cache = nothing, nothing, nothing
        J = one.(u) # Identity Init Jacobian for Klement maintains a Diagonal Structure
    else
        error("Invalid `init_jacobian` value")
    end

    if IJ === :true_jacobian
        linsolve = linsolve_caches(J, _vec(fu), _vec(du), p, alg_; linsolve_kwargs)
    else
        linsolve = nothing
    end

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, J, du; kwargs...)

    @bb u_cache = copy(u)
    @bb fu_cache_2 = copy(fu)
    @bb Jdu = similar(fu)
    if IJ === :true_jacobian
        @bb J_cache = similar(J)
        @bb J_cache_2 = similar(J)
        @bb Jdu_cache = similar(fu)
    else
        J_cache, J_cache_2, Jdu_cache = nothing, nothing, nothing
    end

    return GeneralKlementCache{iip, IJ}(f, alg, u, u_cache, fu, fu_cache, fu_cache_2, du, p,
        uf, linsolve, J, J_cache, J_cache_2, Jdu, Jdu_cache, 0, false, maxiters,
        internalnorm,
        ReturnCode.Default, abstol, reltol, prob, jac_cache, NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), tc_cache, trace)
end

function perform_step!(cache::GeneralKlementCache{iip, IJ}) where {iip, IJ}
    @unpack linsolve, alg = cache
    T = eltype(cache.J)

    if IJ === :true_jacobian
        cache.stats.nsteps == 0 && (cache.J = jacobian!!(cache.J, cache))
        ill_conditioned = __is_ill_conditioned(cache.J)
    elseif IJ === :identity
        ill_conditioned = __is_ill_conditioned(cache.J)
    end

    if ill_conditioned
        if cache.resets == alg.max_resets
            cache.force_stop = true
            cache.retcode = ReturnCode.ConvergenceFailure
            return nothing
        end
        if IJ === :true_jacobian && cache.stats.nsteps != 0
            cache.J = jacobian!!(cache.J, cache)
        else
            cache.J = __reinit_identity_jacobian!!(cache.J)
        end
        cache.resets += 1
    end

    if IJ === :identity
        @bb @. cache.du = cache.fu / cache.J
    else
        # u = u - J \ fu
        linres = dolinsolve(cache, alg.precs, cache.linsolve; A = cache.J,
            b = _vec(cache.fu), linu = _vec(cache.du), cache.p, reltol = cache.abstol)
        cache.linsolve = linres.cache
        cache.du = _restructure(cache.du, linres.u)
    end

    # Line Search
    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)

    evaluate_f(cache, cache.u, cache.p)

    update_trace!(cache, α)
    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    @bb copyto!(cache.u_cache, cache.u)

    cache.force_stop && return nothing

    # Update the Jacobian
    @bb cache.du .*= -1
    if IJ === :identity
        @bb @. cache.Jdu = (cache.J^2) * (cache.du^2)
        @bb @. cache.J += ((cache.fu - cache.fu_cache_2 - cache.J * cache.du) /
                           ifelse(iszero(cache.Jdu), T(1e-5), cache.Jdu)) * cache.du *
                          (cache.J^2)
    else
        @bb cache.J_cache .= cache.J' .^ 2
        @bb @. cache.Jdu = cache.du^2
        @bb cache.Jdu_cache = cache.J_cache × vec(cache.Jdu)
        @bb cache.Jdu = cache.J × vec(cache.du)
        @bb @. cache.fu_cache_2 = (cache.fu - cache.fu_cache_2 - cache.Jdu) /
                                  ifelse(iszero(cache.Jdu_cache), T(1e-5), cache.Jdu_cache)
        @bb cache.J_cache = vec(cache.fu_cache_2) × transpose(_vec(cache.du))
        @bb @. cache.J_cache *= cache.J
        @bb cache.J_cache_2 = cache.J_cache × cache.J
        @bb cache.J .+= cache.J_cache_2
    end

    @bb copyto!(cache.fu_cache_2, cache.fu)

    return nothing
end

function __reinit_internal!(cache::GeneralKlementCache; kwargs...)
    cache.J = __reinit_identity_jacobian!!(cache.J)
    cache.resets = 0
    return nothing
end
