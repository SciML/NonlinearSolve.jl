"""
    GeneralKlement(; max_resets = 100, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS, alpha = true, init_jacobian::Val = Val(:identity),
        autodiff = nothing)

An implementation of `Klement` with line search, preconditioning and customizable linear
solves. It is recommended to use `Broyden` for most problems over this.

## Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `100`.

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
  - `alpha`: If `init_jacobian` is set to `Val(:identity)`, then the initial Jacobian
    inverse is set to be `αI`. Defaults to `1`. Can be set to `nothing` which implies
    `α = max(norm(u), 1) / (2 * norm(fu))`.
  - `init_jacobian`: the method to use for initializing the jacobian. Defaults to
    `Val(:identity)`. Choices include:

      + `Val(:identity)`: Identity Matrix.
      + `Val(:true_jacobian)`: True Jacobian. Our tests suggest that this is not very
        stable. Instead using `Broyden` with `Val(:true_jacobian)` gives faster and more
        reliable convergence.
      + `Val(:true_jacobian_diagonal)`: Diagonal of True Jacobian. This is a good choice for
        differentiable problems.
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
    alpha
end

function __alg_print_modifiers(alg::GeneralKlement{IJ}) where {IJ}
    modifiers = String[]
    IJ !== :identity && push!(modifiers, "init_jacobian = Val(:$(IJ))")
    alg.alpha !== nothing && push!(modifiers, "alpha = $(alg.alpha)")
    return modifiers
end

function set_ad(alg::GeneralKlement{IJ, CJ}, ad) where {IJ, CJ}
    return GeneralKlement{IJ, CJ}(ad, alg.max_resets, alg.linsolve, alg.precs,
        alg.linesearch, alg.alpha)
end

function GeneralKlement(; max_resets::Int = 100, linsolve = nothing, alpha = true,
        linesearch = nothing, precs = DEFAULT_PRECS, init_jacobian::Val = Val(:identity),
        autodiff = nothing)
    IJ = _unwrap_val(init_jacobian)
    @assert IJ ∈ (:identity, :true_jacobian, :true_jacobian_diagonal)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    CJ = IJ !== :identity
    return GeneralKlement{IJ, CJ}(autodiff, max_resets, linsolve, precs, linesearch,
        alpha)
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
    alpha
    alpha_initial
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

    alpha = __initial_alpha(alg_.alpha, u, fu, internalnorm)

    if IJ === :true_jacobian
        alg = get_concrete_algorithm(alg_, prob)
        uf, _, J, fu_cache, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
            lininit = Val(false))
    elseif IJ === :true_jacobian_diagonal
        alg = get_concrete_algorithm(alg_, prob)
        uf, _, J_cache, fu_cache, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
            lininit = Val(false))
        J = __diag(J_cache)
    elseif IJ === :identity
        alg = alg_
        @bb du = similar(u)
        uf, fu_cache, jac_cache = nothing, nothing, nothing
        J = one.(u) # Identity Init Jacobian for Klement maintains a Diagonal Structure
        @bb J .*= alpha
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
        IJ === :identity && (J_cache = nothing)
        J_cache_2, Jdu_cache = nothing, nothing
    end

    return GeneralKlementCache{iip, IJ}(f, alg, u, u_cache, fu, fu_cache, fu_cache_2, du, p,
        uf, linsolve, J, J_cache, J_cache_2, Jdu, Jdu_cache, alpha, alg.alpha, 0, false,
        maxiters, internalnorm, ReturnCode.Default, abstol, reltol, prob, jac_cache,
        NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), tc_cache, trace)
end

function perform_step!(cache::GeneralKlementCache{iip, IJ}) where {iip, IJ}
    @unpack linsolve, alg = cache
    T = eltype(cache.J)

    if IJ === :true_jacobian
        cache.stats.nsteps == 0 && (cache.J = jacobian!!(cache.J, cache))
        ill_conditioned = __is_ill_conditioned(cache.J)
    elseif IJ === :true_jacobian_diagonal
        if cache.stats.nsteps == 0
            cache.J_cache = jacobian!!(cache.J_cache, cache)
            cache.J = __get_diagonal!!(cache.J, cache.J_cache)
        end
        ill_conditioned = __is_ill_conditioned(_vec(cache.J))
    elseif IJ === :identity
        ill_conditioned = __is_ill_conditioned(_vec(cache.J))
    end

    if ill_conditioned
        if cache.resets == alg.max_resets
            cache.force_stop = true
            cache.retcode = ReturnCode.ConvergenceFailure
            return nothing
        end
        if IJ === :true_jacobian && cache.stats.nsteps != 0
            cache.J = jacobian!!(cache.J, cache)
        elseif IJ === :true_jacobian_diagonal && cache.stats.nsteps != 0
            cache.J_cache = jacobian!!(cache.J_cache, cache)
            cache.J = __get_diagonal!!(cache.J, cache.J_cache)
        elseif IJ === :identity
            cache.alpha = __initial_alpha(cache.alpha, cache.alpha_initial, cache.u,
                cache.fu, cache.internalnorm)
            cache.J = __reinit_identity_jacobian!!(cache.J, cache.alpha)
        end
        cache.resets += 1
    end

    if IJ === :true_jacobian_diagonal || IJ === :identity
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
    if IJ === :true_jacobian_diagonal || IJ === :identity
        @bb @. cache.Jdu = (cache.J^2) * (cache.du^2)
        @bb @. cache.J += ((cache.fu - cache.fu_cache_2 - cache.J * cache.du) /
                           ifelse(iszero(cache.Jdu), T(1e-5), cache.Jdu)) * cache.du *
                          (cache.J^2)
    elseif IJ === :true_jacobian
        # Klement Updates to the Full Jacobian don't work for most problems, we should
        # probably be using the Broyden Update Rule here
        @bb @. cache.J_cache = cache.J'^2
        @bb @. cache.Jdu = cache.du^2
        @bb cache.Jdu_cache = cache.J_cache × vec(cache.Jdu)
        @bb cache.Jdu = cache.J × vec(cache.du)
        @bb @. cache.fu_cache_2 = (cache.fu - cache.fu_cache_2 - cache.Jdu) /
                                  ifelse(iszero(cache.Jdu_cache), T(1e-5), cache.Jdu_cache)
        @bb cache.J_cache = vec(cache.fu_cache_2) × transpose(_vec(cache.du))
        @bb @. cache.J_cache *= cache.J
        @bb cache.J_cache_2 = cache.J_cache × cache.J
        @bb cache.J .+= cache.J_cache_2
    else
        error("Invalid `init_jacobian` value")
    end

    @bb copyto!(cache.fu_cache_2, cache.fu)

    return nothing
end

function __reinit_internal!(cache::GeneralKlementCache; kwargs...)
    cache.alpha = __initial_alpha(cache.alpha, cache.alpha_initial, cache.u, cache.fu,
        cache.internalnorm)
    cache.J = __reinit_identity_jacobian!!(cache.J, cache.alpha)
    cache.resets = 0
    return nothing
end
