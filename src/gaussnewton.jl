"""
    GaussNewton(; concrete_jac = nothing, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS, adkwargs...)

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear least squares problems.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are types from ADTypes.jl.
  - `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
    then the Jacobian will not be constructed and instead direct Jacobian-vector products
    `J*v` are computed using forward-mode automatic differentiation or finite differencing
    tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
    for example for a preconditioner, `concrete_jac = true` can be passed in order to force
    the construction of the Jacobian.
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
  - `vjp_autodiff`: Automatic Differentiation Backend used for vector-jacobian products.
    This is applicable if the linear solver doesn't require a concrete jacobian, for eg.,
    Krylov Methods. Defaults to `nothing`, which means if the problem is out of place and
    `Zygote` is loaded then, we use `AutoZygote`. In all other, cases `FiniteDiff` is used.
"""
@concrete struct GaussNewton{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    linesearch
    vjp_autodiff
end

function set_ad(alg::GaussNewton{CJ}, ad) where {CJ}
    return GaussNewton{CJ}(ad, alg.linsolve, alg.precs, alg.linesearch, alg.vjp_autodiff)
end

function GaussNewton(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        linesearch = nothing, vjp_autodiff = nothing, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return GaussNewton{_unwrap_val(concrete_jac)}(ad, linsolve, precs, linesearch,
        vjp_autodiff)
end

@concrete mutable struct GaussNewtonCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    fu
    fu_cache
    du
    dfu
    p
    uf
    linsolve
    J
    JᵀJ
    Jᵀf
    jac_cache
    force_stop
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    prob
    stats::NLStats
    tc_cache_1
    tc_cache_2
    ls_cache
    trace
end

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem{uType, iip}, alg_::GaussNewton,
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob

    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)

    uf, linsolve, J, fu_cache, jac_cache, du, JᵀJ, Jᵀf = jacobian_caches(alg, f, u, p,
        Val(iip); linsolve_with_JᵀJ = Val(__needs_square_A(alg, u)))

    abstol, reltol, tc_cache_1 = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    _, _, tc_cache_2 = init_termination_cache(abstol, reltol, fu, u, termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

    @bb u_cache = copy(u)
    @bb dfu = copy(fu)

    return GaussNewtonCache{iip}(f, alg, u, u_cache, fu, fu_cache, du, dfu, p, uf,
        linsolve, J, JᵀJ, Jᵀf, jac_cache, false, maxiters, internalnorm, ReturnCode.Default,
        abstol, reltol, prob, NLStats(1, 0, 0, 0, 0), tc_cache_1, tc_cache_2,
        init_linesearch_cache(alg.linesearch, f, u, p, fu, Val(iip)), trace)
end

function perform_step!(cache::GaussNewtonCache{iip}) where {iip}
    cache.J = jacobian!!(cache.J, cache)

    # Use normal form to solve the Linear Problem
    if cache.JᵀJ !== nothing
        __update_JᵀJ!(cache)
        __update_Jᵀf!(cache)
        A, b = __maybe_symmetric(cache.JᵀJ), _vec(cache.Jᵀf)
    else
        A, b = cache.J, _vec(cache.fu)
    end

    linres = dolinsolve(cache.alg.precs, cache.linsolve; A, b, linu = _vec(cache.du),
        cache.p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    cache.du = _restructure(cache.du, linres.u)

    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)
    evaluate_f(cache, cache.u, cache.p)
    update_trace!(cache, α)

    check_and_update!(cache.tc_cache_1, cache, cache.fu, cache.u, cache.u_cache)
    if !cache.force_stop
        @bb @. cache.dfu = cache.fu .- cache.dfu
        check_and_update!(cache.tc_cache_2, cache, cache.dfu, cache.u, cache.u_cache)
    end

    @bb copyto!(cache.u_cache, cache.u)
    @bb copyto!(cache.dfu, cache.fu)

    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

# FIXME: Reinit `JᵀJ` operator if `p` is changed
function __reinit_internal!(cache::GaussNewtonCache;
        termination_condition = get_termination_mode(cache.tc_cache_1), kwargs...)
    abstol, reltol, tc_cache_1 = init_termination_cache(cache.abstol, cache.reltol,
        cache.fu, cache.u, termination_condition)
    _, _, tc_cache_2 = init_termination_cache(cache.abstol, cache.reltol, cache.fu,
        cache.u, termination_condition)

    cache.tc_cache_1 = tc_cache_1
    cache.tc_cache_2 = tc_cache_2
    cache.abstol = abstol
    cache.reltol = reltol
    return nothing
end
