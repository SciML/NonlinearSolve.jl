"""
    GaussNewton(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, adkwargs...)

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear least squares problems.

!!! note
    In most practical situations, users should prefer using `LevenbergMarquardt` instead! It
    is a more general extension of `Gauss-Newton` Method.

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

!!! warning

    Jacobian-Free version of `GaussNewton` doesn't work yet, and it forces jacobian
    construction. This will be fixed in the near future.
"""
@concrete struct GaussNewton{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
end

function set_ad(alg::GaussNewton{CJ}, ad) where {CJ}
    return GaussNewton{CJ}(ad, alg.linsolve, alg.precs)
end

function GaussNewton(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    return GaussNewton{_unwrap_val(concrete_jac)}(ad, linsolve, precs)
end

@concrete mutable struct GaussNewtonCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_prev
    fu1
    fu2
    fu_new
    du
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
end

function SciMLBase.__init(prob::NonlinearLeastSquaresProblem{uType, iip}, alg_::GaussNewton,
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm::F = DEFAULT_NORM,
        kwargs...) where {uType, iip, F}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob

    linsolve_with_JᵀJ = Val(_needs_square_A(alg, u0))

    u = alias_u0 ? u0 : deepcopy(u0)
    fu1 = evaluate_f(prob, u)

    if SciMLBase._unwrap_val(linsolve_with_JᵀJ)
        uf, linsolve, J, fu2, jac_cache, du, JᵀJ, Jᵀf = jacobian_caches(alg, f, u, p,
            Val(iip); linsolve_with_JᵀJ)
    else
        uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p,
            Val(iip); linsolve_with_JᵀJ)
        JᵀJ, Jᵀf = nothing, nothing
    end

    abstol, reltol, tc_cache_1 = init_termination_cache(abstol, reltol, fu1, u,
        termination_condition)
    _, _, tc_cache_2 = init_termination_cache(abstol, reltol, fu1, u, termination_condition)

    return GaussNewtonCache{iip}(f, alg, u, copy(u), fu1, fu2, zero(fu1), du, p, uf,
        linsolve, J, JᵀJ, Jᵀf, jac_cache, false, maxiters, internalnorm, ReturnCode.Default,
        abstol, reltol, prob, NLStats(1, 0, 0, 0, 0), tc_cache_1, tc_cache_2)
end

function perform_step!(cache::GaussNewtonCache{true})
    @unpack u, u_prev, fu1, f, p, alg, J, JᵀJ, Jᵀf, linsolve, du = cache
    jacobian!!(J, cache)

    if JᵀJ !== nothing
        __matmul!(JᵀJ, J', J)
        __matmul!(Jᵀf, J', fu1)
    end

    # u = u - JᵀJ \ Jᵀfu
    if cache.JᵀJ === nothing
        linres = dolinsolve(alg.precs, linsolve; A = J, b = _vec(fu1), linu = _vec(du),
            p, reltol = cache.abstol)
    else
        linres = dolinsolve(alg.precs, linsolve; A = __maybe_symmetric(JᵀJ), b = _vec(Jᵀf),
            linu = _vec(du), p, reltol = cache.abstol)
    end
    cache.linsolve = linres.cache
    @. u = u - du
    f(cache.fu_new, u, p)

    check_and_update!(cache.tc_cache_1, cache, cache.fu_new, cache.u, cache.u_prev)
    if !cache.force_stop
        cache.fu1 .= cache.fu_new .- cache.fu1
        check_and_update!(cache.tc_cache_2, cache, cache.fu1, cache.u, cache.u_prev)
    end

    @. u_prev = u
    cache.fu1 .= cache.fu_new
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::GaussNewtonCache{false})
    @unpack u, u_prev, fu1, f, p, alg, linsolve = cache

    cache.J = jacobian!!(cache.J, cache)

    if cache.JᵀJ !== nothing
        cache.JᵀJ = cache.J' * cache.J
        cache.Jᵀf = cache.J' * fu1
    end

    # u = u - J \ fu
    if linsolve === nothing
        cache.du = fu1 / cache.J
    else
        if cache.JᵀJ === nothing
            linres = dolinsolve(alg.precs, linsolve; A = cache.J, b = _vec(fu1),
                linu = _vec(cache.du), p, reltol = cache.abstol)
        else
            linres = dolinsolve(alg.precs, linsolve; A = __maybe_symmetric(cache.JᵀJ),
                b = _vec(cache.Jᵀf), linu = _vec(cache.du), p, reltol = cache.abstol)
        end
        cache.linsolve = linres.cache
    end
    cache.u = @. u - cache.du  # `u` might not support mutation
    cache.fu_new = f(cache.u, p)

    check_and_update!(cache.tc_cache_1, cache, cache.fu_new, cache.u, cache.u_prev)
    if !cache.force_stop
        cache.fu1 = cache.fu_new .- cache.fu1
        check_and_update!(cache.tc_cache_2, cache, cache.fu1, cache.u, cache.u_prev)
    end

    cache.u_prev = cache.u
    cache.fu1 = cache.fu_new
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function SciMLBase.reinit!(cache::GaussNewtonCache{iip}, u0 = cache.u; p = cache.p,
        abstol = cache.abstol, reltol = cache.reltol, maxiters = cache.maxiters,
        termination_condition = get_termination_mode(cache.tc_cache)) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu1, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu1 = cache.f(cache.u, p)
    end

    abstol, reltol, tc_cache_1 = init_termination_cache(abstol, reltol, cache.fu1, cache.u,
        termination_condition)
    _, _, tc_cache_2 = init_termination_cache(abstol, reltol, cache.fu1, cache.u,
        termination_condition)

    cache.abstol = abstol
    cache.reltol = reltol
    cache.tc_cache_1 = tc_cache_1
    cache.tc_cache_2 = tc_cache_2
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
