"""
    PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, alpha_initial = 1e-3, adkwargs...)

An implementation of PseudoTransient method that is used to solve steady state problems in
an accelerated manner. It uses an adaptive time-stepping to integrate an initial value of
nonlinear problem until sufficient accuracy in the desired steady-state is achieved to
switch over to Newton's method and gain a rapid convergence. This implementation
specifically uses "switched evolution relaxation" SER method. For detail information about
the time-stepping and algorithm, please see the paper:
[Coffey, Todd S. and Kelley, C. T. and Keyes, David E. (2003), Pseudotransient Continuation and Differential-Algebraic Equations,
SIAM Journal on Scientific Computing,25, 553-569.](https://doi.org/10.1137/S106482750241044X)

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
  - `alpha_initial` : the initial pseudo time step. it defaults to 1e-3. If it is small,
    you are going to need more iterations to converge but it can be more stable.
"""
@concrete struct PseudoTransient{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    alpha_initial
end

#concrete_jac(::PseudoTransient{CJ}) where {CJ} = CJ
function set_ad(alg::PseudoTransient{CJ}, ad) where {CJ}
    return PseudoTransient{CJ}(ad, alg.linsolve, alg.precs, alg.alpha_initial)
end

function PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, alpha_initial = 1e-3, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    return PseudoTransient{_unwrap_val(concrete_jac)}(ad, linsolve, precs, alpha_initial)
end

@concrete mutable struct PseudoTransientCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_prev
    fu1
    fu2
    du
    p
    alpha
    res_norm
    uf
    linsolve
    J
    jac_cache
    force_stop
    maxiters::Int
    internalnorm
    retcode::ReturnCode.T
    abstol
    reltol
    prob
    stats::NLStats
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::PseudoTransient,
        args...; alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm = DEFAULT_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip}
    alg = get_concrete_algorithm(alg_, prob)

    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu1 = evaluate_f(prob, u)
    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)
    alpha = convert(eltype(u), alg.alpha_initial)
    res_norm = internalnorm(fu1)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu1, u,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu1, ApplyArray(__zero, J), du; kwargs...)

    return PseudoTransientCache{iip}(f, alg, u, copy(u), fu1, fu2, du, p, alpha, res_norm,
        uf, linsolve, J, jac_cache, false, maxiters, internalnorm, ReturnCode.Default,
        abstol, reltol, prob, NLStats(1, 0, 0, 0, 0), tc_cache, trace)
end

function perform_step!(cache::PseudoTransientCache{true})
    @unpack u, u_prev, fu1, f, p, alg, J, linsolve, du, alpha = cache
    jacobian!!(J, cache)

    inv_alpha = inv(alpha)
    if J isa SciMLBase.AbstractSciMLOperator
        J = J - inv_alpha * I
    else
        idxs = diagind(J)
        if fast_scalar_indexing(J)
            @inbounds for i in axes(J, 1)
                J[i, i] = J[i, i] - inv_alpha
            end
        else
            @.. broadcast=false @view(J[idxs])=@view(J[idxs]) - inv_alpha
        end
    end

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve; A = J, b = _vec(fu1), linu = _vec(du),
        p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    @. u = u - du
    f(fu1, u, p)

    update_trace!(cache.trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache), J,
        cache.du)

    new_norm = cache.internalnorm(fu1)
    cache.alpha *= cache.res_norm / new_norm
    cache.res_norm = new_norm

    check_and_update!(cache, cache.fu1, cache.u, cache.u_prev)

    @. u_prev = u
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::PseudoTransientCache{false})
    @unpack u, u_prev, fu1, f, p, alg, linsolve, alpha = cache

    cache.J = jacobian!!(cache.J, cache)

    inv_alpha = inv(alpha)
    cache.J = cache.J - inv_alpha * I
    # u = u - J \ fu
    if linsolve === nothing
        cache.du = fu1 / cache.J
    else
        linres = dolinsolve(alg.precs, linsolve; A = cache.J, b = _vec(fu1),
            linu = _vec(cache.du), p, reltol = cache.abstol)
        cache.linsolve = linres.cache
    end
    cache.u = @. u - cache.du  # `u` might not support mutation
    cache.fu1 = f(cache.u, p)

    update_trace!(cache.trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache), cache.J,
        cache.du)

    new_norm = cache.internalnorm(fu1)
    cache.alpha *= cache.res_norm / new_norm
    cache.res_norm = new_norm

    check_and_update!(cache, cache.fu1, cache.u, cache.u_prev)

    cache.u_prev = cache.u
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function SciMLBase.reinit!(cache::PseudoTransientCache{iip}, u0 = cache.u; p = cache.p,
        alpha = cache.alpha, abstol = cache.abstol, reltol = cache.reltol,
        termination_condition = get_termination_mode(cache.tc_cache),
        maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu1, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu1 = cache.f(cache.u, p)
    end

    reset!(cache.trace)
    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, cache.fu1, cache.u,
        termination_condition)

    cache.alpha = convert(eltype(cache.u), alpha)
    cache.res_norm = cache.internalnorm(cache.fu1)
    cache.abstol = abstol
    cache.reltol = reltol
    cache.tc_cache = tc_cache
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
