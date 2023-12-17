"""
    PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, alpha_initial = 1e-3,update_alpha = switched_evolution_relaxation,adkwargs...)

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
    algorithms, consult the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `alpha_initial` : the initial pseudo time step. it defaults to 1e-3. If it is small,
    you are going to need more iterations to converge but it can be more stable.
  - `update_alpha`  : a function that specifies the schema for updating alpha. The function should
    like update_alpha(alpha::Number,res_norm::Number,nsteps::Int,u,u_prev,fu,norm::F). The default is
    a function that uses "switched evolution relaxation" SER method to update alpha.
"""
@concrete struct PseudoTransient{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    alpha_initial
    update_alpha
end

function set_ad(alg::PseudoTransient{CJ}, ad) where {CJ}
    return PseudoTransient{CJ}(ad, alg.linsolve, alg.precs,
        alg.alpha_initial, alg.update_alpha)
end

function PseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, alpha_initial = 1e-3,
        update_alpha::F = switched_evolution_relaxation,
        autodiff = nothing) where {F}
    return PseudoTransient{_unwrap_val(concrete_jac)}(autodiff, linsolve, precs,
        alpha_initial, update_alpha)
end

"""
    RobustPseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, alpha_initial = 1e-6,threshold::Int = 100)

This is just an alias to the PseudoTransient method, but now it uses a more stable and robust schema for
updating alpha. It has an argument `threshold` that determines for how many steps alpha remains constant before switching to SER method.
See also [`PseudoTransient`](@ref) for the remaining keyword arguments
"""
function RobustPseudoTransient(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, alpha_initial = 1e-6, threshold::Int = 100,
        autodiff = nothing)
    return PseudoTransient{_unwrap_val(concrete_jac)}(autodiff, linsolve, precs,
        alpha_initial, wrapper_robust_update(threshold))
end

@concrete mutable struct PseudoTransientCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_cache
    fu
    fu_cache
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
    u = __maybe_unaliased(u0, alias_u0)
    fu = evaluate_f(prob, u)
    uf, linsolve, J, fu_cache, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)
    alpha = convert(eltype(u), alg.alpha_initial)
    res_norm = internalnorm(fu)

    @bb u_cache = copy(u)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu, u,
        termination_condition)
    trace = init_nonlinearsolve_trace(alg, u, fu, ApplyArray(__zero, J), du; kwargs...)

    return PseudoTransientCache{iip}(f, alg, u, u_cache, fu, fu_cache, du, p, alpha,
        res_norm, uf, linsolve, J, jac_cache, false, maxiters, internalnorm,
        ReturnCode.Default, abstol, reltol, prob, NLStats(1, 0, 0, 0, 0), tc_cache, trace)
end

function perform_step!(cache::PseudoTransientCache{iip}) where {iip}
    @unpack alg = cache

    cache.J = jacobian!!(cache.J, cache)

    inv_α = inv(cache.alpha)
    if cache.J isa SciMLOperators.AbstractSciMLOperator
        A = cache.J - inv_α * I
    elseif setindex_trait(cache.J) === CanSetindex()
        if fast_scalar_indexing(cache.J)
            @inbounds for i in axes(cache.J, 1)
                cache.J[i, i] = cache.J[i, i] - inv_α
            end
        else
            idxs = diagind(cache.J)
            @.. broadcast=false @view(cache.J[idxs])=@view(cache.J[idxs]) - inv_α
        end
        A = cache.J
    else
        cache.J = cache.J - inv_α * I
        A = cache.J
    end

    # u = u - J \ fu
    linres = dolinsolve(cache, alg.precs, cache.linsolve; A, b = _vec(cache.fu),
        linu = _vec(cache.du), cache.p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    cache.du = _restructure(cache.du, linres.u)

    @bb axpy!(-true, cache.du, cache.u)

    evaluate_f(cache, cache.u, cache.p)

    update_trace!(cache, true)

    new_norm = cache.internalnorm(cache.fu)
    cache.alpha = cache.alg.update_alpha(cache.alpha, cache.res_norm, cache.stats.nsteps,
        cache.u, cache.u_cache, cache.fu, cache.internalnorm)
    cache.res_norm = new_norm

    check_and_update!(cache, cache.fu, cache.u, cache.u_cache)

    @bb copyto!(cache.u_cache, cache.u)
    return nothing
end

function __reinit_internal!(cache::PseudoTransientCache; alpha = cache.alg.alpha_initial,
        kwargs...)
    cache.alpha = convert(eltype(cache.u), alpha)
    cache.res_norm = cache.internalnorm(cache.fu)
    return nothing
end
