"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = LineSearch(),
        precs = DEFAULT_PRECS, reuse = true, reusetol = 1e-6, adkwargs...)

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

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
  - `reuse`: Determines if the Jacobian is reused between (quasi-)Newton steps. Defaults to
    `true`. If `true` we check how far we stepped with the same Jacobian, and automatically
    take a new Jacobian if we stepped more than `reusetol` or if convergence slows or starts
    to diverge. If `false`, the Jacobian is updated in each step.
"""
@concrete struct NewtonRaphson{CJ, AD} <:
                 AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    linesearch
    reusetol
    reuse::Bool
end

function set_ad(alg::NewtonRaphson{CJ}, ad) where {CJ}
    return NewtonRaphson{CJ}(ad,
        alg.linsolve,
        alg.precs,
        alg.linesearch,
        alg.reusetol,
        alg.reuse)
end

function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
        linesearch = LineSearch(), precs = DEFAULT_PRECS, reuse = true, reusetol = 1e-1,
        adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return NewtonRaphson{_unwrap_val(concrete_jac)}(ad,
        linsolve,
        precs,
        linesearch,
        reusetol,
        reuse)
end

@concrete mutable struct NewtonRaphsonCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_prev
    Δu
    fu1
    res_norm_prev
    fu2
    du
    p
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
    ls_cache
    tc_cache
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::NewtonRaphson, args...;
        alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm = DEFAULT_NORM,
        linsolve_kwargs = (;), kwargs...) where {uType, iip}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    u_prev = copy(u)
    Δu = zero(u)

    fu1 = evaluate_f(prob, u)
    res_norm_prev = internalnorm(fu1)

    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu1, u,
        termination_condition)

    return NewtonRaphsonCache{iip}(f, alg, u, u_prev, Δu, fu1, res_norm_prev, fu2, du, p,
        uf,
        linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, reltol, prob,
        NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu1, Val(iip)), tc_cache)
end

function perform_step!(cache::NewtonRaphsonCache{true})
    @unpack u, u_prev, Δu, fu1, res_norm_prev, f, p, alg, J, linsolve, du = cache
    @unpack reuse = alg

    if reuse
        # check if residual increased and check how far we stepped
        res_norm = cache.internalnorm(fu1)
        update = (res_norm > res_norm_prev) || (cache.internalnorm(Δu) > alg.reusetol)
        if update || cache.stats.njacs == 0
            jacobian!!(J, cache)
            cache.stats.njacs += 1
            Δu .*= false
            # u = u - J \ fu
            linres = dolinsolve(alg.precs, linsolve; A = J, b = _vec(fu1), linu = _vec(du),
                p, reltol = cache.abstol)
        else
            # u = u - J \ fu
            linres = dolinsolve(alg.precs, linsolve; b = _vec(fu1), linu = _vec(du),
                p, reltol = cache.abstol)
        end
        cache.res_norm_prev = res_norm
    else
        jacobian!!(J, cache)
        cache.stats.njacs += 1
        # u = u - J \ fu
        linres = dolinsolve(alg.precs, linsolve; A = J, b = _vec(fu1), linu = _vec(du),
            p, reltol = cache.abstol)
    end

    cache.linsolve = linres.cache

    # Line Search
    α = perform_linesearch!(cache.ls_cache, u, du)
    _axpy!(-α, du, u)
    f(cache.fu1, u, p)

    check_and_update!(cache, cache.fu1, cache.u, cache.u_prev)
    @. Δu += u - u_prev
    @. u_prev = u
    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::NewtonRaphsonCache{false})
    @unpack u, u_prev, Δu, fu1, res_norm_prev, f, p, alg, linsolve = cache
    @unpack reuse = alg

    if reuse
        # check if residual increased and check how far we stepped
        res_norm = cache.internalnorm(fu1)
        update = (res_norm > res_norm_prev) || (cache.internalnorm(Δu) > alg.reusetol)
        if update || cache.stats.njacs == 0
            cache.J = jacobian!!(cache.J, cache)
            cache.stats.njacs += 1
            # u = u - J \ fu
            if linsolve === nothing
                cache.du = fu1 / cache.J
            else
                linres = dolinsolve(alg.precs, linsolve; A = cache.J, b = _vec(fu1),
                    linu = _vec(cache.du), p, reltol = cache.abstol)
                cache.linsolve = linres.cache
            end
        else
            # u = u - J \ fu
            if linsolve === nothing
                cache.du = fu1 / cache.J
            else
                linres = dolinsolve(alg.precs, linsolve; b = _vec(fu1),
                    linu = _vec(cache.du), p, reltol = cache.abstol)
                cache.linsolve = linres.cache
            end
        end
        cache.res_norm_prev = res_norm
    else
        cache.J = jacobian!!(cache.J, cache)
        # cache.Δu *= false
        cache.stats.njacs += 1

        # u = u - J \ fu
        if linsolve === nothing
            cache.du = fu1 / cache.J
        else
            linres = dolinsolve(alg.precs, linsolve; A = cache.J, b = _vec(fu1),
                linu = _vec(cache.du), p, reltol = cache.abstol)
            cache.linsolve = linres.cache
        end
    end

    # Line Search
    α = perform_linesearch!(cache.ls_cache, u, cache.du)
    cache.u = @. u - α * cache.du  # `u` might not support mutation
    cache.fu1 = f(cache.u, p)

    check_and_update!(cache, cache.fu1, cache.u, cache.u_prev)
    cache.Δu = @. cache.u - cache.u_prev
    cache.u_prev = cache.u
    cache.stats.nf += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function SciMLBase.reinit!(cache::NewtonRaphsonCache{iip}, u0 = cache.u; p = cache.p,
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

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, cache.fu1, cache.u,
        termination_condition)

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
