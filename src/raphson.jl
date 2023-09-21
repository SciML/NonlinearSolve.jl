"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, adkwargs...)

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `AutoForwardDiff()`. Valid choices are types from ADTypes.jl.
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
"""
@concrete struct NewtonRaphson{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    linesearch
end

function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
    linesearch = LineSearch(), precs = DEFAULT_PRECS, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return NewtonRaphson{_unwrap_val(concrete_jac)}(ad, linsolve, precs, linesearch)
end

@concrete mutable struct NewtonRaphsonCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    fu1
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
    prob
    stats::NLStats
    lscache
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::NewtonRaphson, args...;
    alias_u0 = false, maxiters = 1000, abstol = 1e-6, internalnorm = DEFAULT_NORM,
    linsolve_kwargs = (;), kwargs...) where {uType, iip}
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu1 = evaluate_f(prob, u)
    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)

    return NewtonRaphsonCache{iip}(f, alg, u, fu1, fu2, du, p, uf, linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, prob,
        NLStats(1, 0, 0, 0, 0), LineSearchCache(alg.linesearch, f, u, p, fu1, Val(iip)))
end

function perform_step!(cache::NewtonRaphsonCache{true})
    @unpack u, fu1, f, p, alg, J, linsolve, du = cache
    jacobian!!(J, cache)

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve; A = J, b = _vec(fu1), linu = _vec(du),
        p, reltol = cache.abstol)
    cache.linsolve = linres.cache

    # Line Search
    α = perform_linesearch!(cache.lscache, u, du)
    @. u = u - α * du
    f(cache.fu1, u, p)

    cache.internalnorm(fu1) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::NewtonRaphsonCache{false})
    @unpack u, fu1, f, p, alg, linsolve = cache

    cache.J = jacobian!!(cache.J, cache)
    # u = u - J \ fu
    if linsolve === nothing
        cache.du = fu1 / cache.J
    else
        linres = dolinsolve(alg.precs, linsolve; A = cache.J, b = _vec(fu1),
            linu = _vec(cache.du), p, reltol = cache.abstol)
        cache.linsolve = linres.cache
    end

    # Line Search
    α = perform_linesearch!(cache.lscache, u, cache.du)
    cache.u = @. u - α * cache.du  # `u` might not support mutation
    cache.fu1 = f(cache.u, p)

    cache.internalnorm(fu1) < cache.abstol && (cache.force_stop = true)
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function SciMLBase.solve!(cache::NewtonRaphsonCache)
    while !cache.force_stop && cache.stats.nsteps < cache.maxiters
        perform_step!(cache)
        cache.stats.nsteps += 1
    end

    if cache.stats.nsteps == cache.maxiters
        cache.retcode = ReturnCode.MaxIters
    else
        cache.retcode = ReturnCode.Success
    end

    return SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu1;
        cache.retcode, cache.stats)
end
