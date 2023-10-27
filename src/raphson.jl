"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
        precs = DEFAULT_PRECS, adkwargs...)

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
"""
@concrete struct NewtonRaphson{CJ, AD} <:
                 AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    linesearch
end

function set_ad(alg::NewtonRaphson{CJ}, ad) where {CJ}
    return NewtonRaphson{CJ}(ad, alg.linsolve, alg.precs, alg.linesearch)
end

function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
    linesearch = LineSearch(), precs = DEFAULT_PRECS, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return NewtonRaphson{_unwrap_val(concrete_jac)}(ad,
        linsolve,
        precs,
        linesearch)
end

@concrete mutable struct NewtonRaphsonCache{iip} <: AbstractNonlinearSolveCache{iip}
    f
    alg
    u
    u_prev
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
    reltol
    prob
    stats::NLStats
    lscache
    termination_condition
    tc_storage
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::NewtonRaphson, args...;
    alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
    termination_condition = nothing, internalnorm = DEFAULT_NORM,
    linsolve_kwargs = (;), kwargs...) where {uType, iip}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob
    u = alias_u0 ? u0 : deepcopy(u0)
    fu1 = evaluate_f(prob, u)
    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)

    abstol, reltol, termination_condition = _init_termination_elements(abstol,
        reltol, termination_condition, eltype(u))

    mode = DiffEqBase.get_termination_mode(termination_condition)

    storage = mode ∈ DiffEqBase.SAFE_TERMINATION_MODES ? NLSolveSafeTerminationResult() :
              nothing

    return NewtonRaphsonCache{iip}(f, alg, u, copy(u), fu1, fu2, du, p, uf, linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, reltol, prob,
        NLStats(1, 0, 0, 0, 0),
        init_linesearch_cache(alg.linesearch, f, u, p, fu1, Val(iip)),
        termination_condition, storage)
end

function perform_step!(cache::NewtonRaphsonCache{true})
    @unpack u, u_prev, fu1, f, p, alg, J, linsolve, du, tc_storage = cache
    jacobian!!(J, cache)

    termination_condition = cache.termination_condition(tc_storage)

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve; A = J, b = _vec(fu1), linu = _vec(du),
        p, reltol = cache.abstol)
    cache.linsolve = linres.cache

    # Line Search
    α = perform_linesearch!(cache.lscache, u, du)
    _axpy!(-α, du, u)
    f(cache.fu1, u, p)

    termination_condition(cache.fu1, u, u_prev, cache.abstol, cache.reltol) &&
        (cache.force_stop = true)

    @. u_prev = u
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::NewtonRaphsonCache{false})
    @unpack u, u_prev, fu1, f, p, alg, linsolve, tc_storage = cache

    termination_condition = cache.termination_condition(tc_storage)

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

    termination_condition(cache.fu1, cache.u, u_prev, cache.abstol, cache.reltol) &&
        (cache.force_stop = true)

    cache.u_prev = @. cache.u
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function SciMLBase.reinit!(cache::NewtonRaphsonCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, reltol = cache.reltol,
    termination_condition = cache.termination_condition,
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

    termination_condition = _get_reinit_termination_condition(cache, abstol, reltol,
        termination_condition)

    cache.abstol = abstol
    cache.reltol = reltol
    cache.termination_condition = termination_condition
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
