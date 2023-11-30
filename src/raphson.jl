"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = nothing,
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
@concrete struct NewtonRaphson{CJ, AD} <: AbstractNewtonAlgorithm{CJ, AD}
    ad::AD
    linsolve
    precs
    linesearch
end

function set_ad(alg::NewtonRaphson{CJ}, ad) where {CJ}
    return NewtonRaphson{CJ}(ad, alg.linsolve, alg.precs, alg.linesearch)
end

function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS, adkwargs...)
    ad = default_adargs_to_adtype(; adkwargs...)
    linesearch = linesearch isa LineSearch ? linesearch : LineSearch(; method = linesearch)
    return NewtonRaphson{_unwrap_val(concrete_jac)}(ad, linsolve, precs, linesearch)
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
    ls_cache
    tc_cache
    trace
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg_::NewtonRaphson, args...;
        alias_u0 = false, maxiters = 1000, abstol = nothing, reltol = nothing,
        termination_condition = nothing, internalnorm = DEFAULT_NORM, linsolve_kwargs = (;),
        kwargs...) where {uType, iip}
    alg = get_concrete_algorithm(alg_, prob)
    @unpack f, u0, p = prob
    u = __maybe_unaliased(u0, alias_u0)
    fu1 = evaluate_f(prob, u)
    uf, linsolve, J, fu2, jac_cache, du = jacobian_caches(alg, f, u, p, Val(iip);
        linsolve_kwargs)

    abstol, reltol, tc_cache = init_termination_cache(abstol, reltol, fu1, u,
        termination_condition)

    ls_cache = init_linesearch_cache(alg.linesearch, f, u, p, fu1, Val(iip))
    trace = init_nonlinearsolve_trace(alg, u, fu1, ApplyArray(__zero, J), du; kwargs...)

    @bb u_prev = copy(u)

    return NewtonRaphsonCache{iip}(f, alg, u, u_prev, fu1, fu2, du, p, uf, linsolve, J,
        jac_cache, false, maxiters, internalnorm, ReturnCode.Default, abstol, reltol, prob,
        NLStats(1, 0, 0, 0, 0), ls_cache, tc_cache, trace)
end

function perform_step!(cache::NewtonRaphsonCache{iip}) where {iip}
    @unpack alg = cache

    cache.J = jacobian!!(cache.J, cache)

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, cache.linsolve; A = cache.J, b = _vec(cache.fu1),
        linu = _vec(cache.du), cache.p, reltol = cache.abstol)
    cache.linsolve = linres.cache

    !iip && (cache.du = linres.u)

    # Line Search
    α = perform_linesearch!(cache.ls_cache, cache.u, cache.du)
    @bb axpy!(-α, cache.du, cache.u)

    evaluate_f(cache, cache.u, cache.p)

    update_trace!(cache.trace, cache.stats.nsteps + 1, get_u(cache), get_fu(cache), cache.J,
        cache.du, α)

    check_and_update!(cache, cache.fu1, cache.u, cache.u_prev)

    @bb copyto!(cache.u_prev, cache.u)
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end
