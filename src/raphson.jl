"""
```julia
NewtonRaphson(; chunk_size = Val{0}(), autodiff = Val{true}(),
              standardtag = Val{true}(), concrete_jac = nothing,
              diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS)
```

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

  - `chunk_size`: the chunk size used by the internal ForwardDiff.jl automatic differentiation
    system. This allows for multiple derivative columns to be computed simultaneously,
    improving performance. Defaults to `0`, which is equivalent to using ForwardDiff.jl's
    default chunk size mechanism. For more details, see the documentation for
    [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
  - `autodiff`: whether to use forward-mode automatic differentiation for the Jacobian.
    Note that this argument is ignored if an analytical Jacobian is passed, as that will be
    used instead. Defaults to `Val{true}`, which means ForwardDiff.jl via
    SparseDiffTools.jl is used by default. If `Val{false}`, then FiniteDiff.jl is used for
    finite differencing.
  - `standardtag`: whether to use a standardized tag definition for the purposes of automatic
    differentiation. Defaults to true, which thus uses the `NonlinearSolveTag`. If `Val{false}`,
    then ForwardDiff's default function naming tag is used, which results in larger stack
    traces.
  - `concrete_jac`: whether to build a concrete Jacobian. If a Krylov-subspace method is used,
    then the Jacobian will not be constructed and instead direct Jacobian-vector products
    `J*v` are computed using forward-mode automatic differentiation or finite differencing
    tricks (without ever constructing the Jacobian). However, if the Jacobian is still needed,
    for example for a preconditioner, `concrete_jac = true` can be passed in order to force
    the construction of the Jacobian.
  - `diff_type`: the type of finite differencing used if `autodiff = false`. Defaults to
    `Val{:forward}` for forward finite differences. For more details on the choices, see the
    [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) documentation.
  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).

!!! note

    Currently, the linear solver and chunk size choice only applies to in-place defined
    `NonlinearProblem`s. That is expected to change in the future.
"""
struct NewtonRaphson{CS, AD, FDT, L, P, ST, CJ} <:
       AbstractNewtonAlgorithm{CS, AD, FDT, ST, CJ}
    linsolve::L
    precs::P
end

function NewtonRaphson(; chunk_size = Val{0}(), autodiff = Val{true}(),
    standardtag = Val{true}(), concrete_jac = nothing,
    diff_type = Val{:forward}, linsolve = nothing, precs = DEFAULT_PRECS)
    NewtonRaphson{_unwrap_val(chunk_size), _unwrap_val(autodiff), diff_type,
        typeof(linsolve), typeof(precs), _unwrap_val(standardtag),
        _unwrap_val(concrete_jac)}(linsolve,
        precs)
end

mutable struct NewtonRaphsonCache{iip, fType, algType, uType, duType, resType, pType,
    INType, tolType,
    probType, ufType, L, jType, JC}
    f::fType
    alg::algType
    u::uType
    fu::resType
    p::pType
    uf::ufType
    linsolve::L
    J::jType
    du1::duType
    jac_config::JC
    force_stop::Bool
    maxiters::Int
    internalnorm::INType
    retcode::SciMLBase.ReturnCode.T
    abstol::tolType
    prob::probType
    stats::NLStats

    function NewtonRaphsonCache{iip}(f::fType, alg::algType, u::uType, fu::resType,
        p::pType, uf::ufType, linsolve::L, J::jType,
        du1::duType,
        jac_config::JC, force_stop::Bool, maxiters::Int,
        internalnorm::INType,
        retcode::SciMLBase.ReturnCode.T, abstol::tolType,
        prob::probType,
        stats::NLStats) where {
        iip, fType, algType, uType,
        duType, resType, pType, INType,
        tolType,
        probType, ufType, L, jType, JC}
        new{iip, fType, algType, uType, duType, resType, pType, INType, tolType,
            probType, ufType, L, jType, JC}(f, alg, u, fu, p,
            uf, linsolve, J, du1, jac_config,
            force_stop, maxiters, internalnorm,
            retcode, abstol, prob, stats)
    end
end

function jacobian_caches(alg::NewtonRaphson, f, u, p, ::Val{true})
    uf = JacobianWrapper(f, p)

    du1 = zero(u)
    du2 = zero(u)
    tmp = zero(u)
    J, jac_config = build_jac_and_jac_config(alg, f, uf, du1, u, tmp, du2)

    linprob = LinearProblem(J, _vec(zero(u)); u0 = _vec(zero(u)))
    weight = similar(u)
    # Q: Setting this to false leads to residual = 0 in GMRES?
    recursivefill!(weight, true)

    Pl, Pr = wrapprecs(alg.precs(J, nothing, u, p, nothing, nothing, nothing, nothing,
            nothing)..., weight)
    linsolve = init(linprob, alg.linsolve; alias_A = true, alias_b = true, Pl, Pr)

    uf, linsolve, J, du1, jac_config
end

function jacobian_caches(alg::NewtonRaphson, f, u, p, ::Val{false})
    JacobianWrapper(f, p), nothing, nothing, nothing, nothing
end

function SciMLBase.__init(prob::NonlinearProblem{uType, iip}, alg::NewtonRaphson,
    args...;
    alias_u0 = false,
    maxiters = 1000,
    abstol = 1e-6,
    internalnorm = DEFAULT_NORM,
    kwargs...) where {uType, iip}
    if alias_u0
        u = prob.u0
    else
        u = deepcopy(prob.u0)
    end
    f = prob.f
    p = prob.p
    if iip
        fu = zero(u)
        f(fu, u, p)
    else
        fu = f(u, p)
    end
    uf, linsolve, J, du1, jac_config = jacobian_caches(alg, f, u, p, Val(iip))

    return NewtonRaphsonCache{iip}(f, alg, u, fu, p, uf, linsolve, J, du1, jac_config,
        false, maxiters, internalnorm,
        ReturnCode.Default, abstol, prob, NLStats(1, 0, 0, 0, 0))
end

function perform_step!(cache::NewtonRaphsonCache{true})
    @unpack u, fu, f, p, alg = cache
    @unpack J, linsolve, du1 = cache
    jacobian!(J, cache)

    # u = u - J \ fu
    linres = dolinsolve(alg.precs, linsolve, A = J, b = _vec(fu), linu = _vec(du1),
        p = p, reltol = cache.abstol)
    cache.linsolve = linres.cache
    @. u = u - du1
    f(fu, u, p)

    if cache.internalnorm(cache.fu) < cache.abstol
        cache.force_stop = true
    end
    cache.stats.nf += 1
    cache.stats.njacs += 1
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    return nothing
end

function perform_step!(cache::NewtonRaphsonCache{false})
    @unpack u, fu, f, p = cache
    J = jacobian(cache, f)
    cache.u = u - J \ fu
    cache.fu = f(cache.u, p)
    if iszero(cache.fu) || cache.internalnorm(cache.fu) < cache.abstol
        cache.force_stop = true
    end
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

    SciMLBase.build_solution(cache.prob, cache.alg, cache.u, cache.fu;
        retcode = cache.retcode, stats = cache.stats)
end

function SciMLBase.reinit!(cache::NewtonRaphsonCache{iip}, u0 = cache.u; p = cache.p,
    abstol = cache.abstol, maxiters = cache.maxiters) where {iip}
    cache.p = p
    if iip
        recursivecopy!(cache.u, u0)
        cache.f(cache.fu, cache.u, p)
    else
        # don't have alias_u0 but cache.u is never mutated for OOP problems so it doesn't matter
        cache.u = u0
        cache.fu = cache.f(cache.u, p)
    end
    cache.abstol = abstol
    cache.maxiters = maxiters
    cache.stats.nf = 1
    cache.stats.nsteps = 1
    cache.force_stop = false
    cache.retcode = ReturnCode.Default
    return cache
end
