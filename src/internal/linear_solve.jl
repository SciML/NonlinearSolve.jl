abstract type AbstractLinearSolverCache <: Function end

@concrete mutable struct LinearSolverCache <: AbstractLinearSolverCache
    lincache
    linsolve
    A
    b
    precs
    nsolve
    nfactors
end

@inline function LinearSolverCache(alg::AbstractNonlinearSolveAlgorithm, args...; kwargs...)
    return LinearSolverCache(alg.linsolve, args...; kwargs...)
end
@inline function LinearSolverCache(alg, linsolve, A::Number, b, args...; kwargs...)
    return LinearSolverCache(nothing, nothing, A, b, nothing, 0, 0)
end
@inline function LinearSolveCache(alg, ::Nothing, A::SMatrix, b, args...; kwargs...)
    # Default handling for SArrays caching in LinearSolve is not the best. Override it here
    return LinearSolverCache(nothing, nothing, A, _vec(b), nothing, 0, 0)
end
function LinearSolverCache(alg, linsolve, A, b, u; kwargs...)
    linprob = LinearProblem(A, _vec(b); u0 = _vec(u), kwargs...)

    weight = __init_ones(u)
    if __hasfield(alg, Val(:precs))
        precs = alg.precs
        Pl_, Pr_ = precs(A, nothing, u, nothing, nothing, nothing, nothing, nothing,
            nothing)
    else
        precs, Pl_, Pr_ = nothing, nothing, nothing
    end
    Pl, Pr = wrapprecs(Pl_, Pr_, weight)

    lincache = init(linprob, linsolve; alias_A = true, alias_b = true, Pl, Pr)

    return LinearSolverCache(lincache, linsolve, nothing, nothing, precs, 0, 0)
end
# TODO: For Krylov Versions
# linsolve_caches(A::KrylovJᵀJ, b, u, p, alg) = linsolve_caches(A.JᵀJ, b, u, p, alg)

# Direct Linear Solve Case without Caching
function (cache::LinearSolveCache{Nothing})(; A = nothing, b = nothing, kwargs...)
    cache.nsolve += 1
    cache.nfactors += 1
    A === nothing || (cache.A = A)
    b === nothing || (cache.b = b)
    return cache.A \ cache.b
end
# Use LinearSolve.jl
function (cache::LinearSolveCache)(; A = nothing, b = nothing, linu = nothing, du = nothing,
        p = nothing, weight = nothing, cachedata = nothing, reltol = nothing,
        abstol = nothing, reuse_A_if_factorization::Val{R} = Val(false),
        kwargs...) where {R}
    cache.nsolve += 1
    cache.nfactors += 1
    # TODO: Update `A`
    # A === nothing || (cache.A = A)
    #     # Some Algorithms would reuse factorization but it causes the cache to not reset in
    #     # certain cases
    #     if A !== nothing
    #         alg = __getproperty(linsolve, Val(:alg))
    #         if alg !== nothing && ((alg isa LinearSolve.AbstractFactorization) ||
    #             (alg isa LinearSolve.DefaultLinearSolver && !(alg ==
    #                LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES))))
    #             # Factorization Algorithm
    #             if reuse_A_if_factorization
    #                 cache.stats.nfactors -= 1
    #             else
    #                 linsolve.A = A
    #             end
    #         else
    #             linsolve.A = A
    #         end
    #     else
    #         cache.stats.nfactors -= 1
    #     end
    b === nothing || (cache.b = b)
    linu === nothing || (cache.linsolve.u = linu)

    Plprev = cache.linsolve.Pl isa ComposePreconditioner ? cache.linsolve.Pl.outer :
             cache.linsolve.Pl
    Prprev = cache.linsolve.Pr isa ComposePreconditioner ? cache.linsolve.Pr.outer :
             cache.linsolve.Pr

    if cache.precs === nothing
        _Pl, _Pr = nothing, nothing
    else
        _Pl, _Pr = cache.precs(cache.A, du, linu, p, nothing, A !== nothing, Plprev, Prprev,
            cachedata)
    end

    if (_Pl !== nothing || _Pr !== nothing)
        _weight = weight === nothing ?
                  (cache.linsolve.Pr isa Diagonal ? cache.linsolve.Pr.diag :
                   cache.linsolve.Pr.inner.diag) : weight
        Pl, Pr = wrapprecs(_Pl, _Pr, _weight)
        cache.linsolve.Pl = Pl
        cache.linsolve.Pr = Pr
    end

    if reltol === nothing && abstol === nothing
        linres = solve!(cache.linsolve)
    elseif reltol === nothing && abstol !== nothing
        linres = solve!(cache.linsolve; abstol)
    elseif reltol !== nothing && abstol === nothing
        linres = solve!(cache.linsolve; reltol)
    else
        linres = solve!(cache.linsolve; reltol, abstol)
    end

    cache.lincache = linres.cache

    return linres.u
end

@inline function __wrapprecs(_Pl, _Pr, weight)
    if _Pl !== nothing
        Pl = ComposePreconditioner(InvPreconditioner(Diagonal(_vec(weight))), _Pl)
    else
        Pl = InvPreconditioner(Diagonal(_vec(weight)))
    end

    if _Pr !== nothing
        Pr = ComposePreconditioner(Diagonal(_vec(weight)), _Pr)
    else
        Pr = Diagonal(_vec(weight))
    end

    return Pl, Pr
end
