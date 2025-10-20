module NonlinearSolveBaseLinearSolveExt

using ArrayInterface: ArrayInterface

using CommonSolve: CommonSolve, init, solve!
using LinearSolve: LinearSolve, QRFactorization, SciMLLinearSolveAlgorithm
using SciMLBase: ReturnCode, LinearProblem, LinearAliasSpecifier
using SciMLLogging: @SciMLMessage

using LinearAlgebra: ColumnNorm

using NonlinearSolveBase: NonlinearSolveBase, LinearSolveJLCache, LinearSolveResult, Utils, NonlinearVerbosity

function (cache::LinearSolveJLCache)(;
        A = nothing, b = nothing, linu = nothing,
        reuse_A_if_factorization = false, kwargs...
)
    cache.stats.nsolve += 1

    update_A!(cache, A, reuse_A_if_factorization)
    b !== nothing && setproperty!(cache.lincache, :b, b)
    linu !== nothing && NonlinearSolveBase.set_lincache_u!(cache, linu)
    
    linres = solve!(cache.lincache)
    if linres.retcode === ReturnCode.Failure
        return LinearSolveResult(; linres.u, success = false)
    else
        return LinearSolveResult(; linres.u)
    end
end

function NonlinearSolveBase.needs_square_A(linsolve::SciMLLinearSolveAlgorithm, ::Any)
    return LinearSolve.needs_square_A(linsolve)
end
function NonlinearSolveBase.needs_concrete_A(linsolve::SciMLLinearSolveAlgorithm)
    return LinearSolve.needs_concrete_A(linsolve)
end

update_A!(cache::LinearSolveJLCache, ::Nothing, reuse) = cache
function update_A!(cache::LinearSolveJLCache, A, reuse)
    return update_A!(cache, Utils.safe_getproperty(cache.linsolve, Val(:alg)), A, reuse)
end

function update_A!(cache::LinearSolveJLCache, alg, A, reuse)
    # Not a Factorization Algorithm so don't update `nfactors`
    set_lincache_A!(cache.lincache, A)
    return cache
end
function update_A!(cache::LinearSolveJLCache, ::LinearSolve.AbstractFactorization, A, reuse)
    reuse && return cache
    set_lincache_A!(cache.lincache, A)
    cache.stats.nfactors += 1
    return cache
end
function update_A!(
        cache::LinearSolveJLCache, alg::LinearSolve.DefaultLinearSolver, A, reuse)
    if alg ==
       LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES)
        # Force a reset of the cache. This is not properly handled in LinearSolve.jl
        set_lincache_A!(cache.lincache, A)
        return cache
    end
    reuse && return cache
    set_lincache_A!(cache.lincache, A)
    cache.stats.nfactors += 1
    return cache
end

function set_lincache_A!(lincache, new_A)
    if !LinearSolve.default_alias_A(lincache.alg, new_A, lincache.b) &&
       ArrayInterface.can_setindex(lincache.A)
        copyto!(lincache.A, new_A)
        lincache.A = lincache.A # important!! triggers special code in `setproperty!`
        return
    end
    lincache.A = new_A
    return
end

end
