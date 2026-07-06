module NonlinearSolveBaseLinearSolveExt

using ArrayInterface: ArrayInterface

using CommonSolve: CommonSolve, init, solve!
using LinearSolve: LinearSolve, QRFactorization, SciMLLinearSolveAlgorithm
using SciMLBase: SciMLBase, ReturnCode, LinearProblem, LinearAliasSpecifier
using SciMLLogging: @SciMLMessage

using LinearAlgebra: ColumnNorm, Symmetric

using NonlinearSolveBase: NonlinearSolveBase, LinearSolveJLCache, LinearSolveResult, Utils, NonlinearVerbosity, InternalAPI, LinearSolveParameters

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
function NonlinearSolveBase.default_spd_linsolve(::Symmetric{<:Real})
    return LinearSolve.CholeskyFactorization()
end

function NonlinearSolveBase.needs_concrete_A(linsolve::SciMLLinearSolveAlgorithm)
    return LinearSolve.needs_concrete_A(linsolve)
end

update_A!(cache::LinearSolveJLCache, ::Nothing, reuse) = cache
function update_A!(cache::LinearSolveJLCache, A, reuse)
    # Dispatch on the *resolved* algorithm stored in the LinearSolve cache.
    # `cache.linsolve` is the user-passed object (e.g. `KLUFactorization()`), which has
    # no `alg` field, so the old `safe_getproperty(cache.linsolve, Val(:alg))` returned
    # `missing` and always fell through to the non-factorization method below. That
    # method re-sets `A` unconditionally, marking the LinearSolve cache fresh, so
    # factorization algorithms refactorized on every call even when the caller asked
    # for reuse via `reuse_A_if_factorization` (and `nfactors` was never incremented).
    return update_A!(cache, cache.lincache.alg, A, reuse)
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
        cache::LinearSolveJLCache, alg::LinearSolve.DefaultLinearSolver, A, reuse
    )
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

function LinearSolve.update_tolerances!(cache::LinearSolveJLCache; kwargs...)
    return LinearSolve.update_tolerances!(cache.lincache; kwargs...)
end

function InternalAPI.reinit!(cache::LinearSolveJLCache, args...; u = missing, p = missing, kwargs...)
    # `u`/`p` left as `missing` mean "unchanged" — preserve the current values rather than
    # overwriting them with `missing`. Otherwise a `reinit!` that only updates `u` (the
    # usual case in a continuation loop, parameters fixed) would rebuild the parameters as
    # `LinearSolveParameters(u_fixed, missing)`, whose `Missing` p-type mismatches the
    # concretely-typed `p` (e.g. `NullParameters`) the LinearSolve cache was built with,
    # throwing a `setfield!` type error.
    cur = cache.lincache.p
    u_fixed = if u !== missing
        u_vec = Utils.safe_vec(u)
        (; A, b) = cache.lincache
        NonlinearSolveBase.fix_incompatible_linsolve_arguments(A, b, u_vec)
    else
        cur.u
    end
    p_new = p === missing ? cur.p : p
    return SciMLBase.reinit!(cache.lincache; p = LinearSolveParameters(u_fixed, p_new))
end

end
