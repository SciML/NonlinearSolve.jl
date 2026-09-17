module NonlinearSolveBaseLinearSolveExt

using ArrayInterface: ArrayInterface

using CommonSolve: CommonSolve, init, solve!
using LinearSolve: LinearSolve, QRFactorization, SciMLLinearSolveAlgorithm
using SciMLBase: SciMLBase, ReturnCode, LinearProblem, LinearAliasSpecifier
using SciMLLogging: @SciMLMessage

using LinearAlgebra: ColumnNorm, Symmetric

using NonlinearSolveBase: NonlinearSolveBase, LinearSolveJLCache, LinearSolveResult, Utils, NonlinearVerbosity, InternalAPI, LinearSolveParameters

Utils.is_extension_loaded(::Val{:LinearSolve}) = true

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

# `LUFactorization` is the only LinearSolve algorithm whose `solve!` keys on
# `cache.alias_A` for an in-place (`lu!`) refactorization path; its gate is
# "mutable dense non-GPU", which plain `Matrix` mirrors conservatively. Other dense
# LU variants (Generic/MKL/OpenBLAS/RecursiveFactorization) already factorize
# `cache.A` in place unconditionally, so they gain nothing from aliasing.
function NonlinearSolveBase.alias_A_for_refactorization(
        ::LinearSolve.LUFactorization, ::Matrix
    )
    return true
end
# `linsolve === nothing` resolves to `DefaultLinearSolver`, whose dense choices funnel
# into the same alias-gated LU `solve!` body (measured: 21,513 → 560 B per
# refactorization on a 51×51 `Matrix`). Its singular-LU → QR safety fallback stays
# intact under aliasing: `_copy_A_for_safety` keeps a private cached `A_backup`
# reused across refactorizations, so the backup does not reintroduce a per-step
# allocation. The cache is inited on an owned copy either way, so aliasing is at
# worst neutral for default choices without an in-place path.
NonlinearSolveBase.alias_A_for_refactorization(::Nothing, ::Matrix) = true
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

# `MoreTrustRegionDescent(linsolve = LHLFactorization())` on a dense `Matrix`
# Jacobian: one Hessenberg reduction of `BᵀB = D⁻¹JᵀJD⁻¹` per Jacobian, then every
# damping-parameter trial `(BᵀB + λI) y = b` is an O(n²) `lhl_shift!`/`lhl_ldiv!`
# instead of an O(n³) refactorization — see `more_trust_region.jl`. `lhl`, `lhl!`,
# `lhl_shift!`, `lhl_ldiv!`, `lhl_isreduced` are LinearSolve's own re-exports of
# LHLFactorization.jl (a hard dependency of LinearSolve), so the extension needs
# no new package; the type itself did not exist before LinearSolve 5.16, hence the
# `isdefined` gate.
if isdefined(LinearSolve, :LHLFactorization) && isdefined(LinearSolve, :lhl)
    NonlinearSolveBase.uses_lhl(::LinearSolve.LHLFactorization) = true

    function NonlinearSolveBase._more_lhl_workspace(
            J_::Matrix{T}, dtd, linsolve::LinearSolve.LHLFactorization, stats
        ) where {T}
        n = size(J_, 2)
        BtB = Matrix{T}(undef, n, n)
        s = Vector{T}(undef, n)
        NonlinearSolveBase._more_lhl_gram!(BtB, s, J_, dtd)
        thread = linsolve isa LinearSolve.LHLFactorization{true}
        ws = LinearSolve.lhl(BtB; balance = linsolve.balance, thread)
        stats.nfactors += 1
        return NonlinearSolveBase._MoreLHLWorkspace(
            ws, BtB, s, Vector{T}(undef, n), Vector{T}(undef, n),
            Vector{T}(undef, n), Vector{T}(undef, n), stats, linsolve.refine,
            linsolve.balance, thread
        )
    end

    # `dtd` must already reflect the current scaling update — the reduced matrix
    # `BᵀB = D⁻¹JᵀJD⁻¹` depends on it, unlike the `lmpar` path's scale-free `R`
    function NonlinearSolveBase._more_lhl_refactor!(
            w::NonlinearSolveBase._MoreLHLWorkspace, J_, dtd
        )
        NonlinearSolveBase._more_lhl_gram!(w.BtB, w.s, J_, dtd)
        LinearSolve.lhl!(w.ws, w.BtB; balance = w.balance, thread = w.thread)
        w.stats.nfactors += 1
        return w
    end

    # The O(n²) per-λ work: factor `BᵀB + λI` on the stored reduction. A singular
    # shifted Hessenberg (`info != 0`, e.g. the `λ = 0` Gauss–Newton solve on a
    # rank-deficient Jacobian) is reported as a solve failure, like `lmpar`'s
    # rank check declining the undamped step
    function NonlinearSolveBase._more_lhl_shift!(
            w::NonlinearSolveBase._MoreLHLWorkspace{T}, λ
        ) where {T}
        LinearSolve.lhl_isreduced(w.ws) || return false
        LinearSolve.lhl_shift!(w.ws, λ, one(T))
        w.stats.nsolve += 1
        return w.ws.info == 0
    end

    NonlinearSolveBase._more_lhl_ldiv!(w::NonlinearSolveBase._MoreLHLWorkspace, x) =
        LinearSolve.lhl_ldiv!(x, w.ws)
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
