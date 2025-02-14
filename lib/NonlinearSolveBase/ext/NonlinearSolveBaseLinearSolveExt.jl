module NonlinearSolveBaseLinearSolveExt

using ArrayInterface: ArrayInterface

using CommonSolve: CommonSolve, init, solve!
using LinearSolve: LinearSolve, QRFactorization, SciMLLinearSolveAlgorithm
using SciMLBase: ReturnCode, LinearProblem, LinearAliasSpecifier

using LinearAlgebra: ColumnNorm

using NonlinearSolveBase: NonlinearSolveBase, LinearSolveJLCache, LinearSolveResult, Utils

function (cache::LinearSolveJLCache)(;
        A = nothing, b = nothing, linu = nothing,
        reuse_A_if_factorization = false, verbose = true, kwargs...
)
    cache.stats.nsolve += 1

    update_A!(cache, A, reuse_A_if_factorization)
    b !== nothing && setproperty!(cache.lincache, :b, b)
    linu !== nothing && NonlinearSolveBase.set_lincache_u!(cache, linu)

    linres = solve!(cache.lincache)
    cache.lincache = linres.cache
    # Unfortunately LinearSolve.jl doesn't have the most uniform ReturnCode handling
    if linres.retcode === ReturnCode.Failure
        structured_mat = ArrayInterface.isstructured(cache.lincache.A)
        is_gpuarray = ArrayInterface.device(cache.lincache.A) isa ArrayInterface.GPU

        if !(cache.linsolve isa QRFactorization{ColumnNorm}) && !is_gpuarray &&
           !structured_mat
            if verbose
                @warn "Potential Rank Deficient Matrix Detected. Attempting to solve using \
                       Pivoted QR Factorization."
            end
            @assert (A !== nothing)&&(b !== nothing) "This case is not yet supported. \
                                                      Please open an issue at \
                                                      https://github.com/SciML/NonlinearSolve.jl"
            if cache.additional_lincache === nothing # First time
                linprob = LinearProblem(A, b; u0 = linres.u)
                cache.additional_lincache = init(
                    linprob, QRFactorization(ColumnNorm()); alias_u0 = false,
                    alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
                )
            else
                cache.additional_lincache.A = A
                cache.additional_lincache.b = b
                cache.additional_lincache.Pl = cache.lincache.Pl
                cache.additional_lincache.Pr = cache.lincache.Pr
            end
            linres = solve!(cache.additional_lincache)
            cache.additional_lincache = linres.cache
            linres.retcode === ReturnCode.Failure &&
                return LinearSolveResult(; linres.u, success = false)
            return LinearSolveResult(; linres.u)
        elseif !(cache.linsolve isa QRFactorization{ColumnNorm})
            if verbose
                if structured_mat || is_gpuarray
                    mat_desc = structured_mat ? "Structured" : "GPU"
                    @warn "Potential Rank Deficient Matrix Detected. But Matrix is \
                           $(mat_desc). Currently, we don't attempt to solve Rank Deficient \
                           $(mat_desc) Matrices. Please open an issue at \
                           https://github.com/SciML/NonlinearSolve.jl"
                end
            end
        end
        return LinearSolveResult(; linres.u, success = false)
    end

    return LinearSolveResult(; linres.u)
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
