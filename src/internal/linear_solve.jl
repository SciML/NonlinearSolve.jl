const LinearSolveFailureCode = isdefined(ReturnCode, :InternalLinearSolveFailure) ?
                               ReturnCode.InternalLinearSolveFailure : ReturnCode.Failure

"""
    LinearSolverCache(alg, linsolve, A, b, u; stats, kwargs...)

Construct a cache for solving linear systems of the form `A * u = b`. Following cases are
handled:

 1. `A` is Number, then we solve it with `u = b / A`
 2. `A` is `SMatrix`, then we solve it with `u = A \\ b` (using the defaults from base
    Julia)
 3. `A` is `Diagonal`, then we solve it with `u = b ./ A.diag`
 4. In all other cases, we use `alg` to solve the linear system using
    [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).

### Solving the System

```julia
(cache::LinearSolverCache)(;
    A = nothing, b = nothing, linu = nothing, du = nothing, p = nothing,
    weight = nothing, cachedata = nothing, reuse_A_if_factorization = false, kwargs...)
```

Returns the solution of the system `u` and stores the updated cache in `cache.lincache`.

#### Special Handling for Rank-deficient Matrix `A`

If we detect a failure in the linear solve (mostly due to using an algorithm that doesn't
support rank-deficient matrices), we emit a warning and attempt to solve the problem using
Pivoted QR factorization. This is quite efficient if there are only a few rank-deficient
that originate in the problem. However, if these are quite frequent for the main nonlinear
system, then it is recommended to use a different linear solver that supports rank-deficient
matrices.

#### Keyword Arguments

  - `reuse_A_if_factorization`: If `true`, then the factorization of `A` is reused if
    possible. This is useful when solving the same system with different `b` values.
    If the algorithm is an iterative solver, then we reset the internal linear solve cache.

One distinct feature of this compared to the cache from LinearSolve is that it respects the
aliasing arguments even after cache construction, i.e., if we passed in an `A` that `A` is
not mutated, we do this by copying over `A` to a preconstructed cache.
"""
@concrete mutable struct LinearSolverCache <: AbstractLinearSolverCache
    lincache
    linsolve
    additional_lincache::Any
    A
    b
    precs
    stats::NLStats
end

@inline __fix_strange_type_combination(A, b, u) = u
@inline function __fix_strange_type_combination(A, b, u::SArray)
    A isa SArray && b isa SArray && return u
    @warn "Solving Linear System A::$(typeof(A)) x::$(typeof(u)) = b::$(typeof(u)) is not \
           properly supported. Converting `x` to a mutable array. Check the return type \
           of the nonlinear function provided for optimal performance." maxlog=1
    return MArray(u)
end

@inline __set_lincache_u!(cache, u) = (cache.lincache.u = u)
@inline function __set_lincache_u!(cache, u::SArray)
    cache.lincache.u isa MArray && return __set_lincache_u!(cache, MArray(u))
    cache.lincache.u = u
end

function LinearSolverCache(alg, linsolve, A, b, u; stats, kwargs...)
    u_fixed = __fix_strange_type_combination(A, b, u)

    if (A isa Number && b isa Number) ||
       (linsolve === nothing && A isa SMatrix) ||
       (A isa Diagonal) ||
       (linsolve isa typeof(\))
        return LinearSolverCache(nothing, nothing, nothing, A, b, nothing, stats)
    end
    @bb u_ = copy(u_fixed)
    linprob = LinearProblem(A, b; u0 = u_, kwargs...)

    if __hasfield(alg, Val(:precs))
        precs = alg.precs
        Pl_, Pr_ = precs(A, nothing, u, ntuple(Returns(nothing), 6)...)
    else
        precs, Pl_, Pr_ = nothing, nothing, nothing
    end
    Pl, Pr = __wrapprecs(Pl_, Pr_, u)

    # Unalias here, we will later use these as caches
    lincache = init(linprob, linsolve; alias_A = false, alias_b = false, Pl, Pr)

    return LinearSolverCache(lincache, linsolve, nothing, nothing, nothing, precs, stats)
end

@kwdef @concrete struct LinearSolveResult
    u
    success::Bool = true
end

# Direct Linear Solve Case without Caching
function (cache::LinearSolverCache{Nothing})(;
        A = nothing, b = nothing, linu = nothing, kwargs...)
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1
    A === nothing || (cache.A = A)
    b === nothing || (cache.b = b)
    if A isa Diagonal
        _diag = _restructure(cache.b, cache.A.diag)
        @bb @. linu = cache.b / _diag
        res = linu
    else
        res = cache.A \ cache.b
    end
    return LinearSolveResult(; u = res)
end

# Use LinearSolve.jl
function (cache::LinearSolverCache)(;
        A = nothing, b = nothing, linu = nothing, du = nothing,
        p = nothing, weight = nothing, cachedata = nothing,
        reuse_A_if_factorization = false, verbose = true, kwargs...)
    cache.stats.nsolve += 1

    __update_A!(cache, A, reuse_A_if_factorization)
    b !== nothing && (cache.lincache.b = b)
    linu !== nothing && __set_lincache_u!(cache, linu)

    Plprev = cache.lincache.Pl
    Prprev = cache.lincache.Pr

    if cache.precs === nothing
        _Pl, _Pr = nothing, nothing
    else
        _Pl, _Pr = cache.precs(cache.lincache.A, du, linu, p, nothing,
            A !== nothing, Plprev, Prprev, cachedata)
    end

    if (_Pl !== nothing || _Pr !== nothing)
        Pl, Pr = __wrapprecs(_Pl, _Pr, linu)
        cache.lincache.Pl = Pl
        cache.lincache.Pr = Pr
    end

    linres = solve!(cache.lincache)
    cache.lincache = linres.cache
    # Unfortunately LinearSolve.jl doesn't have the most uniform ReturnCode handling
    if linres.retcode === ReturnCode.Failure
        structured_mat = ArrayInterface.isstructured(cache.lincache.A)
        is_gpuarray = ArrayInterface.device(cache.lincache.A) isa ArrayInterface.GPU
        if !(cache.linsolve isa QRFactorization{ColumnNorm}) &&
           !is_gpuarray &&
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
                    alias_A = false, alias_b = false, cache.lincache.Pl, cache.lincache.Pr)
            else
                cache.additional_lincache.A = A
                cache.additional_lincache.b = b
                cache.additional_lincache.Pl = cache.lincache.Pl
                cache.additional_lincache.Pr = cache.lincache.Pr
            end
            linres = solve!(cache.additional_lincache)
            cache.additional_lincache = linres.cache
            linres.retcode === ReturnCode.Failure &&
                return LinearSolveResult(; u = linres.u, success = false)
            return LinearSolveResult(; u = linres.u)
        elseif !(cache.linsolve isa QRFactorization{ColumnNorm})
            if verbose
                if structured_mat
                    @warn "Potential Rank Deficient Matrix Detected. But Matrix is \
                           Structured. Currently, we don't attempt to solve Rank Deficient \
                           Structured Matrices. Please open an issue at \
                           https://github.com/SciML/NonlinearSolve.jl"
                elseif is_gpuarray
                    @warn "Potential Rank Deficient Matrix Detected. But Matrix is on GPU. \
                           Currently, we don't attempt to solve Rank Deficient GPU \
                           Matrices. Please open an issue at \
                           https://github.com/SciML/NonlinearSolve.jl"
                end
            end
        end
        return LinearSolveResult(; u = linres.u, success = false)
    end

    return LinearSolveResult(; u = linres.u)
end

@inline __update_A!(cache::LinearSolverCache, ::Nothing, reuse) = cache
@inline function __update_A!(cache::LinearSolverCache, A, reuse)
    return __update_A!(cache, __getproperty(cache.lincache, Val(:alg)), A, reuse)
end
@inline function __update_A!(cache, alg, A, reuse)
    # Not a Factorization Algorithm so don't update `nfactors`
    __set_lincache_A(cache.lincache, A)
    return cache
end
@inline function __update_A!(cache, ::AbstractFactorization, A, reuse)
    reuse && return cache
    __set_lincache_A(cache.lincache, A)
    cache.stats.nfactors += 1
    return cache
end
@inline function __update_A!(cache, alg::DefaultLinearSolver, A, reuse)
    if alg == DefaultLinearSolver(DefaultAlgorithmChoice.KrylovJL_GMRES)
        # Force a reset of the cache. This is not properly handled in LinearSolve.jl
        __set_lincache_A(cache.lincache, A)
        return cache
    end
    reuse && return cache
    __set_lincache_A(cache.lincache, A)
    cache.stats.nfactors += 1
    return cache
end

function __set_lincache_A(lincache, new_A)
    if LinearSolve.default_alias_A(lincache.alg, new_A, lincache.b)
        lincache.A = new_A
    else
        if can_setindex(lincache.A)
            copyto!(lincache.A, new_A)
            lincache.A = lincache.A
        else
            lincache.A = new_A
        end
    end
end

function __wrapprecs(_Pl, _Pr, u)
    Pl = _Pl !== nothing ? _Pl : IdentityOperator(length(u))
    Pr = _Pr !== nothing ? _Pr : IdentityOperator(length(u))
    return Pl, Pr
end

@inline __needs_square_A(_, ::Number) = false
@inline __needs_square_A(::Nothing, ::Number) = false
@inline __needs_square_A(::Nothing, _) = false
@inline __needs_square_A(linsolve, _) = LinearSolve.needs_square_A(linsolve)
@inline __needs_square_A(::typeof(\), _) = false
@inline __needs_square_A(::typeof(\), ::Number) = false  # Ambiguity Fix
