import LinearSolve: AbstractFactorization, DefaultAlgorithmChoice, DefaultLinearSolver

"""
    LinearSolverCache(alg, linsolve, A, b, u; kwargs...)

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
(cache::LinearSolverCache)(; A = nothing, b = nothing, linu = nothing,
    du = nothing, p = nothing, weight = nothing, cachedata = nothing,
    reuse_A_if_factorization = false, kwargs...)
```

Returns the solution of the system `u` and stores the updated cache in `cache.lincache`.

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
    A
    b
    precs
    nsolve::Int
    nfactors::Int
end

# FIXME: Do we need to reinit the precs?
function reinit_cache!(cache::LinearSolverCache, args...; kwargs...)
    cache.nsolve = 0
    cache.nfactors = 0
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

function LinearSolverCache(alg, linsolve, A, b, u; kwargs...)
    u_fixed = __fix_strange_type_combination(A, b, u)

    if (A isa Number && b isa Number) || (linsolve === nothing && A isa SMatrix) ||
       (A isa Diagonal) || (linsolve isa typeof(\))
        return LinearSolverCache(nothing, nothing, A, b, nothing, 0, 0)
    end
    @bb u_ = copy(u_fixed)
    linprob = LinearProblem(A, b; u0 = u_, kwargs...)

    weight = __init_ones(u_fixed)
    if __hasfield(alg, Val(:precs))
        precs = alg.precs
        Pl_, Pr_ = precs(A, nothing, u, nothing, nothing, nothing, nothing, nothing,
            nothing)
    else
        precs, Pl_, Pr_ = nothing, nothing, nothing
    end
    Pl, Pr = __wrapprecs(Pl_, Pr_, weight)

    # Unalias here, we will later use these as caches
    lincache = init(linprob, linsolve; alias_A = false, alias_b = false, Pl, Pr)

    return LinearSolverCache(lincache, linsolve, nothing, nothing, precs, 0, 0)
end

# Direct Linear Solve Case without Caching
function (cache::LinearSolverCache{Nothing})(; A = nothing, b = nothing, linu = nothing,
        kwargs...)
    cache.nsolve += 1
    cache.nfactors += 1
    A === nothing || (cache.A = A)
    b === nothing || (cache.b = b)
    if A isa Diagonal
        _diag = _restructure(cache.b, cache.A.diag)
        @bb @. linu = cache.b / _diag
        res = linu
    else
        res = cache.A \ cache.b
    end
    return res
end
# Use LinearSolve.jl
function (cache::LinearSolverCache)(; A = nothing, b = nothing, linu = nothing,
        du = nothing, p = nothing, weight = nothing, cachedata = nothing,
        reuse_A_if_factorization = false, kwargs...)
    cache.nsolve += 1

    __update_A!(cache, A, reuse_A_if_factorization)
    b !== nothing && (cache.lincache.b = b)
    linu !== nothing && __set_lincache_u!(cache, linu)

    Plprev = cache.lincache.Pl isa ComposePreconditioner ? cache.lincache.Pl.outer :
             cache.lincache.Pl
    Prprev = cache.lincache.Pr isa ComposePreconditioner ? cache.lincache.Pr.outer :
             cache.lincache.Pr

    if cache.precs === nothing
        _Pl, _Pr = nothing, nothing
    else
        _Pl, _Pr = cache.precs(cache.lincache.A, du, linu, p, nothing, A !== nothing,
            Plprev, Prprev, cachedata)
    end

    if (_Pl !== nothing || _Pr !== nothing)
        _weight = weight === nothing ?
                  (cache.lincache.Pr isa Diagonal ? cache.lincache.Pr.diag :
                   cache.lincache.Pr.inner.diag) : weight
        Pl, Pr = __wrapprecs(_Pl, _Pr, _weight)
        cache.lincache.Pl = Pl
        cache.lincache.Pr = Pr
    end

    linres = solve!(cache.lincache)
    cache.lincache = linres.cache

    return linres.u
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
    cache.nfactors += 1
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
    cache.nfactors += 1
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

@inline __needs_square_A(_, ::Number) = false
@inline __needs_square_A(::Nothing, ::Number) = false
@inline __needs_square_A(::Nothing, _) = false
@inline __needs_square_A(linsolve, _) = LinearSolve.needs_square_A(linsolve)
@inline __needs_square_A(::typeof(\), _) = false
@inline __needs_square_A(::typeof(\), ::Number) = false  # Ambiguity Fix
