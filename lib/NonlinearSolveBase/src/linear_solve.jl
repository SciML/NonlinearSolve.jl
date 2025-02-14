@kwdef @concrete struct LinearSolveResult
    u
    success::Bool = true
end

@concrete mutable struct LinearSolveJLCache <: AbstractLinearSolverCache
    lincache
    linsolve
    additional_lincache::Any
    stats::NLStats
end

@concrete mutable struct NativeJLLinearSolveCache <: AbstractLinearSolverCache
    A
    b
    stats::NLStats
end

"""
    construct_linear_solver(alg, linsolve, A, b, u; stats, kwargs...)

Construct a cache for solving linear systems of the form `A * u = b`. Following cases are
handled:

 1. `A` is Number, then we solve it with `u = b / A`
 2. `A` is `SMatrix`, then we solve it with `u = A \\ b` (using the defaults from base
    Julia) (unless a preconditioner is specified)
 3. If `linsolve` is `\\`, then we solve it with directly using `ldiv!(u, A, b)`
 4. In all other cases, we use `alg` to solve the linear system using
    [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)

### Solving the System

```julia
(cache::LinearSolverCache)(;
    A = nothing, b = nothing, linu = nothing, reuse_A_if_factorization = false, kwargs...
)
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
function construct_linear_solver(alg, linsolve, A, b, u; stats, kwargs...)
    if (A isa Number && b isa Number) || (A isa Diagonal)
        return NativeJLLinearSolveCache(A, b, stats)
    elseif linsolve isa typeof(\)
        return NativeJLLinearSolveCache(A, b, stats)
    elseif linsolve === nothing
        if (A isa SMatrix || A isa WrappedArray{<:Any, <:SMatrix})
            return NativeJLLinearSolveCache(A, b, stats)
        end
    end

    u_fixed = fix_incompatible_linsolve_arguments(A, b, u)
    @bb u_cache = copy(u_fixed)
    linprob = LinearProblem(A, b; u0 = u_cache, kwargs...)

    # unlias here, we will later use these as caches
    lincache = init(linprob, linsolve; alias = LinearAliasSpecifier(alias_A = false, alias_b = false))
    return LinearSolveJLCache(lincache, linsolve, nothing, stats)
end

function (cache::NativeJLLinearSolveCache)(;
        A = nothing, b = nothing, linu = nothing, kwargs...)
    cache.stats.nsolve += 1
    cache.stats.nfactors += 1

    A === nothing || (cache.A = A)
    b === nothing || (cache.b = b)

    if linu !== nothing && ArrayInterface.can_setindex(linu) &&
       applicable(ldiv!, linu, cache.A, cache.b) && applicable(ldiv!, cache.A, linu)
        ldiv!(linu, cache.A, cache.b)
        res = linu
    else
        res = cache.A \ cache.b
    end
    return LinearSolveResult(; u = res)
end

fix_incompatible_linsolve_arguments(A, b, u) = u
fix_incompatible_linsolve_arguments(::SArray, ::SArray, u::SArray) = u
function fix_incompatible_linsolve_arguments(A, b, u::SArray)
    (Core.Compiler.return_type(\, Tuple{typeof(A), typeof(b)}) <: typeof(u)) && return u
    @warn "Solving Linear System A::$(typeof(A)) x::$(typeof(u)) = b::$(typeof(u)) is not \
           properly supported. Converting `x` to a mutable array. Check the return type \
           of the nonlinear function provided for optimal performance." maxlog=1
    return MArray(u)
end

set_lincache_u!(cache, u) = setproperty!(cache.lincache, :u, u)
function set_lincache_u!(cache, u::SArray)
    cache.lincache.u isa MArray && return set_lincache_u!(cache, MArray(u))
    cache.lincache.u = u
end

function wrap_preconditioners(Pl, Pr, u)
    Pl = Pl === nothing ? IdentityOperator(length(u)) : Pl
    Pr = Pr === nothing ? IdentityOperator(length(u)) : Pr
    return Pl, Pr
end

# Traits. Core traits are expanded in LinearSolve extension
needs_square_A(::Any, ::Number) = false
needs_square_A(::Nothing, ::Number) = false
needs_square_A(::Nothing, ::Any) = false
needs_square_A(::typeof(\), ::Number) = false
needs_square_A(::typeof(\), ::Any) = false

needs_concrete_A(::Union{Nothing, Missing}) = false
needs_concrete_A(::typeof(\)) = true
