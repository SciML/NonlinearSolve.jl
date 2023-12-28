"""
    NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm

Don't perform a line search. Just return the initial step length of `1`.
"""
struct NoLineSearch <: AbstractNonlinearSolveLineSearchAlgorithm end

@concrete struct NoLineSearchCache
    α
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::NoLineSearch, f::F, fu, u,
        p, args...; kwargs...) where {F}
    return NoLineSearchCache(promote_type(eltype(fu), eltype(u))(true))
end

SciMLBase.solve!(cache::NoLineSearchCache, u, du) = cache.α

"""
    LineSearchesJL(; method = nothing, autodiff = nothing, alpha = true)

Wrapper over algorithms from
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl/). Allows automatic
construction of the objective functions for the line search algorithms utilizing automatic
differentiation for fast Vector Jacobian Products.

### Arguments

  - `method`: the line search algorithm to use. Defaults to `nothing`, which means that the
    step size is fixed to the value of `alpha`.
  - `autodiff`: the automatic differentiation backend to use for the line search. Defaults to
    `AutoFiniteDiff()`, which means that finite differencing is used to compute the VJP.
    `AutoZygote()` will be faster in most cases, but it requires `Zygote.jl` to be manually
    installed and loaded.
  - `alpha`: the initial step size to use. Defaults to `true` (which is equivalent to `1`).
"""
@kwdef @concrete struct LineSearchesJL <: AbstractNonlinearSolveLineSearchAlgorithm
    method = LineSearches.Static()
    autodiff = nothing
    α = true
end

Base.@deprecate_binding LineSearch LineSearchesJL true

# TODO: Implement
