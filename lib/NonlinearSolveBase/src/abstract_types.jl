module InternalAPI

function init end
function solve! end
function reinit! end

end

abstract type AbstractDescentDirection end

supports_line_search(::AbstractDescentDirection) = false
supports_trust_region(::AbstractDescentDirection) = false

function get_linear_solver(alg::AbstractDescentDirection)
    return Utils.safe_getproperty(alg, Val(:linsolve))
end

abstract type AbstractDescentCache end

SciMLBase.get_du(cache::AbstractDescentCache) = cache.δu
SciMLBase.get_du(cache::AbstractDescentCache, ::Val{1}) = SciMLBase.get_du(cache)
SciMLBase.get_du(cache::AbstractDescentCache, ::Val{N}) where {N} = cache.δus[N - 1]
set_du!(cache::AbstractDescentCache, δu) = (cache.δu = δu)
set_du!(cache::AbstractDescentCache, δu, ::Val{1}) = set_du!(cache, δu)
set_du!(cache::AbstractDescentCache, δu, ::Val{N}) where {N} = (cache.δus[N - 1] = δu)

function last_step_accepted(cache::AbstractDescentCache)
    hasfield(typeof(cache), :last_step_accepted) && return cache.last_step_accepted
    return true
end

abstract type AbstractNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end

get_name(alg::AbstractNonlinearSolveAlgorithm) = Utils.safe_getproperty(alg, Val(:name))

"""
    concrete_jac(alg::AbstractNonlinearSolveAlgorithm)::Bool

Whether the algorithm uses a concrete Jacobian.
"""
function concrete_jac(alg::AbstractNonlinearSolveAlgorithm)
    return concrete_jac(Utils.safe_getproperty(alg, Val(:concrete_jac)))
end
concrete_jac(::Missing) = false
concrete_jac(::Nothing) = false
concrete_jac(v::Bool) = v
concrete_jac(::Val{false}) = false
concrete_jac(::Val{true}) = true

abstract type AbstractNonlinearSolveCache end

"""
    AbstractLinearSolverCache

Abstract Type for all Linear Solvers used in NonlinearSolve. Subtypes of these are
meant to be constructured via [`construct_linear_solver`](@ref).
"""
abstract type AbstractLinearSolverCache end

"""
    AbstractJacobianCache

Abstract Type for all Jacobian Caches used in NonlinearSolve. Subtypes of these are
meant to be constructured via [`construct_jacobian_cache`](@ref).
"""
abstract type AbstractJacobianCache end
