"""
    Klement(; max_resets = 100, linsolve = NoLineSearch(), linesearch = nothing,
        precs = DEFAULT_PRECS, alpha = nothing, init_jacobian::Val = Val(:identity),
        autodiff = nothing)

An implementation of `Klement` [klement2014using](@citep) with line search, preconditioning
and customizable linear solves. It is recommended to use [`Broyden`](@ref) for most problems
over this.

### Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `100`.

  - `alpha`: If `init_jacobian` is set to `Val(:identity)`, then the initial Jacobian
    inverse is set to be `αI`. Defaults to `1`. Can be set to `nothing` which implies
    `α = max(norm(u), 1) / (2 * norm(fu))`.
  - `init_jacobian`: the method to use for initializing the jacobian. Defaults to
    `Val(:identity)`. Choices include:

      + `Val(:identity)`: Identity Matrix.
      + `Val(:true_jacobian)`: True Jacobian. Our tests suggest that this is not very
        stable. Instead using `Broyden` with `Val(:true_jacobian)` gives faster and more
        reliable convergence.
      + `Val(:true_jacobian_diagonal)`: Diagonal of True Jacobian. This is a good choice for
        differentiable problems.
"""
function Klement(; max_resets::Int = 100, linsolve = nothing, alpha = nothing,
        linesearch = NoLineSearch(), precs = DEFAULT_PRECS, autodiff = nothing,
        init_jacobian::Val{IJ} = Val(:identity)) where {IJ}
    if !(linesearch isa AbstractNonlinearSolveLineSearchAlgorithm)
        Base.depwarn("Passing in a `LineSearches.jl` algorithm directly is deprecated. \
                      Please use `LineSearchesJL` instead.", :Klement)
        linesearch = LineSearchesJL(; method = linesearch)
    end

    if IJ === :identity
        initialization = IdentityInitialization(alpha, DiagonalStructure())
    elseif IJ === :true_jacobian
        initialization = TrueJacobianInitialization(FullStructure(), autodiff)
    elseif IJ === :true_jacobian_diagonal
        initialization = TrueJacobianInitialization(DiagonalStructure(), autodiff)
    else
        throw(ArgumentError("`init_jacobian` must be one of `:identity`, `:true_jacobian`, \
                             or `:true_jacobian_diagonal`"))
    end

    CJ = IJ === :true_jacobian || IJ === :true_jacobian_diagonal

    return ApproximateJacobianSolveAlgorithm{CJ, :Klement}(; linesearch,
        descent = NewtonDescent(; linsolve, precs), update_rule = KlementUpdateRule(),
        reinit_rule = IllConditionedJacobianReset(), max_resets, initialization)
end

# Essentially checks ill conditioned Jacobian
"""
    IllConditionedJacobianReset()

Recommend resetting the Jacobian if the current jacobian is ill-conditioned. This is used
in [`Klement`](@ref).
"""
struct IllConditionedJacobianReset <: AbstractResetCondition end

@concrete struct IllConditionedJacobianResetCache
    condition_number_threshold
end

function __internal_init(alg::IllConditionedJacobianReset, J, fu, u, du, args...; kwargs...)
    condition_number_threshold = if J isa AbstractMatrix
        inv(eps(real(eltype(J)))^(1 // 2))
    else
        nothing
    end
    return IllConditionedJacobianResetCache(condition_number_threshold)
end

function __internal_solve!(cache::IllConditionedJacobianResetCache, J, fu, u, du)
    J isa Number && return iszero(J)
    J isa Diagonal && return any(iszero, diag(J))
    J isa AbstractMatrix && return cond(J) ≥ cache.condition_number_threshold
    J isa AbstractVector && return any(iszero, J)
    return false
end

# Update Rule
"""
    KlementUpdateRule()

Update rule for [`Klement`](@ref).
"""
@concrete struct KlementUpdateRule <: AbstractApproximateJacobianUpdateRule{false} end

@concrete mutable struct KlementUpdateRuleCache <:
                         AbstractApproximateJacobianUpdateRuleCache{false}
    Jdu
    J_cache
    J_cache_2
    Jdu_cache
    fu_cache
end

function __internal_init(prob::AbstractNonlinearProblem, alg::KlementUpdateRule, J, fu, u,
        du, args...; kwargs...)
    @bb Jdu = similar(fu)
    if J isa Diagonal || J isa Number
        J_cache, J_cache_2, Jdu_cache = nothing, nothing, nothing
    else
        @bb J_cache = similar(J)
        @bb J_cache_2 = similar(J)
        @bb Jdu_cache = similar(Jdu)
    end
    @bb fu_cache = copy(fu)
    return KlementUpdateRuleCache(Jdu, J_cache, J_cache_2, Jdu_cache, fu_cache)
end

function __internal_solve!(cache::KlementUpdateRuleCache, J::Number, fu, u, du)
    Jdu = J^2 * du^2
    J = J + ((fu - cache.fu_cache - J * du) / ifelse(iszero(Jdu), 1e-5, Jdu)) * du * J^2
    cache.fu_cache = fu
    return J
end

function __internal_solve!(cache::KlementUpdateRuleCache, J_::Diagonal, fu, u, du)
    T = eltype(u)
    J = _restructure(u, diag(J_))
    @bb @. cache.Jdu = (J^2) * (du^2)
    @bb @. J += ((fu - cache.fu_cache - J * du) /
                 ifelse(iszero(cache.Jdu), T(1e-5), cache.Jdu)) * du * (J^2)
    @bb copyto!(cache.fu_cache, fu)
    return Diagonal(vec(J))
end

function __internal_solve!(cache::KlementUpdateRuleCache, J::AbstractMatrix, fu, u, du)
    T = eltype(u)
    @bb @. cache.J_cache = J'^2
    @bb @. cache.Jdu = du^2
    @bb cache.Jdu_cache = cache.J_cache × vec(cache.Jdu)
    @bb cache.Jdu = J × vec(du)
    @bb @. cache.fu_cache = (fu - cache.fu_cache - cache.Jdu) /
                            ifelse(iszero(cache.Jdu_cache), T(1e-5), cache.Jdu_cache)
    @bb cache.J_cache = vec(cache.fu_cache) × transpose(_vec(du))
    @bb @. cache.J_cache *= J
    @bb cache.J_cache_2 = cache.J_cache × J
    @bb J .+= cache.J_cache_2
    @bb copyto!(cache.fu_cache, fu)
    return J
end
