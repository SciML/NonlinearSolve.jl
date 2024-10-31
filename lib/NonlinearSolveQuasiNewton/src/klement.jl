"""
    Klement(;
        max_resets = 100, linsolve = nothing, linesearch = nothing,
        alpha = nothing, init_jacobian::Val = Val(:identity),
        autodiff = nothing
    )

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
function Klement(;
        max_resets = 100, linsolve = nothing, linesearch = nothing,
        alpha = nothing, init_jacobian::Val = Val(:identity),
        autodiff = nothing
)
    concrete_jac = Val(init_jacobian isa Val{:true_jacobian} ||
                       init_jacobian isa Val{:true_jacobian_diagonal})
    return QuasiNewtonAlgorithm(;
        linesearch,
        descent = NewtonDescent(; linsolve),
        update_rule = KlementUpdateRule(),
        reinit_rule = IllConditionedJacobianReset(),
        max_resets,
        initialization = klement_init(init_jacobian, autodiff, alpha),
        concrete_jac,
        name = :Klement
    )
end

function klement_init(::Val{:identity}, autodiff, alpha)
    return IdentityInitialization(alpha, DiagonalStructure())
end
function klement_init(::Val{:true_jacobian}, autodiff, alpha)
    return TrueJacobianInitialization(FullStructure(), autodiff)
end
function klement_init(::Val{:true_jacobian_diagonal}, autodiff, alpha)
    return TrueJacobianInitialization(DiagonalStructure(), autodiff)
end
function klement_init(::Val{IJ}, autodiff, alpha) where {IJ}
    error("Unknown `init_jacobian = Val($(Meta.quot(IJ)))`. Please choose a valid \
           `init_jacobian`.")
end

"""
    KlementUpdateRule()

Update rule for [`Klement`](@ref).
"""
struct KlementUpdateRule <: AbstractApproximateJacobianUpdateRule end

function Base.getproperty(rule::KlementUpdateRule, sym::Symbol)
    sym == :store_inverse_jacobian && return Val(false)
    return getfield(rule, sym)
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem, alg::KlementUpdateRule,
        J, fu, u, du, args...; kwargs...
)
    @bb Jdu = similar(fu)
    if J isa Diagonal || J isa Number
        J_cache, J_cache_2, Jdu_cache = nothing, nothing, nothing
    else
        @bb J_cache = similar(J)
        @bb J_cache_2 = similar(J)
        @bb Jdu_cache = similar(Jdu)
    end
    @bb fu_cache = copy(fu)
    return KlementUpdateRuleCache(Jdu, J_cache, J_cache_2, Jdu_cache, fu_cache, alg)
end

@concrete mutable struct KlementUpdateRuleCache <:
                         AbstractApproximateJacobianUpdateRuleCache
    Jdu
    J_cache
    J_cache_2
    Jdu_cache
    fu_cache
    rule <: KlementUpdateRule
end

function InternalAPI.solve!(
        cache::KlementUpdateRuleCache, J::Number, fu, u, du; kwargs...
)
    Jdu = J^2 * du^2
    J = J + ((fu - cache.fu_cache - J * du) / ifelse(iszero(Jdu), 1e-5, Jdu)) * du * J^2
    cache.fu_cache = fu
    return J
end

function InternalAPI.solve!(
        cache::KlementUpdateRuleCache, J::Diagonal, fu, u, du; kwargs...
)
    T = eltype(u)
    J = Utils.restructure(u, diag(J))
    @bb @. cache.Jdu = (J^2) * (du^2)
    @bb @. J += ((fu - cache.fu_cache - J * du) /
                 ifelse(iszero(cache.Jdu), T(1e-5), cache.Jdu)) * du * (J^2)
    @bb copyto!(cache.fu_cache, fu)
    return Diagonal(vec(J))
end

function InternalAPI.solve!(
        cache::KlementUpdateRuleCache, J::AbstractMatrix, fu, u, du; kwargs...
)
    T = eltype(u)
    @bb @. cache.J_cache = J'^2
    @bb @. cache.Jdu = du^2
    @bb cache.Jdu_cache = cache.J_cache × vec(cache.Jdu)
    @bb cache.Jdu = J × vec(du)
    @bb @. cache.fu_cache = (fu - cache.fu_cache - cache.Jdu) /
                            ifelse(iszero(cache.Jdu_cache), T(1e-5), cache.Jdu_cache)
    @bb cache.J_cache = vec(cache.fu_cache) × transpose(Utils.safe_vec(du))
    @bb @. cache.J_cache *= J
    @bb cache.J_cache_2 = cache.J_cache × J
    @bb J .+= cache.J_cache_2
    @bb copyto!(cache.fu_cache, fu)
    return J
end
