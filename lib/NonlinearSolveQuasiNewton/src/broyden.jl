"""
    Broyden(;
        max_resets::Int = 100, linesearch = nothing, reset_tolerance = nothing,
        init_jacobian::Val = Val(:identity), autodiff = nothing, alpha = nothing,
        update_rule = Val(:good_broyden)
    )

An implementation of `Broyden`'s Method [broyden1965class](@cite) with resetting and line
search.

### Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `100`.

  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `alpha`: If `init_jacobian` is set to `Val(:identity)`, then the initial Jacobian
    inverse is set to be `(αI)⁻¹`. Defaults to `nothing` which implies
    `α = max(norm(u), 1) / (2 * norm(fu))`.
  - `init_jacobian`: the method to use for initializing the jacobian. Defaults to
    `Val(:identity)`. Choices include:

      + `Val(:identity)`: Identity Matrix.
      + `Val(:true_jacobian)`: True Jacobian. This is a good choice for differentiable
        problems.
  - `update_rule`: Update Rule for the Jacobian. Choices are:

      + `Val(:good_broyden)`: Good Broyden's Update Rule
      + `Val(:bad_broyden)`: Bad Broyden's Update Rule
      + `Val(:diagonal)`: Only update the diagonal of the Jacobian. This algorithm may be
        useful for specific problems, but whether it will work may depend strongly on the
        problem
"""
function Broyden(;
        max_resets::Int = 100, linesearch = nothing, reset_tolerance = nothing,
        init_jacobian::Val = Val(:identity), autodiff = nothing, alpha = nothing,
        update_rule = Val(:good_broyden)
)
    return QuasiNewtonAlgorithm(;
        linesearch,
        descent = NewtonDescent(),
        update_rule = broyden_update_rule(update_rule),
        max_resets,
        initialization = broyden_init(init_jacobian, update_rule, autodiff, alpha),
        reinit_rule = NoChangeInStateReset(; reset_tolerance),
        concrete_jac = Val(init_jacobian isa Val{:true_jacobian}),
        name = :Broyden
    )
end

function broyden_init(::Val{:identity}, ::Val{:diagonal}, autodiff, alpha)
    return IdentityInitialization(alpha, DiagonalStructure())
end
function broyden_init(::Val{:identity}, ::Val, autodiff, alpha)
    IdentityInitialization(alpha, FullStructure())
end
function broyden_init(::Val{:true_jacobian}, ::Val, autodiff, alpha)
    return TrueJacobianInitialization(FullStructure(), autodiff)
end
function broyden_init(::Val{IJ}, ::Val{UR}, autodiff, alpha) where {IJ, UR}
    error("Unknown combination of `init_jacobian = Val($(Meta.quot(IJ)))` and \
           `update_rule = Val($(Meta.quot(UR)))`. Please choose a valid combination.")
end

broyden_update_rule(::Val{:good_broyden}) = GoodBroydenUpdateRule()
broyden_update_rule(::Val{:bad_broyden}) = BadBroydenUpdateRule()
broyden_update_rule(::Val{:diagonal}) = GoodBroydenUpdateRule()
function broyden_update_rule(::Val{UR}) where {UR}
    error("Unknown update rule `update_rule = Val($(Meta.quot(UR)))`. Please choose a \
           valid update rule.")
end

"""
    BadBroydenUpdateRule()

Broyden Update Rule corresponding to "bad broyden's method" [broyden1965class](@cite).
"""
struct BadBroydenUpdateRule <: AbstractApproximateJacobianUpdateRule end

"""
    GoodBroydenUpdateRule()

Broyden Update Rule corresponding to "good broyden's method" [broyden1965class](@cite).
"""
struct GoodBroydenUpdateRule <: AbstractApproximateJacobianUpdateRule end

for rType in (:BadBroydenUpdateRule, :GoodBroydenUpdateRule)
    @eval function Base.getproperty(rule::$(rType), sym::Symbol)
        sym == :store_inverse_jacobian && return Val(true)
        return getfield(rule, sym)
    end
end

function InternalAPI.init(
        prob::AbstractNonlinearProblem,
        alg::Union{BadBroydenUpdateRule, GoodBroydenUpdateRule},
        J, fu, u, du, args...; internalnorm::F = L2_NORM, kwargs...
) where {F}
    @bb J⁻¹dfu = similar(u)
    @bb dfu = copy(fu)
    if alg isa GoodBroydenUpdateRule || J isa Diagonal
        @bb u_cache = similar(u)
    else
        u_cache = nothing
    end
    if J isa Diagonal
        du_cache = nothing
    else
        @bb du_cache = similar(du)
    end
    return BroydenUpdateRuleCache(J⁻¹dfu, dfu, u_cache, du_cache, internalnorm, alg)
end

@concrete mutable struct BroydenUpdateRuleCache <:
                         AbstractApproximateJacobianUpdateRuleCache
    J⁻¹dfu
    dfu
    u_cache
    du_cache
    internalnorm
    rule <: Union{BadBroydenUpdateRule, GoodBroydenUpdateRule}
end

function InternalAPI.solve!(
        cache::BroydenUpdateRuleCache, J⁻¹, fu, u, du; kwargs...
)
    T = eltype(u)
    @bb @. cache.dfu = fu - cache.dfu
    @bb cache.J⁻¹dfu = J⁻¹ × vec(cache.dfu)
    if cache.rule isa GoodBroydenUpdateRule
        @bb cache.u_cache = transpose(J⁻¹) × vec(du)
        denom = dot(du, cache.J⁻¹dfu)
        rmul = transpose(Utils.safe_vec(cache.u_cache))
    else
        denom = cache.internalnorm(cache.dfu)^2
        rmul = transpose(Utils.safe_vec(cache.dfu))
    end
    @bb @. cache.du_cache = (du - cache.J⁻¹dfu) / ifelse(iszero(denom), T(1e-5), denom)
    @bb J⁻¹ += vec(cache.du_cache) × rmul
    @bb copyto!(cache.dfu, fu)
    return J⁻¹
end

function InternalAPI.solve!(
        cache::BroydenUpdateRuleCache, J⁻¹::Diagonal, fu, u, du; kwargs...
)
    T = eltype(u)
    @bb @. cache.dfu = fu - cache.dfu
    J⁻¹_diag = Utils.restructure(cache.dfu, diag(J⁻¹))
    if cache.rule isa GoodBroydenUpdateRule
        @bb @. J⁻¹_diag = J⁻¹_diag * cache.dfu * du
        denom = sum(J⁻¹_diag)
        @bb @. J⁻¹_diag = J⁻¹_diag +
                          (du - J⁻¹_diag * cache.dfu) * du * J⁻¹_diag /
                          ifelse(iszero(denom), T(1e-5), denom)
    else
        denom = cache.internalnorm(cache.dfu)^2
        @bb @. J⁻¹_diag = J⁻¹_diag +
                          (du - J⁻¹_diag * cache.dfu) * cache.dfu /
                          ifelse(iszero(denom), T(1e-5), denom)
    end
    @bb copyto!(cache.dfu, fu)
    return Diagonal(vec(J⁻¹_diag))
end
