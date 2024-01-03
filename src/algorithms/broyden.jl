"""
    Broyden(; max_resets::Int = 100, linesearch = nothing, reset_tolerance = nothing,
        init_jacobian::Val = Val(:identity), autodiff = nothing, alpha = nothing)

An implementation of `Broyden` with resetting and line search.

## Arguments

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
        problem.
"""
function Broyden(; max_resets = 100, linesearch = NoLineSearch(), reset_tolerance = nothing,
        init_jacobian::Val{IJ} = Val(:identity), autodiff = nothing, alpha = nothing,
        update_rule::Val{UR} = Val(:good_broyden)) where {IJ, UR}
    # TODO: Support alpha
    if IJ === :identity
        if UR === :diagonal
            initialization = IdentityInitialization(DiagonalStructure())
        else
            initialization = IdentityInitialization(FullStructure())
        end
    elseif IJ === :true_jacobian
        initialization = TrueJacobianInitialization(FullStructure(), autodiff)
    else
        throw(ArgumentError("`init_jacobian` must be one of `:identity` or \
                             `:true_jacobian`"))
    end

    update_rule = if UR === :good_broyden
        GoodBroydenUpdateRule()
    elseif UR === :bad_broyden
        BadBroydenUpdateRule()
    elseif UR === :diagonal
        GoodBroydenUpdateRule()
    else
        throw(ArgumentError("`update_rule` must be one of `:good_broyden`, `:bad_broyden`, \
                             or `:diagonal`"))
    end

    return ApproximateJacobianSolveAlgorithm{IJ === :true_jacobian, :Broyden}(; linesearch,
        descent = NewtonDescent(), update_rule, max_resets, initialization,
        reinit_rule = NoChangeInStateReset(; reset_tolerance))
end

# Essentially checks ill conditioned Jacobian
@kwdef @concrete struct NoChangeInStateReset <: AbstractResetCondition
    reset_tolerance = nothing
    check_du::Bool = true
    check_dfu::Bool = true
end

@concrete mutable struct NoChangeInStateResetCache
    dfu
    reset_tolerance
    check_du
    check_dfu
end

function SciMLBase.init(alg::NoChangeInStateReset, J, fu, u, du, args...; kwargs...)
    if alg.check_dfu
        @bb dfu = copy(fu)
    else
        dfu = fu
    end
    T = real(eltype(u))
    tol = alg.reset_tolerance === nothing ? sqrt(eps(T)) : T(alg.reset_tolerance)
    return NoChangeInStateResetCache(dfu, tol, alg.check_du, alg.check_dfu)
end

function SciMLBase.solve!(cache::NoChangeInStateResetCache, J, fu, u, du)
    cache.check_du && any(x -> abs(x) ≤ cache.reset_tolerance, du) && return true
    if cache.check_dfu
        @bb @. cache.dfu = fu - cache.dfu
        any(x -> abs(x) ≤ cache.reset_tolerance, cache.dfu) && return true
    end
    return false
end

# Broyden Update Rules
@concrete struct BadBroydenUpdateRule <: AbstractApproximateJacobianUpdateRule{true} end

@concrete struct GoodBroydenUpdateRule <: AbstractApproximateJacobianUpdateRule{true} end

@concrete mutable struct BroydenUpdateRuleCache{mode} <:
                         AbstractApproximateJacobianUpdateRuleCache{true}
    J⁻¹dfu
    dfu
    u_cache
    du_cache
    internalnorm
end

function SciMLBase.init(prob::AbstractNonlinearProblem,
        alg::Union{GoodBroydenUpdateRule, BadBroydenUpdateRule}, J, fu, u, du, args...;
        internalnorm::F = DEFAULT_NORM, kwargs...) where {F}
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
    mode = alg isa GoodBroydenUpdateRule ? :good : :bad
    return BroydenUpdateRuleCache{mode}(J⁻¹dfu, dfu, u_cache, du_cache, internalnorm)
end

function SciMLBase.solve!(cache::BroydenUpdateRuleCache{mode}, J⁻¹, fu, u, du) where {mode}
    T = eltype(u)
    @bb @. cache.dfu = fu - cache.dfu
    @bb cache.J⁻¹dfu = J⁻¹ × vec(cache.dfu)
    if mode === :good
        @bb cache.u_cache = transpose(J⁻¹) × vec(du)
        denom = dot(du, cache.J⁻¹dfu)
        rmul = transpose(_vec(cache.u_cache))
    else
        denom = cache.internalnorm(cache.dfu)^2
        rmul = transpose(_vec(cache.dfu))
    end
    @bb @. cache.du_cache = (du - cache.J⁻¹dfu) / ifelse(iszero(denom), T(1e-5), denom)
    @bb J⁻¹ += vec(cache.du_cache) × rmul
    @bb copyto!(cache.dfu, fu)
    return J⁻¹
end

function SciMLBase.solve!(cache::BroydenUpdateRuleCache{mode}, J⁻¹::Diagonal, fu, u,
        du) where {mode}
    T = eltype(u)
    @bb @. cache.dfu = fu - cache.dfu
    J⁻¹_diag = _restructure(cache.dfu, diag(J⁻¹))
    if mode === :good
        @bb @. cache.J⁻¹dfu = J⁻¹_diag * cache.dfu * du
        denom = sum(cache.J⁻¹dfu)
        @bb @. J⁻¹_diag += (du - J⁻¹_diag * cache.dfu) * du * J⁻¹_diag /
                           ifelse(iszero(denom), T(1e-5), denom)
    else
        denom = cache.internalnorm(cache.dfu)^2
        @bb @. J⁻¹_diag += (du - J⁻¹_diag * cache.dfu) * cache.dfu /
                           ifelse(iszero(denom), T(1e-5), denom)
    end
    @bb copyto!(cache.dfu, fu)
    return Diagonal(J⁻¹_diag)
end
