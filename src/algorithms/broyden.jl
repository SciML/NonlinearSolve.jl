"""
    Broyden(; max_resets::Int = 100, linesearch = nothing, reset_tolerance = nothing,
        init_jacobian::Val = Val(:identity), autodiff = nothing, alpha = nothing)

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
        max_resets = 100, linesearch = nothing, reset_tolerance = nothing,
        init_jacobian = Val(:identity), autodiff = nothing, alpha = nothing,
        update_rule = Val(:good_broyden))
    initialization = broyden_init(init_jacobian, update_rule, autodiff, alpha)
    update_rule = broyden_update_rule(update_rule)
    return ApproximateJacobianSolveAlgorithm{
        init_jacobian isa Val{:true_jacobian}, :Broyden}(;
        linesearch, descent = NewtonDescent(), update_rule, max_resets, initialization,
        reinit_rule = NoChangeInStateReset(; reset_tolerance))
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

# Checks for no significant change for `nsteps`
"""
    NoChangeInStateReset(; nsteps::Int = 3, reset_tolerance = nothing,
        check_du::Bool = true, check_dfu::Bool = true)

Recommends a reset if the state or the function value has not changed significantly in
`nsteps` steps. This is used in [`Broyden`](@ref).

### Keyword Arguments

  - `nsteps`: the number of steps to check for no change. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `check_du`: whether to check the state. Defaults to `true`.
  - `check_dfu`: whether to check the function value. Defaults to `true`.
"""
@kwdef @concrete struct NoChangeInStateReset <: AbstractResetCondition
    nsteps::Int = 3
    reset_tolerance = nothing
    check_du::Bool = true
    check_dfu::Bool = true
end

@concrete mutable struct NoChangeInStateResetCache
    dfu
    reset_tolerance
    check_du
    check_dfu
    nsteps::Int
    steps_since_change_du::Int
    steps_since_change_dfu::Int
end

function reinit_cache!(cache::NoChangeInStateResetCache, args...; kwargs...)
    cache.steps_since_change_du = 0
    cache.steps_since_change_dfu = 0
end

function __internal_init(alg::NoChangeInStateReset, J, fu, u, du, args...; kwargs...)
    if alg.check_dfu
        @bb dfu = copy(fu)
    else
        dfu = fu
    end
    T = real(eltype(u))
    tol = alg.reset_tolerance === nothing ? eps(T)^(3 // 4) : T(alg.reset_tolerance)
    return NoChangeInStateResetCache(
        dfu, tol, alg.check_du, alg.check_dfu, alg.nsteps, 0, 0)
end

function __internal_solve!(cache::NoChangeInStateResetCache, J, fu, u, du)
    reset_tolerance = cache.reset_tolerance
    if cache.check_du
        if any(@closure(x->abs(x) ≤ reset_tolerance), du)
            cache.steps_since_change_du += 1
            if cache.steps_since_change_du ≥ cache.nsteps
                cache.steps_since_change_du = 0
                cache.steps_since_change_dfu = 0
                return true
            end
        else
            cache.steps_since_change_du = 0
            cache.steps_since_change_dfu = 0
        end
    end
    if cache.check_dfu
        @bb @. cache.dfu = fu - cache.dfu
        if any(@closure(x->abs(x) ≤ reset_tolerance), cache.dfu)
            cache.steps_since_change_dfu += 1
            if cache.steps_since_change_dfu ≥ cache.nsteps
                cache.steps_since_change_dfu = 0
                cache.steps_since_change_du = 0
                @bb copyto!(cache.dfu, fu)
                return true
            end
        else
            cache.steps_since_change_dfu = 0
            cache.steps_since_change_du = 0
        end
        @bb copyto!(cache.dfu, fu)
    end
    return false
end

# Broyden Update Rules
"""
    BadBroydenUpdateRule()

Broyden Update Rule corresponding to "bad broyden's method" [broyden1965class](@cite).
"""
@concrete struct BadBroydenUpdateRule <: AbstractApproximateJacobianUpdateRule{true} end

"""
    GoodBroydenUpdateRule()

Broyden Update Rule corresponding to "good broyden's method" [broyden1965class](@cite).
"""
@concrete struct GoodBroydenUpdateRule <: AbstractApproximateJacobianUpdateRule{true} end

@concrete mutable struct BroydenUpdateRuleCache{mode} <:
                         AbstractApproximateJacobianUpdateRuleCache{true}
    J⁻¹dfu
    dfu
    u_cache
    du_cache
    internalnorm
end

function __internal_init(prob::AbstractNonlinearProblem,
        alg::Union{GoodBroydenUpdateRule, BadBroydenUpdateRule}, J, fu, u,
        du, args...; internalnorm::F = L2_NORM, kwargs...) where {F}
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

function __internal_solve!(cache::BroydenUpdateRuleCache{mode}, J⁻¹, fu, u, du) where {mode}
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

function __internal_solve!(
        cache::BroydenUpdateRuleCache{mode}, J⁻¹::Diagonal, fu, u, du) where {mode}
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
