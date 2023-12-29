"""
    Klement(; max_resets = 100, linsolve = NoLineSearch(), linesearch = nothing,
        precs = DEFAULT_PRECS, alpha = true, init_jacobian::Val = Val(:identity),
        autodiff = nothing)

An implementation of `Klement` with line search, preconditioning and customizable linear
solves. It is recommended to use `Broyden` for most problems over this.

## Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `100`.

  - `linsolve`: the [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) used for the
    linear solves within the Newton method. Defaults to `nothing`, which means it uses the
    LinearSolve.jl default algorithm choice. For more information on available algorithm
    choices, see the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `precs`: the choice of preconditioners for the linear solver. Defaults to using no
    preconditioners. For more information on specifying preconditioners for LinearSolve
    algorithms, consult the
    [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/).
  - `linesearch`: the line search algorithm to use. Defaults to [`NoLineSearch()`](@ref),
    which means that no line search is performed.  Algorithms from `LineSearches.jl` must be
    wrapped in `LineSearchesJL` before being supplied.
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
  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are types from ADTypes.jl. (Used if `init_jacobian = Val(:true_jacobian)`)
"""
function Klement(; max_resets::Int = 100, linsolve = nothing, alpha = true,
        linesearch = NoLineSearch(), precs = DEFAULT_PRECS,
        init_jacobian::Val{IJ} = Val(:identity), autodiff = nothing) where {IJ}
    if !(linesearch isa AbstractNonlinearSolveLineSearchAlgorithm)
        Base.depwarn("Passing in a `LineSearches.jl` algorithm directly is deprecated. \
                      Please use `LineSearchesJL` instead.", :Klement)
        linesearch = LineSearchesJL(; method = linesearch)
    end

    # TODO: Support alpha
    if IJ === :identity
        initialization = IdentityInitialization(DiagonalStructure())
    elseif IJ === :true_jacobian
        initialization = TrueJacobianInitialization(FullStructure(), autodiff)
    elseif IJ === :true_jacobian_diagonal
        initialization = TrueJacobianInitialization(DiagonalStructure(), autodiff)
    else
        throw(ArgumentError("`init_jacobian` must be one of `:identity`, `:true_jacobian`, \
                             or `:true_jacobian_diagonal`"))
    end

    return ApproximateJacobianSolveAlgorithm{true, :Klement}(linesearch,
        NewtonDescent(; linsolve, precs), KlementUpdateRule(), klement_reset_condition,
        UInt(max_resets), initialization)
end

# Essentially checks ill conditioned Jacobian
klement_reset_condition(J) = false
klement_reset_condition(J::Number) = iszero(J)
klement_reset_condition(J::AbstractMatrix) = cond(J) ≥ inv(eps(real(eltype(J)))^(1 // 2))
klement_reset_condition(J::AbstractVector) = any(iszero, J)
klement_reset_condition(J::Diagonal) = any(iszero, diag(J))

# Update Rule
@concrete struct KlementUpdateRule <: AbstractApproximateJacobianUpdateRule{false} end

@concrete mutable struct KlementUpdateRuleCache <:
                         AbstractApproximateJacobianUpdateRuleCache{false}
    Jdu
    J_cache
    J_cache_2
    Jdu_cache
    fu_cache
end

function SciMLBase.init(prob::AbstractNonlinearProblem, alg::KlementUpdateRule, J, fu, u,
        du, args...; kwargs...)
    @bb Jdu = similar(fu)
    if J isa Diagonal || J isa Number
        J_cache, J_cache_2, Jdu_cache = nothing, nothing, nothing
    else
        @bb J_cache = similar(J)
        @bb J_cache_2 = similar(J)
        @bb Jdu_cache = similar(Jdu)
    end
    @bb fu_cache = similar(fu)
    return KlementUpdateRuleCache(Jdu, J_cache, J_cache_2, Jdu_cache, fu_cache)
end

function SciMLBase.solve!(cache::KlementUpdateRuleCache, J::Number, fu, u, du)
    Jdu = J^2 * du^2
    J = J + ((fu - cache.fu_cache - J * du) / ifelse(iszero(Jdu), 1e-5, Jdu)) * du * J^2
    cache.fu_cache = fu
    return J
end

function SciMLBase.solve!(cache::KlementUpdateRuleCache, J_::Diagonal, fu, u, du)
    T = eltype(u)
    J = _restructure(u, diag(J_))
    @bb @. cache.Jdu = (J^2) * (du^2)
    @bb @. J += ((fu - cache.fu_cache - J * du) /
                 ifelse(iszero(cache.Jdu), T(1e-5), cache.Jdu)) * du * (J^2)
    @bb copyto!(cache.fu_cache, fu)
    return Diagonal(J)
end

function SciMLBase.solve!(cache::KlementUpdateRuleCache, J::AbstractMatrix, fu, u, du)
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
