"""
    LimitedMemoryBroyden(;
        max_resets::Int = 3, linesearch = nothing, threshold::Val = Val(10),
        reset_tolerance = nothing, alpha = nothing
    )

An implementation of `LimitedMemoryBroyden` [ziani2008autoadaptative](@cite) with resetting
and line search.

### Keyword Arguments

  - `max_resets`: the maximum number of resets to perform. Defaults to `3`.
  - `reset_tolerance`: the tolerance for the reset check. Defaults to
    `sqrt(eps(real(eltype(u))))`.
  - `threshold`: the number of vectors to store in the low rank approximation. Defaults
    to `Val(10)`.
  - `alpha`: The initial Jacobian inverse is set to be `(αI)⁻¹`. Defaults to `nothing`
    which implies `α = max(norm(u), 1) / (2 * norm(fu))`.
"""
function LimitedMemoryBroyden(;
        max_resets::Int = 3, linesearch = nothing, threshold::Union{Val, Int} = Val(10),
        reset_tolerance = nothing, alpha = nothing
)
    threshold isa Int && (threshold = Val(threshold))
    return QuasiNewtonAlgorithm(;
        linesearch,
        descent = NewtonDescent(),
        update_rule = GoodBroydenUpdateRule(),
        max_resets,
        initialization = BroydenLowRankInitialization(alpha, threshold),
        reinit_rule = NoChangeInStateReset(; reset_tolerance),
        name = :LimitedMemoryBroyden,
        concrete_jac = Val(false)
    )
end
