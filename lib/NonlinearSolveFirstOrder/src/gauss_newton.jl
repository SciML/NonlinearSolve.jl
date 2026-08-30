"""
    GaussNewton(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    )

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

### Keyword Arguments

  - `jacobian_reuse`: a [`JacobianReuse`](@ref) policy, `true` to force the default policy
    on, or `false` to force it off. Defaults to `nothing`, which reuses the Jacobian when
    `length(u0) ≥ $(JACOBIAN_REUSE_SIZE_CUTOFF)`.
"""
function GaussNewton(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    )
    return GeneralizedFirstOrderAlgorithm(;
        linesearch,
        descent = NewtonDescent(; linsolve),
        autodiff, vjp_autodiff, jvp_autodiff,
        concrete_jac,
        jacobian_reuse,
        name = :GaussNewton
    )
end
