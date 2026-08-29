"""
    GaussNewton(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    )

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

The Jacobian and its factorization are adaptively reused across accepted steps when
`length(u0) ≥ $(JACOBIAN_REUSE_SIZE_CUTOFF)`; pass
`jacobian_reuse = false` to force exact Gauss-Newton steps, or a [`JacobianReuse`](@ref)
policy to configure the reuse.
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
