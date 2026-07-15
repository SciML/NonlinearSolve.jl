"""
    GaussNewton(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    )

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.

Set `jacobian_reuse = JacobianReuse()` (or `true`) to adaptively reuse the Jacobian and
factorization across accepted steps. Reuse is disabled by default.
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
