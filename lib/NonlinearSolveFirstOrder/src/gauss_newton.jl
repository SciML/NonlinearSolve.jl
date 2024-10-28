"""
    GaussNewton(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing, precs = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    )

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.
"""
function GaussNewton(;
        concrete_jac = nothing, linsolve = nothing, linesearch = missing, precs = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
)
    return GeneralizedFirstOrderAlgorithm(;
        linesearch,
        descent = NewtonDescent(; linsolve, precs),
        autodiff, vjp_autodiff, jvp_autodiff,
        concrete_jac,
        name = :GaussNewton
    )
end
