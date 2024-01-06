"""
    GradientDescent(; autodiff = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch())

An Implementation of Gradient Descent with Line Search.
"""
function GradientDescent(; autodiff = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch())
    return GeneralizedFirstOrderAlgorithm(; concrete_jac = false, name = :GradientDescent,
        linesearch, descent = SteepestDescent(), jacobian_ad = autodiff,
        forward_ad = nothing, reverse_ad = nothing)
end
