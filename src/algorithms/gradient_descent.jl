"""
    GradientDescent(; autodiff = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch())

An Implementation of Gradient Descent with Line Search.
"""
function GradientDescent(; autodiff = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch())
    descent = SteepestDescent()

    return GeneralizedFirstOrderRootFindingAlgorithm{false, :GradientDescent}(linesearch,
        descent, autodiff, nothing, nothing)
end
