"""
    GradientDescent(; autodiff = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch())

### Keyword Arguments

  - `autodiff`: determines the backend used for the Gradient Calculation. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are types from ADTypes.jl. If `vjp` is supplied, we use that function.
  - `linesearch`: the line search algorithm to use. Defaults to [`NoLineSearch()`](@ref),
    which means that no line search is performed. Algorithms from `LineSearches.jl` must be
    wrapped in `LineSearchesJL` before being supplied.
"""
function GradientDescent(; autodiff = nothing,
        linesearch::AbstractNonlinearSolveLineSearchAlgorithm = NoLineSearch())
    descent = SteepestDescent()

    return GeneralizedFirstOrderRootFindingAlgorithm{false, :GradientDescent}(linesearch,
        descent, autodiff, nothing, nothing)
end
