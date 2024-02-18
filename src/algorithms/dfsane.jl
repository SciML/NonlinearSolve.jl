"""
    DFSane(; σ_min = 1 // 10^10, σ_max = 1e10, σ_1 = 1, M::Int = 10, γ = 1 // 10^4,
        τ_min = 1 // 10, τ_max = 1 // 2, n_exp::Int = 2, max_inner_iterations::Int = 100,
        η_strategy = (fn_1, n, x_n, f_n) -> fn_1 / n^2)

A low-overhead and allocation-free implementation of the df-sane method for solving
large-scale nonlinear systems of equations. For in depth information about all the
parameters and the algorithm, see [la2006spectral](@citet).

### Keyword Arguments

  - `σ_min`: the minimum value of the spectral coefficient `σₙ` which is related to the step
    size in the algorithm. Defaults to `1e-10`.
  - `σ_max`: the maximum value of the spectral coefficient `σₙ` which is related to the step
    size in the algorithm. Defaults to `1e10`.

For other keyword arguments, see [`RobustNonMonotoneLineSearch`](@ref).
"""
function DFSane(; σ_min = 1 // 10^10, σ_max = 1e10, σ_1 = 1, M::Int = 10, γ = 1 // 10^4,
        τ_min = 1 // 10, τ_max = 1 // 2, n_exp::Int = 2, max_inner_iterations::Int = 100,
        η_strategy::ETA = (fn_1, n, x_n, f_n) -> fn_1 / n^2) where {ETA}
    linesearch = RobustNonMonotoneLineSearch(;
        gamma = γ, sigma_1 = σ_1, M, tau_min = τ_min, tau_max = τ_max,
        n_exp, η_strategy, maxiters = max_inner_iterations)
    return GeneralizedDFSane{:DFSane}(linesearch, σ_min, σ_max, nothing)
end
