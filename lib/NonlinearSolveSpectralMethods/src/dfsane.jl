"""
    DFSane(;
        sigma_min = 1 // 10^10, sigma_max = 1e10, sigma_1 = 1, M::Int = 10,
        gamma = 1 // 10^4, tau_min = 1 // 10, tau_max = 1 // 2, n_exp::Int = 2,
        max_inner_iterations::Int = 100, eta_strategy = (fn_1, n, x_n, f_n) -> fn_1 / n^2
    )

A low-overhead and allocation-free implementation of the df-sane method for solving
large-scale nonlinear systems of equations. For in depth information about all the
parameters and the algorithm, see [la2006spectral](@citet).

### Keyword Arguments

  - `sigma_min`: the minimum value of the spectral coefficient `σ` which is related to the
    step size in the algorithm. Defaults to `1e-10`.
  - `sigma_max`: the maximum value of the spectral coefficient `σₙ` which is related to the
    step size in the algorithm. Defaults to `1e10`.

For other keyword arguments, see RobustNonMonotoneLineSearch in LineSearch.jl.
"""
function DFSane(;
        sigma_min = 1 // 10^10, sigma_max = 1.0e10, sigma_1 = 1, M::Int = 10,
        gamma = 1 // 10^4, tau_min = 1 // 10, tau_max = 1 // 2, n_exp::Int = 2,
        max_inner_iterations::Int = 100, eta_strategy::F = (
            fn_1, n, x_n, f_n,
        ) -> fn_1 / n^2
    ) where {F}
    linesearch = RobustNonMonotoneLineSearch(;
        gamma = gamma, sigma_1 = sigma_1, M, tau_min = tau_min, tau_max = tau_max,
        n_exp, η_strategy = eta_strategy, maxiters = max_inner_iterations
    )
    return GeneralizedDFSane(;
        linesearch, sigma_min, sigma_max, sigma_1 = nothing, name = :DFSane
    )
end
