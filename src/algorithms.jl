# Define Algorithms extended via extensions
"""
    LSOptimSolver(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)

Wrapper over [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl) for solving
`NonlinearLeastSquaresProblem`.

## Arguments:

- `alg`: Algorithm to use. Can be `:lm` or `:dogleg`.
- `linsolve`: Linear solver to use. Can be `:qr`, `:cholesky` or `:lsmr`. If
  `nothing`, then `LeastSquaresOptim.jl` will choose the best linear solver based
  on the Jacobian structure.

!!! note
    This algorithm is only available if `LeastSquaresOptim.jl` is installed.
"""
struct LSOptimSolver{alg, linsolve} <: AbstractNonlinearSolveAlgorithm
    autodiff::Symbol

    function LSOptimSolver(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)
        @assert alg in (:lm, :dogleg)
        @assert linsolve === nothing || linsolve in (:qr, :cholesky, :lsmr)
        @assert autodiff in (:central, :forward)

        return new{alg, linsolve}(autodiff)
    end
end
