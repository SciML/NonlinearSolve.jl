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
- `autodiff`: Automatic differentiation / Finite Differences. Can be `:central` or `:forward`.

!!! note
    This algorithm is only available if `LeastSquaresOptim.jl` is installed.
"""
struct LSOptimSolver{alg, linsolve} <: AbstractNonlinearSolveAlgorithm
    autodiff::Symbol
end

function LSOptimSolver(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)
    @assert alg in (:lm, :dogleg)
    @assert linsolve === nothing || linsolve in (:qr, :cholesky, :lsmr)
    @assert autodiff in (:central, :forward)

    if !extension_loaded(Val(:LeastSquaresOptim))
        @warn "LeastSquaresOptim.jl is not loaded! It needs to be explicitly loaded \
               before `solve(prob, LSOptimSolver())` is called."
    end

    return LSOptimSolver{alg, linsolve}(autodiff)
end

"""
    FastLevenbergMarquardtSolver(linsolve = :cholesky)

Wrapper over [FastLevenbergMarquardt.jl](https://github.com/kamesy/FastLevenbergMarquardt.jl) for solving
`NonlinearLeastSquaresProblem`.

!!! warning
    This is not really the fastest solver. It is called that since the original package
    is called "Fast". `LevenbergMarquardt()` is almost always a better choice.

!!! warning
    This algorithm requires the jacobian function to be provided!

## Arguments:

- `linsolve`: Linear solver to use. Can be `:qr` or `:cholesky`.

!!! note
    This algorithm is only available if `FastLevenbergMarquardt.jl` is installed.
"""
@concrete struct FastLevenbergMarquardtSolver{linsolve} <: AbstractNonlinearSolveAlgorithm
    factor
    factoraccept
    factorreject
    factorupdate::Symbol
    minscale
    maxscale
    minfactor
    maxfactor
end

function FastLevenbergMarquardtSolver(linsolve::Symbol = :cholesky; factor = 1e-6,
    factoraccept = 13.0, factorreject = 3.0, factorupdate = :marquardt,
    minscale = 1e-12, maxscale = 1e16, minfactor = 1e-28, maxfactor = 1e32)
    @assert linsolve in (:qr, :cholesky)
    @assert factorupdate in (:marquardt, :nielson)

    if !extension_loaded(Val(:FastLevenbergMarquardt))
        @warn "FastLevenbergMarquardt.jl is not loaded! It needs to be explicitly loaded \
               before `solve(prob, FastLevenbergMarquardtSolver())` is called."
    end

    return FastLevenbergMarquardtSolver{linsolve}(factor, factoraccept, factorreject,
        factorupdate, minscale, maxscale, minfactor, maxfactor)
end
