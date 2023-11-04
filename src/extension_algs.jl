# This file only include the algorithm struct to be exported by LinearSolve.jl. The main
# functionality is implemented as package extensions
"""
    LeastSquaresOptimJL(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)

Wrapper over [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl)
for solving `NonlinearLeastSquaresProblem`.

## Arguments:

- `alg`: Algorithm to use. Can be `:lm` or `:dogleg`.
- `linsolve`: Linear solver to use. Can be `:qr`, `:cholesky` or `:lsmr`. If
  `nothing`, then `LeastSquaresOptim.jl` will choose the best linear solver based
  on the Jacobian structure.
- `autodiff`: Automatic differentiation / Finite Differences. Can be `:central` or `:forward`.

!!! note
    This algorithm is only available if `LeastSquaresOptim.jl` is installed.
"""
struct LeastSquaresOptimJL{alg, linsolve} <: AbstractNonlinearSolveAlgorithm
    autodiff::Symbol
end

function LeastSquaresOptimJL(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)
    @assert alg in (:lm, :dogleg)
    @assert linsolve === nothing || linsolve in (:qr, :cholesky, :lsmr)
    @assert autodiff in (:central, :forward)

    if Base.get_extension(@__MODULE__, :NonlinearSolveLeastSquaresOptimExt) === nothing
        error("LeastSquaresOptimJL requires LeastSquaresOptim.jl to be loaded")
    end

    return LeastSquaresOptimJL{alg, linsolve}(autodiff)
end

"""
    FastLevenbergMarquardtJL(linsolve = :cholesky)

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
@concrete struct FastLevenbergMarquardtJL{linsolve} <: AbstractNonlinearSolveAlgorithm
    factor
    factoraccept
    factorreject
    factorupdate::Symbol
    minscale
    maxscale
    minfactor
    maxfactor
end

function FastLevenbergMarquardtJL(linsolve::Symbol = :cholesky; factor = 1e-6,
        factoraccept = 13.0, factorreject = 3.0, factorupdate = :marquardt,
        minscale = 1e-12, maxscale = 1e16, minfactor = 1e-28, maxfactor = 1e32)
    @assert linsolve in (:qr, :cholesky)
    @assert factorupdate in (:marquardt, :nielson)

    if Base.get_extension(@__MODULE__, :NonlinearSolveFastLevenbergMarquardtExt) === nothing
        error("LeastSquaresOptimJL requires FastLevenbergMarquardt.jl to be loaded")
    end

    return FastLevenbergMarquardtJL{linsolve}(factor, factoraccept, factorreject,
        factorupdate, minscale, maxscale, minfactor, maxfactor)
end

abstract type MINPACKAlgorithm <: SciMLBase.AbstractNonlinearAlgorithm end

"""
```julia
CMINPACK(;show_trace::Bool=false, tracing::Bool=false, method::Symbol=:hybr,
          io::IO=stdout)
```

### Keyword Arguments

- `show_trace`: whether to show the trace.
- `tracing`: who the hell knows what this does. If you find out, please open an issue/PR.
- `method`: the choice of method for the solver.
- `io`: the IO to print any tracing output to.

### Method Choices

The keyword argument `method` can take on different value depending on which method of `fsolve` you are calling. The standard choices of `method` are:

- `:hybr`: Modified version of Powell's algorithm. Uses MINPACK routine [`hybrd1`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrd1.c)
- `:lm`: Levenberg-Marquardt. Uses MINPACK routine [`lmdif1`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmdif1.c)
- `:lmdif`: Advanced Levenberg-Marquardt (more options available with `;kwargs...`). See MINPACK routine [`lmdif`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmdif.c) for more information
- `:hybrd`: Advanced modified version of Powell's algorithm (more options available with `;kwargs...`). See MINPACK routine [`hybrd`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrd.c) for more information

If a Jacobian is supplied as part of the [`NonlinearFunction`](@ref nonlinearfunctions),
then the following methods are allowed:

- `:hybr`: Advanced modified version of Powell's algorithm with user supplied Jacobian. Additional arguments are available via `;kwargs...`. See MINPACK routine [`hybrj`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrj.c) for more information
- `:lm`: Advanced Levenberg-Marquardt with user supplied Jacobian. Additional arguments are available via `;kwargs...`. See MINPACK routine [`lmder`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmder.c) for more information
"""
struct CMINPACK <: MINPACKAlgorithm
    show_trace::Bool
    tracing::Bool
    method::Symbol
    io::IO
end

function CMINPACK(; show_trace::Bool = false, tracing::Bool = false, method::Symbol = :hybr,
                  io::IO = stdout)
    CMINPACK(show_trace, tracing, method, io)
end