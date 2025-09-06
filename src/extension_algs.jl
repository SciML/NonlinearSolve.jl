# This file only include the algorithm struct to be exported by NonlinearSolve.jl. The main
# functionality is implemented as package extensions
"""
    LeastSquaresOptimJL(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)

Wrapper over [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl)
for solving `NonlinearLeastSquaresProblem`.

### Arguments

  - `alg`: Algorithm to use. Can be `:lm` or `:dogleg`.

### Keyword Arguments

  - `linsolve`: Linear solver to use. Can be `:qr`, `:cholesky` or `:lsmr`. If `nothing`,
    then `LeastSquaresOptim.jl` will choose the best linear solver based on the Jacobian
    structure.
  - `autodiff`: Automatic differentiation / Finite Differences. Can be `:central` or
    `:forward`.

!!! note

    This algorithm is only available if `LeastSquaresOptim.jl` is installed and loaded.
"""
@concrete struct LeastSquaresOptimJL <: AbstractNonlinearSolveAlgorithm
    autodiff
    alg::Symbol
    linsolve <: Union{Symbol, Nothing}
    name::Symbol
end

function LeastSquaresOptimJL(alg = :lm; linsolve = nothing, autodiff = :central)
    @assert alg in (:lm, :dogleg)
    @assert linsolve === nothing || linsolve in (:qr, :cholesky, :lsmr)
    autodiff isa Symbol && @assert autodiff in (:central, :forward)

    if Base.get_extension(@__MODULE__, :NonlinearSolveLeastSquaresOptimExt) === nothing
        error("`LeastSquaresOptimJL` requires `LeastSquaresOptim.jl` to be loaded")
    end

    return LeastSquaresOptimJL(autodiff, alg, linsolve, :LeastSquaresOptimJL)
end

"""
    FastLevenbergMarquardtJL(
        linsolve::Symbol = :cholesky;
        factor = 1e-6, factoraccept = 13.0, factorreject = 3.0, factorupdate = :marquardt,
        minscale = 1e-12, maxscale = 1e16, minfactor = 1e-28, maxfactor = 1e32,
        autodiff = nothing
    )

Wrapper over [FastLevenbergMarquardt.jl](https://github.com/kamesy/FastLevenbergMarquardt.jl)
for solving `NonlinearLeastSquaresProblem`. For details about the other keyword arguments
see the documentation for `FastLevenbergMarquardt.jl`.

### Arguments

  - `linsolve`: Linear solver to use. Can be `:qr` or `:cholesky`.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!

!!! note

    This algorithm is only available if `FastLevenbergMarquardt.jl` is installed and loaded.
"""
@concrete struct FastLevenbergMarquardtJL <: AbstractNonlinearSolveAlgorithm
    autodiff
    linsolve::Symbol
    factor
    factoraccept
    factorreject
    factorupdate::Symbol
    minscale
    maxscale
    minfactor
    maxfactor
    name::Symbol
end

function FastLevenbergMarquardtJL(
        linsolve::Symbol = :cholesky; factor = 1e-6, factoraccept = 13.0,
        factorreject = 3.0, factorupdate = :marquardt, minscale = 1e-12,
        maxscale = 1e16, minfactor = 1e-28, maxfactor = 1e32, autodiff = nothing
)
    @assert linsolve in (:qr, :cholesky)
    @assert factorupdate in (:marquardt, :nielson)

    if Base.get_extension(@__MODULE__, :NonlinearSolveFastLevenbergMarquardtExt) === nothing
        error("`FastLevenbergMarquardtJL` requires `FastLevenbergMarquardt.jl` to be loaded")
    end

    return FastLevenbergMarquardtJL(
        autodiff, linsolve, factor, factoraccept, factorreject,
        factorupdate, minscale, maxscale, minfactor, maxfactor, :FastLevenbergMarquardtJL
    )
end

"""
    CMINPACK(; method::Symbol = :auto, autodiff = missing)

### Keyword Arguments

  - `method`: the choice of method for the solver.
  - `autodiff`: Defaults to `missing`, which means we will default to letting `MINPACK`
    construct the jacobian if `f.jac` is not provided. In other cases, we use it to generate
    a jacobian similar to other NonlinearSolve solvers.

### Submethod Choice

The keyword argument `method` can take on different value depending on which method of
`fsolve` you are calling. The standard choices of `method` are:

  - `:hybr`: Modified version of Powell's algorithm. Uses MINPACK routine
    [`hybrd1`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrd1.c)
  - `:lm`: Levenberg-Marquardt. Uses MINPACK routine
    [`lmdif1`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmdif1.c)
  - `:lmdif`: Advanced Levenberg-Marquardt (more options available with `; kwargs...`). See
    MINPACK routine [`lmdif`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmdif.c)
    for more information
  - `:hybrd`: Advanced modified version of Powell's algorithm (more options available with
    `; kwargs...`). See MINPACK routine
    [`hybrd`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrd.c)
    for more information

If a Jacobian is supplied as part of the [`NonlinearFunction`](@ref nonlinearfunctions),
then the following methods are allowed:

  - `:hybr`: Advanced modified version of Powell's algorithm with user supplied Jacobian.
    Additional arguments are available via `; kwargs...`. See MINPACK routine
    [`hybrj`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/hybrj.c)
    for more information
  - `:lm`: Advanced Levenberg-Marquardt with user supplied Jacobian. Additional arguments
    are available via `; kwargs...`. See MINPACK routine
    [`lmder`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmder.c)
    for more information

The default choice of `:auto` selects `:hybr` for NonlinearProblem and `:lm` for
NonlinearLeastSquaresProblem.

!!! note

    This algorithm is only available if `MINPACK.jl` is installed and loaded.
"""
@concrete struct CMINPACK <: AbstractNonlinearSolveAlgorithm
    method::Symbol
    autodiff
    name::Symbol
end

function CMINPACK(; method::Symbol = :auto, autodiff = missing)
    if Base.get_extension(@__MODULE__, :NonlinearSolveMINPACKExt) === nothing
        error("`CMINPACK` requires `MINPACK.jl` to be loaded")
    end
    return CMINPACK(method, autodiff, :CMINPACK)
end

"""
    NLsolveJL(;
        method = :trust_region, autodiff = :central, linesearch = Static(),
        linsolve = (x, A, b) -> copyto!(x, A\\b), factor = one(Float64), autoscale = true,
        m = 10, beta = one(Float64)
    )

### Keyword Arguments

  - `method`: the choice of method for solving the nonlinear system.
  - `autodiff`: the choice of method for generating the Jacobian. Defaults to `:central` or
    central differencing via FiniteDiff.jl. The other choices are `:forward` or `ADTypes`
    similar to other solvers in NonlinearSolve.
  - `linesearch`: the line search method to be used within the solver method. The choices
    are line search types from
    [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).
  - `linsolve`: a function `linsolve(x, A, b)` that solves `Ax = b`.
  - `factor`: determines the size of the initial trust region. This size is set to the
    product of factor and the euclidean norm of `u0` if nonzero, or else to factor itself.
  - `autoscale`: if true, then the variables will be automatically rescaled. The scaling
    factors are the norms of the Jacobian columns.
  - `m`: the amount of history in the Anderson method. Naive "Picard"-style iteration can be
    achieved by setting m=0, but that isn't advisable for contractions whose Lipschitz
    constants are close to 1. If convergence fails, though, you may consider lowering it.
  - `beta`: It is also known as DIIS or Pulay mixing, this method is based on the
    acceleration of the fixed-point iteration xₙ₊₁ = xₙ + beta*f(xₙ), where by default
    beta = 1.

!!! warning

    Line Search Algorithms from [`LineSearch.jl`](https://github.com/SciML/LineSearch.jl)
    aren't supported by `NLsolveJL`. Instead, use the line search algorithms from
    [`LineSearches.jl`](https://github.com/JuliaNLSolvers/LineSearches.jl).

### Submethod Choice

Choices for methods in `NLsolveJL`:

  - `:anderson`: Anderson-accelerated fixed-point iteration
  - `:broyden`: Broyden's quasi-Newton method
  - `:newton`: Classical Newton method with an optional line search
  - `:trust_region`: Trust region Newton method (the default choice)

For more information on these arguments, consult the
[NLsolve.jl documentation](https://github.com/JuliaNLSolvers/NLsolve.jl).

!!! note

    This algorithm is only available if `NLsolve.jl` is installed and loaded.
"""
@concrete struct NLsolveJL <: AbstractNonlinearSolveAlgorithm
    method::Symbol
    autodiff
    linesearch
    linsolve
    factor
    autoscale::Bool
    m::Int
    beta
    name::Symbol
end

function NLsolveJL(;
        method = :trust_region, autodiff = :central, linesearch = missing, beta = 1.0,
        linsolve = (x, A, b) -> copyto!(x, A \ b), factor = 1.0, autoscale = true, m = 10
)
    if Base.get_extension(@__MODULE__, :NonlinearSolveNLsolveExt) === nothing
        error("`NLsolveJL` requires `NLsolve.jl` to be loaded")
    end

    if autodiff isa Symbol && autodiff !== :central && autodiff !== :forward
        error("`autodiff` must be `:central` or `:forward`.")
    end

    return NLsolveJL(
        method, autodiff, linesearch, linsolve, factor, autoscale, m, beta, :NLsolveJL
    )
end

"""
    NLSolversJL(method; autodiff = nothing)
    NLSolversJL(; method, autodiff = nothing)

Wrapper over NLSolvers.jl Nonlinear Equation Solvers. We automatically construct the
jacobian function and supply it to the solver.

### Arguments

  - `method`: the choice of method for solving the nonlinear system. See the documentation
    for NLSolvers.jl for more information.
  - `autodiff`: the choice of method for generating the Jacobian. Defaults to `nothing`
    which means that a default is selected according to the problem specification. Can be
    any valid ADTypes.jl autodiff type (conditional on that backend being supported in
    NonlinearSolve.jl).
"""
struct NLSolversJL{M, AD} <: AbstractNonlinearSolveAlgorithm
    method::M
    autodiff::AD
    name::Symbol

    function NLSolversJL(method, autodiff)
        if Base.get_extension(@__MODULE__, :NonlinearSolveNLSolversExt) === nothing
            error("NLSolversJL requires NLSolvers.jl to be loaded")
        end

        return new{typeof(method), typeof(autodiff)}(method, autodiff, :NLSolversJL)
    end
end

NLSolversJL(method; autodiff = nothing) = NLSolversJL(method, autodiff)
NLSolversJL(; method, autodiff = nothing) = NLSolversJL(method, autodiff)

"""
    SpeedMappingJL(;
        σ_min = 0.0, stabilize::Bool = false, check_obj::Bool = false,
        orders::Vector{Int} = [3, 3, 2]
    )

Wrapper over [SpeedMapping.jl](https://github.com/NicolasL-S/SpeedMapping.jl) for solving
Fixed Point Problems. We allow using this algorithm to solve root finding problems as well.

### Keyword Arguments

  - `σ_min`: Setting to `1` may avoid stalling (see [lepage2021alternating](@cite)).
  - `stabilize`: performs a stabilization mapping before extrapolating. Setting to `true`
    may improve the performance for applications like accelerating the EM or MM algorithms
    (see [lepage2021alternating](@cite)).
  - `check_obj`: In case of NaN or Inf values, the algorithm restarts at the best past
    iterate.
  - `orders`: determines ACX's alternating order. Must be between `1` and `3` (where `1`
    means no extrapolation). The two recommended orders are `[3, 2]` and `[3, 3, 2]`, the
    latter being potentially better for highly non-linear applications (see
    [lepage2021alternating](@cite)).

!!! note

    This algorithm is only available if `SpeedMapping.jl` is installed and loaded.
"""
@concrete struct SpeedMappingJL <: AbstractNonlinearSolveAlgorithm
    σ_min
    stabilize::Bool
    check_obj::Bool
    orders::Vector{Int}
    name::Symbol
end

function SpeedMappingJL(;
        σ_min = 0.0, stabilize::Bool = false, check_obj::Bool = false,
        orders::Vector{Int} = [3, 3, 2]
)
    if Base.get_extension(@__MODULE__, :NonlinearSolveSpeedMappingExt) === nothing
        error("`SpeedMappingJL` requires `SpeedMapping.jl` to be loaded")
    end

    return SpeedMappingJL(σ_min, stabilize, check_obj, orders, :SpeedMappingJL)
end

"""
    FixedPointAccelerationJL(;
        algorithm = :Anderson, m = missing, condition_number_threshold = missing,
        extrapolation_period = missing, replace_invalids = :NoAction
    )

Wrapper over [FixedPointAcceleration.jl](https://s-baumann.github.io/FixedPointAcceleration.jl/)
for solving Fixed Point Problems. We allow using this algorithm to solve root finding
problems as well.

### Keyword Arguments

  - `algorithm`: The algorithm to use. Can be `:Anderson`, `:MPE`, `:RRE`, `:VEA`, `:SEA`,
    `:Simple`, `:Aitken` or `:Newton`.
  - `m`: The number of previous iterates to use for the extrapolation. Only valid for
    `:Anderson`.
  - `condition_number_threshold`: The condition number threshold for Least Squares Problem.
    Only valid for `:Anderson`.
  - `extrapolation_period`: The number of iterates between extrapolations. Only valid for
    `:MPE`, `:RRE`, `:VEA` and `:SEA`. Defaults to `7` for `:MPE` & `:RRE`, and `6` for
    `:SEA` and `:VEA`. For `:SEA` and `:VEA`, this must be a multiple of `2`.
  - `replace_invalids`: The method to use for replacing invalid iterates. Can be
    `:ReplaceInvalids`, `:ReplaceVector` or `:NoAction`.

!!! note

    This algorithm is only available if `FixedPointAcceleration.jl` is installed and loaded.
"""
@concrete struct FixedPointAccelerationJL <: AbstractNonlinearSolveAlgorithm
    algorithm::Symbol
    extrapolation_period::Int
    replace_invalids::Symbol
    dampening
    m::Int
    condition_number_threshold
    name::Symbol
end

function FixedPointAccelerationJL(;
        algorithm = :Anderson, m = missing, condition_number_threshold = missing,
        extrapolation_period = missing, replace_invalids = :NoAction, dampening = 1.0
)
    if Base.get_extension(@__MODULE__, :NonlinearSolveFixedPointAccelerationExt) === nothing
        error("`FixedPointAccelerationJL` requires `FixedPointAcceleration.jl` to be loaded")
    end

    @assert algorithm in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
    @assert replace_invalids in (:ReplaceInvalids, :ReplaceVector, :NoAction)

    if algorithm !== :Anderson
        @assert condition_number_threshold===missing "`condition_number_threshold` is only valid for Anderson acceleration"
        @assert m===missing "`m` is only valid for Anderson acceleration"
    end
    condition_number_threshold === missing && (condition_number_threshold = 1e3)
    m === missing && (m = 10)

    if algorithm !== :MPE && algorithm !== :RRE && algorithm !== :VEA && algorithm !== :SEA
        @assert extrapolation_period===missing "`extrapolation_period` is only valid for MPE, RRE, VEA and SEA"
    end
    if extrapolation_period === missing
        extrapolation_period = algorithm === :SEA || algorithm === :VEA ? 6 : 7
    else
        if (algorithm === :SEA || algorithm === :VEA) && extrapolation_period % 2 != 0
            throw(AssertionError("`extrapolation_period` must be multiples of 2 for SEA and VEA"))
        end
    end

    return FixedPointAccelerationJL(
        algorithm, extrapolation_period, replace_invalids,
        dampening, m, condition_number_threshold, :FixedPointAccelerationJL
    )
end

"""
    SIAMFANLEquationsJL(;
        method = :newton, delta = 1e-3, linsolve = nothing, autodiff = missing
    )

### Keyword Arguments

  - `method`: the choice of method for solving the nonlinear system.
  - `delta`: initial pseudo time step, default is 1e-3.
  - `linsolve` : JFNK linear solvers, choices are `gmres` and `bicgstab`.
  - `m`: Depth for Anderson acceleration, default as 0 for Picard iteration.
  - `beta`: Anderson mixing parameter, change f(x) to (1-beta)x+beta*f(x),
    equivalent to accelerating damped Picard iteration.
  - `autodiff`: Defaults to `missing`, which means we will default to letting
    `SIAMFANLEquations` construct the jacobian if `f.jac` is not provided. In other cases,
    we use it to generate a jacobian similar to other NonlinearSolve solvers.

### Submethod Choice

  - `:newton`: Classical Newton method.
  - `:pseudotransient`: Pseudo transient method.
  - `:secant`: Secant method for scalar equations.
  - `:anderson`: Anderson acceleration for fixed point iterations.

!!! note

    This algorithm is only available if `SIAMFANLEquations.jl` is installed and loaded.
"""
@concrete struct SIAMFANLEquationsJL <: AbstractNonlinearSolveAlgorithm
    method::Symbol
    delta
    linsolve <: Union{Symbol, Nothing}
    m::Int
    beta
    autodiff
    name::Symbol
end

function SIAMFANLEquationsJL(;
        method = :newton, delta = 1e-3, linsolve = nothing, m = 0, beta = 1.0,
        autodiff = missing
)
    if Base.get_extension(@__MODULE__, :NonlinearSolveSIAMFANLEquationsExt) === nothing
        error("`SIAMFANLEquationsJL` requires `SIAMFANLEquations.jl` to be loaded")
    end
    return SIAMFANLEquationsJL(
        method, delta, linsolve, m, beta, autodiff, :SIAMFANLEquationsJL
    )
end

"""
    PETScSNES(; petsclib = missing, autodiff = nothing, mpi_comm = missing, kwargs...)

Wrapper over [PETSc.jl](https://github.com/JuliaParallel/PETSc.jl) SNES solvers.

### Keyword Arguments

  - `petsclib`: PETSc library to use. If set to `missing`, then we will use the first
    available PETSc library in `PETSc.petsclibs` based on the problem element type.
  - `autodiff`: the choice of method for generating the Jacobian. Defaults to `nothing`
    which means that a default is selected according to the problem specification. Can be
    any valid ADTypes.jl autodiff type (conditional on that backend being supported in
    NonlinearSolve.jl). If set to `missing`, then PETSc computes the Jacobian using finite
    differences.
  - `mpi_comm`: MPI communicator to use. If set to `missing`, then we will use
    `MPI.COMM_SELF`.
  - `kwargs`: Keyword arguments to be passed to the PETSc SNES solver. See [PETSc SNES
    documentation](https://petsc.org/release/manual/snes/) and
    [SNESSetFromOptions](https://petsc.org/release/manualpages/SNES/SNESSetFromOptions)
    for more information.

### Options via `CommonSolve.solve`

These options are forwarded from `solve` to the PETSc SNES solver. If these are provided to
`kwargs`, then they will be ignored.

| `solve` option | PETSc SNES option |
|:-------------- |:----------------- |
| `atol`         | `snes_atol`       |
| `rtol`         | `snes_rtol`       |
| `maxiters`     | `snes_max_it`     |
| `show_trace`   | `snes_monitor`    |

!!! note

    This algorithm is only available if `PETSc.jl` is installed and loaded.
"""
@concrete struct PETScSNES <: AbstractNonlinearSolveAlgorithm
    petsclib
    mpi_comm
    autodiff
    snes_options
end

function PETScSNES(; petsclib = missing, autodiff = nothing, mpi_comm = missing, kwargs...)
    if Base.get_extension(@__MODULE__, :NonlinearSolvePETScExt) === nothing
        error("`PETScSNES` requires `PETSc.jl` to be loaded")
    end
    return PETScSNES(petsclib, mpi_comm, autodiff, kwargs)
end
