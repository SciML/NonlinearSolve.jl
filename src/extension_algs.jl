# This file only include the algorithm struct to be exported by LinearSolve.jl. The main
# functionality is implemented as package extensions
"""
    LeastSquaresOptimJL(alg = :lm; linsolve = nothing, autodiff::Symbol = :central)

Wrapper over [LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl)
for solving `NonlinearLeastSquaresProblem`.

## Arguments:

  - `alg`: Algorithm to use. Can be `:lm` or `:dogleg`.
  - `linsolve`: Linear solver to use. Can be `:qr`, `:cholesky` or `:lsmr`. If `nothing`,
    then `LeastSquaresOptim.jl` will choose the best linear solver based on the Jacobian
    structure.
  - `autodiff`: Automatic differentiation / Finite Differences. Can be `:central` or
    `:forward`.

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
    FastLevenbergMarquardtJL(linsolve = :cholesky; autodiff = nothing)

Wrapper over [FastLevenbergMarquardt.jl](https://github.com/kamesy/FastLevenbergMarquardt.jl)
for solving `NonlinearLeastSquaresProblem`.

!!! warning

    This is not really the fastest solver. It is called that since the original package
    is called "Fast". `LevenbergMarquardt()` is almost always a better choice.

## Arguments:

  - `linsolve`: Linear solver to use. Can be `:qr` or `:cholesky`.
  - `autodiff`: determines the backend used for the Jacobian. Note that this argument is
    ignored if an analytical Jacobian is passed, as that will be used instead. Defaults to
    `nothing` which means that a default is selected according to the problem specification!
    Valid choices are `nothing`, `AutoForwardDiff` or `AutoFiniteDiff`.

!!! note

    This algorithm is only available if `FastLevenbergMarquardt.jl` is installed.
"""
@concrete struct FastLevenbergMarquardtJL{linsolve} <: AbstractNonlinearSolveAlgorithm
    ad
    factor
    factoraccept
    factorreject
    factorupdate::Symbol
    minscale
    maxscale
    minfactor
    maxfactor
end

function set_ad(alg::FastLevenbergMarquardtJL{linsolve}, ad) where {linsolve}
    return FastLevenbergMarquardtJL{linsolve}(ad, alg.factor, alg.factoraccept,
        alg.factorreject, alg.factorupdate, alg.minscale, alg.maxscale, alg.minfactor,
        alg.maxfactor)
end

function FastLevenbergMarquardtJL(linsolve::Symbol = :cholesky; factor = 1e-6,
        factoraccept = 13.0, factorreject = 3.0, factorupdate = :marquardt,
        minscale = 1e-12, maxscale = 1e16, minfactor = 1e-28, maxfactor = 1e32,
        autodiff = nothing)
    @assert linsolve in (:qr, :cholesky)
    @assert factorupdate in (:marquardt, :nielson)
    @assert autodiff === nothing || autodiff isa AutoFiniteDiff ||
            autodiff isa AutoForwardDiff

    if Base.get_extension(@__MODULE__, :NonlinearSolveFastLevenbergMarquardtExt) === nothing
        error("FastLevenbergMarquardtJL requires FastLevenbergMarquardt.jl to be loaded")
    end

    return FastLevenbergMarquardtJL{linsolve}(autodiff, factor, factoraccept, factorreject,
        factorupdate, minscale, maxscale, minfactor, maxfactor)
end

"""
    CMINPACK(; method::Symbol = :auto)

### Keyword Arguments

  - `method`: the choice of method for the solver.

### Method Choices

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
    are available via `;kwargs...`. See MINPACK routine
    [`lmder`](https://github.com/devernay/cminpack/blob/d1f5f5a273862ca1bbcf58394e4ac060d9e22c76/lmder.c)
    for more information

The default choice of `:auto` selects `:hybr` for NonlinearProblem and `:lm` for
NonlinearLeastSquaresProblem.
"""
struct CMINPACK <: AbstractNonlinearSolveAlgorithm
    show_trace::Bool
    tracing::Bool
    method::Symbol
end

function CMINPACK(; show_trace = missing, tracing = missing, method::Symbol = :auto)
    if Base.get_extension(@__MODULE__, :NonlinearSolveMINPACKExt) === nothing
        error("CMINPACK requires MINPACK.jl to be loaded")
    end

    if show_trace !== missing
        Base.depwarn("`show_trace` for CMINPACK has been deprecated and will be removed \
                      in v4. Use the `show_trace` keyword argument via the logging API \
                      https://docs.sciml.ai/NonlinearSolve/stable/basics/Logging/ \
                      instead.", :CMINPACK)
    else
        show_trace = false
    end

    if tracing !== missing
        Base.depwarn("`tracing` for CMINPACK has been deprecated and will be removed \
                      in v4. Use the `store_trace` keyword argument via the logging API \
                      https://docs.sciml.ai/NonlinearSolve/stable/basics/Logging/ \
                      instead.", :CMINPACK)
    else
        tracing = false
    end

    return CMINPACK(show_trace, tracing, method)
end

"""
    NLsolveJL(; method = :trust_region, autodiff = :central, linesearch = Static(),
        linsolve = (x, A, b) -> copyto!(x, A\\b), factor = one(Float64), autoscale = true,
        m = 10, beta = one(Float64))

### Keyword Arguments

  - `method`: the choice of method for solving the nonlinear system.
  - `autodiff`: the choice of method for generating the Jacobian. Defaults to `:central` or
    central differencing via FiniteDiff.jl. The other choices are `:forward`
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
  - `beta`: It is also known as DIIS or Pulay mixing, this method is based on the acceleration
    of the fixed-point iteration xₙ₊₁ = xₙ + beta*f(xₙ), where by default beta = 1.

### Submethod Choice

Choices for methods in `NLsolveJL`:

  - `:anderson`: Anderson-accelerated fixed-point iteration
  - `:broyden`: Broyden's quasi-Newton method
  - `:newton`: Classical Newton method with an optional line search
  - `:trust_region`: Trust region Newton method (the default choice) For more information on
    these arguments, consult the
    [NLsolve.jl documentation](https://github.com/JuliaNLSolvers/NLsolve.jl).
"""
@concrete struct NLsolveJL <: AbstractNonlinearSolveAlgorithm
    method::Symbol
    autodiff::Symbol
    store_trace::Bool
    extended_trace::Bool
    linesearch
    linsolve
    factor
    autoscale::Bool
    m::Int
    beta
    show_trace::Bool
end

function NLsolveJL(; method = :trust_region, autodiff = :central, store_trace = missing,
        extended_trace = missing, linesearch = LineSearches.Static(),
        linsolve = (x, A, b) -> copyto!(x, A \ b), factor = 1.0, autoscale = true, m = 10,
        beta = one(Float64), show_trace = missing)
    if Base.get_extension(@__MODULE__, :NonlinearSolveNLsolveExt) === nothing
        error("NLsolveJL requires NLsolve.jl to be loaded")
    end

    if show_trace !== missing
        Base.depwarn("`show_trace` for NLsolveJL has been deprecated and will be removed \
                      in v4. Use the `show_trace` keyword argument via the logging API \
                      https://docs.sciml.ai/NonlinearSolve/stable/basics/Logging/ \
                      instead.", :NLsolveJL)
    else
        show_trace = false
    end

    if store_trace !== missing
        Base.depwarn("`store_trace` for NLsolveJL has been deprecated and will be removed \
                      in v4. Use the `store_trace` keyword argument via the logging API \
                      https://docs.sciml.ai/NonlinearSolve/stable/basics/Logging/ \
                      instead.", :NLsolveJL)
    else
        store_trace = false
    end

    if extended_trace !== missing
        Base.depwarn("`extended_trace` for NLsolveJL has been deprecated and will be \
                      removed in v4. Use the `trace_level = TraceAll()` keyword argument \
                      via the logging API \
                      https://docs.sciml.ai/NonlinearSolve/stable/basics/Logging/ instead.",
            :NLsolveJL)
    else
        extended_trace = false
    end

    return NLsolveJL(method, autodiff, store_trace, extended_trace, linesearch, linsolve,
        factor, autoscale, m, beta, show_trace)
end

"""
    SpeedMappingJL(; σ_min = 0.0, stabilize::Bool = false, check_obj::Bool = false,
        orders::Vector{Int} = [3, 3, 2], time_limit::Real = 1000)

Wrapper over [SpeedMapping.jl](https://nicolasl-s.github.io/SpeedMapping.jl) for solving
Fixed Point Problems. We allow using this algorithm to solve root finding problems as well.

## Arguments:

  - `σ_min`: Setting to `1` may avoid stalling (see paper).
  - `stabilize`: performs a stabilization mapping before extrapolating. Setting to `true`
    may improve the performance for applications like accelerating the EM or MM algorithms
    (see paper).
  - `check_obj`: In case of NaN or Inf values, the algorithm restarts at the best past
    iterate.
  - `orders`: determines ACX's alternating order. Must be between `1` and `3` (where `1`
    means no extrapolation). The two recommended orders are `[3, 2]` and `[3, 3, 2]`, the
    latter being potentially better for highly non-linear applications (see paper).
  - `time_limit`: time limit for the algorithm.

## References:

  - N. Lepage-Saucier, Alternating cyclic extrapolation methods for optimization algorithms,
    arXiv:2104.04974 (2021). https://arxiv.org/abs/2104.04974.
"""
@concrete struct SpeedMappingJL <: AbstractNonlinearSolveAlgorithm
    σ_min
    stabilize::Bool
    check_obj::Bool
    orders::Vector{Int}
    time_limit
end

function SpeedMappingJL(; σ_min = 0.0, stabilize::Bool = false, check_obj::Bool = false,
        orders::Vector{Int} = [3, 3, 2], time_limit::Real = 1000)
    if Base.get_extension(@__MODULE__, :NonlinearSolveSpeedMappingExt) === nothing
        error("SpeedMappingJL requires SpeedMapping.jl to be loaded")
    end

    return SpeedMappingJL(σ_min, stabilize, check_obj, orders, time_limit)
end

"""
    FixedPointAccelerationJL(; algorithm = :Anderson, m = missing,
        condition_number_threshold = missing, extrapolation_period = missing,
        replace_invalids = :NoAction)

Wrapper over [FixedPointAcceleration.jl](https://s-baumann.github.io/FixedPointAcceleration.jl/)
for solving Fixed Point Problems. We allow using this algorithm to solve root finding
problems as well.

## Arguments:

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
"""
@concrete struct FixedPointAccelerationJL <: AbstractNonlinearSolveAlgorithm
    algorithm::Symbol
    extrapolation_period::Int
    replace_invalids::Symbol
    dampening
    m::Int
    condition_number_threshold
end

function FixedPointAccelerationJL(; algorithm = :Anderson, m = missing,
        condition_number_threshold = missing, extrapolation_period = missing,
        replace_invalids = :NoAction, dampening = 1.0)
    if Base.get_extension(@__MODULE__, :NonlinearSolveFixedPointAccelerationExt) === nothing
        error("FixedPointAccelerationJL requires FixedPointAcceleration.jl to be loaded")
    end

    @assert algorithm in (:Anderson, :MPE, :RRE, :VEA, :SEA, :Simple, :Aitken, :Newton)
    @assert replace_invalids in (:ReplaceInvalids, :ReplaceVector, :NoAction)

    if algorithm !== :Anderson
        if condition_number_threshold !== missing
            error("`condition_number_threshold` is only valid for Anderson acceleration")
        end
        if m !== missing
            error("`m` is only valid for Anderson acceleration")
        end
    end
    condition_number_threshold === missing && (condition_number_threshold = 1e3)
    m === missing && (m = 10)

    if algorithm !== :MPE && algorithm !== :RRE && algorithm !== :VEA && algorithm !== :SEA
        if extrapolation_period !== missing
            error("`extrapolation_period` is only valid for MPE, RRE, VEA and SEA")
        end
    end
    if extrapolation_period === missing
        if algorithm === :SEA || algorithm === :VEA
            extrapolation_period = 6
        else
            extrapolation_period = 7
        end
    else
        if (algorithm === :SEA || algorithm === :VEA) && extrapolation_period % 2 != 0
            error("`extrapolation_period` must be multiples of 2 for SEA and VEA")
        end
    end

    return FixedPointAccelerationJL(algorithm, extrapolation_period, replace_invalids,
        dampening, m, condition_number_threshold)
end

"""
    SIAMFANLEquationsJL(; method = :newton, delta = 1e-3, linsolve = nothing)

### Keyword Arguments

  - `method`: the choice of method for solving the nonlinear system.
  - `delta`: initial pseudo time step, default is 1e-3.
  - `linsolve` : JFNK linear solvers, choices are `gmres` and `bicgstab`.

### Submethod Choice

  - `:newton`: Classical Newton method.
  - `:pseudotransient`: Pseudo transient method.
  - `:secant`: Secant method for scalar equations.
"""
@concrete struct SIAMFANLEquationsJL{L <: Union{Symbol, Nothing}} <:
                 AbstractNonlinearSolveAlgorithm
    method::Symbol
    delta
    linsolve::L
end

function SIAMFANLEquationsJL(; method = :newton, delta = 1e-3, linsolve = nothing)
    if Base.get_extension(@__MODULE__, :NonlinearSolveSIAMFANLEquationsExt) === nothing
        error("SIAMFANLEquationsJL requires SIAMFANLEquations.jl to be loaded")
    end
    return SIAMFANLEquationsJL(method, delta, linsolve)
end
