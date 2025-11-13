module NonlinearSolveSciPy

using ConcreteStructs: @concrete
using Reexport: @reexport

using PythonCall: pyimport, pyfunc, Py

const scipy_optimize = Ref{Union{Py, Nothing}}(nothing)
const PY_NONE = Ref{Union{Py, Nothing}}(nothing)
const _SCIPY_AVAILABLE = Ref{Bool}(false)

function __init__()
    try
        scipy_optimize[] = pyimport("scipy.optimize")
        PY_NONE[] = pyimport("builtins").None
        _SCIPY_AVAILABLE[] = true
    catch
        _SCIPY_AVAILABLE[] = false
    end
end

using SciMLBase
using NonlinearSolveBase: AbstractNonlinearSolveAlgorithm,
                          construct_extension_function_wrapper

"""
    SciPyLeastSquares(; method="trf", loss="linear")

Wrapper over `scipy.optimize.least_squares` (via PythonCall) for solving
`NonlinearLeastSquaresProblem`s.  The keyword arguments correspond to the
`method` ("trf", "dogbox", "lm") and the robust loss function ("linear",
"soft_l1", "huber", "cauchy", "arctan").
"""
@concrete struct SciPyLeastSquares <: AbstractNonlinearSolveAlgorithm
    method::String
    loss::String
    name::Symbol
end

function SciPyLeastSquares(; method::String = "trf", loss::String = "linear")
    _SCIPY_AVAILABLE[] ||
        error("`SciPyLeastSquares` requires the Python package `scipy` to be available to PythonCall.")
    valid_methods = ("trf", "dogbox", "lm")
    valid_losses = ("linear", "soft_l1", "huber", "cauchy", "arctan")
    method in valid_methods ||
        throw(ArgumentError(
            lazy"Invalid method: $method. Valid methods are: $(join(valid_methods, \", \"))"))
    loss in valid_losses ||
        throw(ArgumentError(
            lazy"Invalid loss: $loss. Valid loss functions are: $(join(valid_losses, \",
            \"))"))
    return SciPyLeastSquares(method, loss, :SciPyLeastSquares)
end

SciPyLeastSquaresTRF() = SciPyLeastSquares(method = "trf")
SciPyLeastSquaresDogbox() = SciPyLeastSquares(method = "dogbox")
SciPyLeastSquaresLM() = SciPyLeastSquares(method = "lm")

"""
    SciPyRoot(; method="hybr")

Wrapper over `scipy.optimize.root` for solving `NonlinearProblem`s.  Available
methods include "hybr" (default), "lm", "broyden1", "broyden2", "anderson",
"diagbroyden", "linearmixing", "excitingmixing", "krylov", "df-sane" – any
method accepted by SciPy.
"""
@concrete struct SciPyRoot <: AbstractNonlinearSolveAlgorithm
    method::String
    name::Symbol
end

function SciPyRoot(; method::String = "hybr")
    _SCIPY_AVAILABLE[] ||
        error("`SciPyRoot` requires the Python package `scipy` to be available to PythonCall.")
    return SciPyRoot(method, :SciPyRoot)
end

"""
    SciPyRootScalar(; method="brentq")

Wrapper over `scipy.optimize.root_scalar` for scalar `IntervalNonlinearProblem`s
(bracketing problems).  The default method uses Brent's algorithm ("brentq").
Other valid options: "bisect", "brentq", "brenth", "ridder", "toms748",
"secant", "newton" (secant/Newton do not require brackets).
"""
@concrete struct SciPyRootScalar <: AbstractNonlinearSolveAlgorithm
    method::String
    name::Symbol
end

function SciPyRootScalar(; method::String = "brentq")
    _SCIPY_AVAILABLE[] ||
        error("`SciPyRootScalar` requires the Python package `scipy` to be available to PythonCall.")
    return SciPyRootScalar(method, :SciPyRootScalar)
end

"""
Internal: wrap a Julia residual function into a Python callable
"""
function _make_py_residual(f::F, p) where F
    return pyfunc(x_py -> begin
        x = Vector{Float64}(x_py)
        r = f(x, p)
        return r
    end)
end

"""
Internal: wrap a Julia scalar function into a Python callable
"""
function _make_py_scalar(f::F, p) where F
    return pyfunc(x_py -> begin
        x = Float64(x_py)
        return f(x, p)
    end)
end

function SciMLBase.__solve(
        prob::SciMLBase.NonlinearLeastSquaresProblem, alg::SciPyLeastSquares;
        abstol = nothing, maxiters = 10_000, alias_u0::Bool = false,
        kwargs...)
    # Construct Python residual
    py_f = _make_py_residual(prob.f, prob.p)

    # Bounds handling (lb/ub may be missing)
    has_lb = hasproperty(prob, :lb)
    has_ub = hasproperty(prob, :ub)
    if has_lb || has_ub
        lb = has_lb ? getproperty(prob, :lb) : fill(-Inf, length(prob.u0))
        ub = has_ub ? getproperty(prob, :ub) : fill(Inf, length(prob.u0))
        bounds = (lb, ub)
    else
        bounds = nothing
    end

    # Filter out Julia-specific kwargs that scipy doesn't understand
    scipy_kwargs = filter(kwargs) do (k, v)
        k ∉ (:alias, :verbose)
    end

    res = scipy_optimize[].least_squares(py_f, collect(prob.u0);
        method = alg.method,
        loss = alg.loss,
        max_nfev = maxiters,
        bounds = bounds === nothing ? PY_NONE[] : bounds,
        scipy_kwargs...)

    u_vec = Vector{Float64}(res.x)
    resid = Vector{Float64}(res.fun)

    u = prob.u0 isa Number ? u_vec[1] : reshape(u_vec, size(prob.u0))

    ret = res.success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    njev = try
        Int(res.njev)
    catch
        0
    end
    stats = SciMLBase.NLStats(res.nfev, njev, 0, 0, res.nfev)

    return SciMLBase.build_solution(prob, alg, u, resid; retcode = ret,
        original = res, stats = stats)
end

function SciMLBase.__solve(prob::SciMLBase.NonlinearProblem, alg::SciPyRoot;
        abstol = nothing, maxiters = 10_000, alias_u0::Bool = false,
        kwargs...)
    f!, u0, resid = construct_extension_function_wrapper(prob; alias_u0)

    py_f = pyfunc(x_py -> begin
        x = Vector{Float64}(x_py)
        f!(resid, x)
        return resid
    end)

    tol = abstol === nothing ? nothing : abstol

    # Filter out Julia-specific kwargs that scipy doesn't understand
    scipy_kwargs = filter(kwargs) do (k, v)
        k ∉ (:alias, :verbose)
    end

    res = scipy_optimize[].root(py_f, collect(u0);
        method = alg.method,
        tol = tol,
        options = Dict("maxiter" => maxiters),
        scipy_kwargs...)

    u_vec = Vector{Float64}(res.x)
    f!(resid, u_vec)

    u_out = prob.u0 isa Number ? u_vec[1] : reshape(u_vec, size(prob.u0))

    ret = res.success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    nfev = try
        Int(res.nfev)
    catch
        ;
        0
    end
    niter = try
        Int(res.nit)
    catch
        ;
        0
    end
    stats = SciMLBase.NLStats(nfev, 0, 0, 0, niter)

    return SciMLBase.build_solution(prob, alg, u_out, resid; retcode = ret,
        original = res, stats = stats)
end

function SciMLBase.__solve(prob::SciMLBase.IntervalNonlinearProblem, alg::SciPyRootScalar;
        abstol = nothing, maxiters = 10_000, kwargs...)
    f = prob.f
    p = prob.p
    py_f = _make_py_scalar(f, p)

    a, b = prob.tspan

    # Filter out Julia-specific kwargs that scipy doesn't understand
    scipy_kwargs = filter(kwargs) do (k, v)
        k ∉ (:alias, :verbose)
    end

    res = scipy_optimize[].root_scalar(py_f;
        method = alg.method,
        bracket = (a, b),
        maxiter = maxiters,
        xtol = abstol,
        scipy_kwargs...)

    u_root = res.root
    resid = f(u_root, p)

    ret = res.converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    nfev = try
        Int(res.function_calls)
    catch
        ;
        0
    end
    niter = try
        Int(res.iterations)
    catch
        ;
        0
    end
    stats = SciMLBase.NLStats(nfev, 0, 0, 0, niter)

    return SciMLBase.build_solution(prob, alg, u_root, resid; retcode = ret,
        original = res, stats = stats)
end

@reexport using SciMLBase, NonlinearSolveBase

export SciPyLeastSquares, SciPyLeastSquaresTRF, SciPyLeastSquaresDogbox,
       SciPyLeastSquaresLM,
       SciPyRoot, SciPyRootScalar

end
