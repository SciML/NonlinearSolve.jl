module NonlinearSolveSciPyExt

# This file is loaded as an extension when PythonCall is available
using PythonCall
const scipy_optimize = try
    pyimport("scipy.optimize")
catch err
    error("Python package `scipy` could not be imported. Install it in the Python environment used by PythonCall.")
end

using SciMLBase
using NonlinearSolve

# Re-export algorithm type so that `using NonlinearSolve` brings it in when the
# extension is loaded.  
import ..NonlinearSolve: SciPyLeastSquares, SciPyRoot, SciPyRootScalar
using NonlinearSolveBase: construct_extension_function_wrapper

""" Internal: wrap a Julia residual function into a Python callable """
function _make_py_residual(f, p)
    return pyfunc(x_py -> begin
        x = Vector{Float64}(x_py)     
        r  = f(x, p)
        return r                     
    end)
end

""" Internal: wrap a Julia scalar function into a Python callable """
function _make_py_scalar(f, p)
    return pyfunc(x_py -> begin
        x = Float64(x_py)
        return f(x, p)
    end)
end

function SciMLBase.__solve(prob::SciMLBase.NonlinearLeastSquaresProblem, alg::SciPyLeastSquares;
                           abstol = nothing, maxiters = 10_000, alias_u0::Bool = false,
                           kwargs...)
    # Construct Python residual
    py_f = _make_py_residual(prob.f, prob.p)

    # Bounds handling (lb/ub may be missing)
    has_lb = hasproperty(prob, :lb)
    has_ub = hasproperty(prob, :ub)
    if has_lb || has_ub
        lb = has_lb ? getproperty(prob, :lb) : fill(-Inf, length(prob.u0))
        ub = has_ub ? getproperty(prob, :ub) : fill( Inf, length(prob.u0))
        bounds = (lb, ub)
    else
        bounds = nothing
    end

    res = scipy_optimize.least_squares(py_f, collect(prob.u0);
                                       method = alg.method,
                                       loss   = alg.loss,
                                       max_nfev = maxiters,
                                       bounds = bounds === nothing ? py_none : bounds,
                                       kwargs...)

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
    # Get in-place residual wrapper from NonlinearSolveBase.
    f!, u0, resid = construct_extension_function_wrapper(prob; alias_u0)

    py_f = pyfunc(x_py -> begin
        x = Vector{Float64}(x_py)
        f!(resid, x)
        return resid
    end)

    tol = abstol === nothing ? nothing : abstol

    res = scipy_optimize.root(py_f, collect(u0);
                              method = alg.method,
                              tol = tol,
                              options = Dict("maxiter" => maxiters),
                              kwargs...)

    u_vec = Vector{Float64}(res.x)
    f!(resid, u_vec)  

    u_out = prob.u0 isa Number ? u_vec[1] : reshape(u_vec, size(prob.u0))

    ret = res.success ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    nfev = try Int(res.nfev) catch; 0 end
    niter = try Int(res.nit) catch; 0 end
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

    res = scipy_optimize.root_scalar(py_f;
                                     method = alg.method,
                                     bracket = (a, b),
                                     maxiter = maxiters,
                                     xtol = abstol,
                                     kwargs...)

    u_root = res.root
    resid = f(u_root, p)

    ret = res.converged ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.Failure
    nfev = try Int(res.function_calls) catch; 0 end
    niter = try Int(res.iterations) catch; 0 end
    stats = SciMLBase.NLStats(nfev, 0, 0, 0, niter)

    return SciMLBase.build_solution(prob, alg, u_root, resid; retcode = ret,
                                    original = res, stats = stats)
end

end 

