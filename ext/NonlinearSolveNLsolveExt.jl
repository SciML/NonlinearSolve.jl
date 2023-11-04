module NonlinearSolveNLsolveExt

using NonlinearSolve
using NLsolve
using LineSearches
using DiffEqBase
using SciMLBase

function SciMLBase.__solve(prob::Union{SciMLBase.AbstractSteadyStateProblem,
        SciMLBase.AbstractNonlinearProblem},
    alg::algType,
    args...;
    abstol = 1e-6,
    maxiters = 1000,
    kwargs...) where {algType <: NonlinearSolve.SciMLNLSolveAlgorithm}
    if typeof(prob.u0) <: Number
        u0 = [prob.u0]
    else
        u0 = deepcopy(prob.u0)
    end

    iip = isinplace(prob)

    sizeu = size(prob.u0)
    p = prob.p

    # unwrapping alg params
    method = alg.method
    autodiff = alg.autodiff
    store_trace = alg.store_trace
    extended_trace = alg.extended_trace
    linesearch = alg.linesearch
    linsolve = alg.linsolve
    factor = alg.factor
    autoscale = alg.autoscale
    m = alg.m
    beta = alg.beta
    show_trace = alg.show_trace

    ### Fix the more general function to Sundials allowed style
    if typeof(prob.f) <: ODEFunction
        t = Inf
        if !iip && typeof(prob.u0) <: Number
            f! = (du, u) -> (du .= prob.f(first(u), p, t); Cint(0))
        elseif !iip && typeof(prob.u0) <: Vector{Float64}
            f! = (du, u) -> (du .= prob.f(u, p, t); Cint(0))
        elseif !iip && typeof(prob.u0) <: AbstractArray
            f! = (du, u) -> (du .= vec(prob.f(reshape(u, sizeu), p, t)); Cint(0))
        elseif typeof(prob.u0) <: Vector{Float64}
            f! = (du, u) -> prob.f(du, u, p, t)
        else # Then it's an in-place function on an abstract array
            f! = (du, u) -> (prob.f(reshape(du, sizeu), reshape(u, sizeu), p, t);
            du = vec(du);
            0)
        end
    elseif typeof(prob.f) <: NonlinearFunction
        if !iip && typeof(prob.u0) <: Number
            f! = (du, u) -> (du .= prob.f(first(u), p); Cint(0))
        elseif !iip && typeof(prob.u0) <: Vector{Float64}
            f! = (du, u) -> (du .= prob.f(u, p); Cint(0))
        elseif !iip && typeof(prob.u0) <: AbstractArray
            f! = (du, u) -> (du .= vec(prob.f(reshape(u, sizeu), p)); Cint(0))
        elseif typeof(prob.u0) <: Vector{Float64}
            f! = (du, u) -> prob.f(du, u, p)
        else # Then it's an in-place function on an abstract array
            f! = (du, u) -> (prob.f(reshape(du, sizeu), reshape(u, sizeu), p);
            du = vec(du);
            0)
        end
    end

    resid = similar(u0)
    f!(resid, u0)

    if SciMLBase.has_jac(prob.f)
        if !iip && typeof(prob.u0) <: Number
            g! = (du, u) -> (du .= prob.f.jac(first(u), p); Cint(0))
        elseif !iip && typeof(prob.u0) <: Vector{T} where {T <: Number}
            g! = (du, u) -> (du .= prob.f.jac(u, p); Cint(0))
        elseif !iip && typeof(prob.u0) <: AbstractArray
            g! = (du, u) -> (du .= vec(prob.f.jac(reshape(u, sizeu), p)); Cint(0))
        elseif typeof(prob.u0) <: Vector{T} where {T <: Number}
            g! = (du, u) -> prob.f.jac(du, u, p)
        else # Then it's an in-place function on an abstract array
            g! = (du, u) -> (prob.f.jac(reshape(du, sizeu), reshape(u, sizeu), p);
            du = vec(du);
            0)
        end
        if prob.f.jac_prototype !== nothing
            J = zero(prob.f.jac_prototype)
            df = OnceDifferentiable(f!, g!, u0, resid, J)
        else
            df = OnceDifferentiable(f!, g!, u0, resid)
        end
    else
        df = OnceDifferentiable(f!, u0, resid, autodiff = autodiff)
    end

    original = nlsolve(df, u0,
        ftol = abstol,
        iterations = maxiters,
        method = method,
        store_trace = store_trace,
        extended_trace = extended_trace,
        linesearch = linesearch,
        linsolve = linsolve,
        factor = factor,
        autoscale = autoscale,
        m = m,
        beta = beta,
        show_trace = show_trace)

    u = reshape(original.zero, size(u0))
    f!(resid, u)
    retcode = original.x_converged || original.f_converged ? ReturnCode.Success :
              ReturnCode.Failure
    stats = SciMLBase.NLStats(original.f_calls,
        original.g_calls,
        original.g_calls,
        original.g_calls,
        original.iterations)
    SciMLBase.build_solution(prob, alg, u, resid; retcode = retcode,
        original = original, stats = stats)
end

end