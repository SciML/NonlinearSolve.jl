module NonlinearSolveMINPACKExt

using NonlinearSolve, SciMLBase
using MINPACK

function SciMLBase.solve(prob::Union{SciMLBase.NonlinearProblem{uType, isinplace},
                                     SciMLBase.NonlinearLeastSquaresProblem{uType, isinplace}},
                         alg::CMINPACK,
                         reltol = 1e-3,
                         abstol = 1e-6,
                         maxiters = 100000,
                         timeseries = [],
                         ts = [],
                         ks = [], ;
                         kwargs...) where {uType, isinplace}
    if prob.u0 isa Number
        u0 = [prob.u0]
    else
        u0 = deepcopy(prob.u0)
    end

    sizeu = size(prob.u0)
    p = prob.p

    # unwrapping alg params
    method = alg.method
    show_trace = alg.show_trace
    tracing = alg.tracing
    io = alg.io

    if !isinplace && prob.u0 isa Number
        f! = (du, u) -> (du .= prob.f(first(u), p); Cint(0))
    elseif !isinplace && prob.u0 isa Vector{Float64}
        f! = (du, u) -> (du .= prob.f(u, p); Cint(0))
    elseif !isinplace && prob.u0 isa AbstractArray
        f! = (du, u) -> (du .= vec(prob.f(reshape(u, sizeu), p)); Cint(0))
    elseif prob.u0 isa Vector{Float64}
        f! = (du, u) -> prob.f(du, u, p)
    else # Then it's an in-place function on an abstract array
        f! = (du, u) -> (prob.f(reshape(du, sizeu), reshape(u, sizeu), p);
                         du = vec(du);
                         0)
    end

    u = zero(u0)
    resid = similar(u0)

    m = prob.f.resid_prototype === nothing ? length(u0) : length(prob.f.resid_prototype)

    if SciMLBase.has_jac(prob.f)
        if !isinplace && prob.u0 isa Number
            g! = (du, u) -> (du .= prob.jac(first(u), p); Cint(0))
        elseif !isinplace && prob.u0 isa Vector{Float64}
            g! = (du, u) -> (du .= prob.jac(u, p); Cint(0))
        elseif !isinplace && prob.u0 isa AbstractArray
            g! = (du, u) -> (du .= vec(prob.jac(reshape(u, sizeu), p)); Cint(0))
        elseif prob.u0 isa Vector{Float64}
            g! = (du, u) -> prob.jac(du, u, p)
        else # Then it's an in-place function on an abstract array
            g! = (du, u) -> (prob.jac(reshape(du, sizeu), reshape(u, sizeu), p);
                             du = vec(du);
                             0)
        end
        original = MINPACK.fsolve(f!, g!, u0, m;
                                  tol = abstol,
                                  show_trace, tracing, method,
                                  iterations = maxiters, io, kwargs...)
    else
        original = MINPACK.fsolve(f!, u0, m;
                                  tol = abstol,
                                  show_trace, tracing, method,
                                  iterations = maxiters, io, kwargs...)
    end

    u = reshape(original.x, size(u))
    resid = original.f
    retcode = original.converged ? ReturnCode.Success : ReturnCode.Failure
    SciMLBase.build_solution(prob, alg, u, resid; retcode = retcode,
                             original = original)
end

end