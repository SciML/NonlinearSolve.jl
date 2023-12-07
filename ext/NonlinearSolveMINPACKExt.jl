module NonlinearSolveMINPACKExt

using NonlinearSolve, SciMLBase
using MINPACK

function SciMLBase.__solve(prob::Union{NonlinearProblem{uType, iip},
            NonlinearLeastSquaresProblem{uType, iip}}, alg::CMINPACK, args...;
        abstol = 1e-6, maxiters = 100000, alias_u0::Bool = false,
        kwargs...) where {uType, iip}
    if prob.u0 isa Number
        u0 = [prob.u0]
    else
        u0 = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    end

    sizeu = size(prob.u0)
    p = prob.p

    # unwrapping alg params
    show_trace = alg.show_trace
    tracing = alg.tracing

    if !iip && prob.u0 isa Number
        f! = (du, u) -> (du .= prob.f(first(u), p); Cint(0))
    elseif !iip && prob.u0 isa Vector{Float64}
        f! = (du, u) -> (du .= prob.f(u, p); Cint(0))
    elseif !iip && prob.u0 isa AbstractArray
        f! = (du, u) -> (du .= vec(prob.f(reshape(u, sizeu), p)); Cint(0))
    elseif prob.u0 isa Vector{Float64}
        f! = (du, u) -> prob.f(du, u, p)
    else # Then it's an in-place function on an abstract array
        f! = (du, u) -> (prob.f(reshape(du, sizeu), reshape(u, sizeu), p); du = vec(du); 0)
    end

    u = zero(u0)
    resid = NonlinearSolve.evaluate_f(prob, u)
    m = length(resid)
    size_jac = (length(resid), length(u))

    method = ifelse(alg.method === :auto,
        ifelse(prob isa NonlinearLeastSquaresProblem, :lm, :hybr), alg.method)

    if SciMLBase.has_jac(prob.f)
        if !iip && prob.u0 isa Number
            g! = (du, u) -> (du .= prob.f.jac(first(u), p); Cint(0))
        elseif !iip && prob.u0 isa Vector{Float64}
            g! = (du, u) -> (du .= prob.f.jac(u, p); Cint(0))
        elseif !iip && prob.u0 isa AbstractArray
            g! = (du, u) -> (du .= vec(prob.f.jac(reshape(u, sizeu), p)); Cint(0))
        elseif prob.u0 isa Vector{Float64}
            g! = (du, u) -> prob.f.jac(du, u, p)
        else # Then it's an in-place function on an abstract array
            g! = function (du, u)
                prob.f.jac(reshape(du, size_jac), reshape(u, sizeu), p)
                return Cint(0)
            end
        end
        original = MINPACK.fsolve(f!, g!, u0, m; tol = abstol, show_trace, tracing, method,
            iterations = maxiters, kwargs...)
    else
        original = MINPACK.fsolve(f!, u0, m; tol = abstol, show_trace, tracing, method,
            iterations = maxiters, kwargs...)
    end

    u = reshape(original.x, size(u))
    resid = original.f
    retcode = original.converged ? ReturnCode.Success : ReturnCode.Failure

    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original)
end

end
