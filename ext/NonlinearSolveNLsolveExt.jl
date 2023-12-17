module NonlinearSolveNLsolveExt

using NonlinearSolve, NLsolve, DiffEqBase, SciMLBase
import UnPack: @unpack

function SciMLBase.__solve(prob::NonlinearProblem, alg::NLsolveJL, args...; abstol = 1e-6,
        maxiters = 1000, alias_u0::Bool = false, termination_condition = nothing, kwargs...)
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "NLsolveJL does not support termination conditions!"

    if typeof(prob.u0) <: Number
        u0 = [prob.u0]
    else
        u0 = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    end

    iip = isinplace(prob)

    sizeu = size(prob.u0)
    p = prob.p

    # unwrapping alg params
    @unpack method, autodiff, store_trace, extended_trace, linesearch, linsolve = alg
    @unpack factor, autoscale, m, beta, show_trace = alg

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

    if prob.u0 isa Number
        resid = [NonlinearSolve.evaluate_f(prob, first(u0))]
    else
        resid = NonlinearSolve.evaluate_f(prob, u0)
    end

    size_jac = (length(resid), length(u0))

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
        if prob.f.jac_prototype !== nothing
            J = zero(prob.f.jac_prototype)
            df = OnceDifferentiable(f!, g!, vec(u0), vec(resid), J)
        else
            df = OnceDifferentiable(f!, g!, vec(u0), vec(resid))
        end
    else
        df = OnceDifferentiable(f!, vec(u0), vec(resid); autodiff)
    end

    original = nlsolve(df, vec(u0); ftol = abstol, iterations = maxiters, method,
        store_trace, extended_trace, linesearch, linsolve, factor, autoscale, m, beta,
        show_trace)

    u = reshape(original.zero, size(u0))
    f!(vec(resid), vec(u))
    retcode = original.x_converged || original.f_converged ? ReturnCode.Success :
              ReturnCode.Failure
    stats = SciMLBase.NLStats(original.f_calls, original.g_calls, original.g_calls,
        original.g_calls, original.iterations)
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original, stats)
end

end
