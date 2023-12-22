module NonlinearSolveSpeedMappingExt

using NonlinearSolve, SpeedMapping, DiffEqBase, SciMLBase
import UnPack: @unpack

function SciMLBase.__solve(prob::NonlinearProblem, alg::SpeedMappingJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        store_trace::Val{store_info} = Val(false), termination_condition = nothing,
        kwargs...) where {store_info}
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "SpeedMappingJL does not support termination conditions!"

    if typeof(prob.u0) <: Number
        u0 = [prob.u0]
    else
        u0 = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    end

    T = eltype(u0)
    iip = isinplace(prob)
    p = prob.p

    if prob.u0 isa Number
        resid = [NonlinearSolve.evaluate_f(prob, first(u0))]
    else
        resid = NonlinearSolve.evaluate_f(prob, u0)
    end

    if !iip && prob.u0 isa Number
        m! = (du, u) -> (du .= prob.f(first(u), p) .+ first(u))
    elseif !iip
        m! = (du, u) -> (du .= prob.f(u, p) .+ u)
    else
        m! = (du, u) -> (prob.f(du, u, p); du .+= u)
    end

    tol = abstol === nothing ? real(oneunit(T)) * (eps(real(one(T))))^(4 // 5) : abstol

    sol = speedmapping(u0; m!, tol, Lp = Inf, maps_limit = maxiters, alg.orders,
        alg.check_obj, store_info, alg.σ_min, alg.stabilize)
    res = prob.u0 isa Number ? first(sol.minimizer) : sol.minimizer
    resid = NonlinearSolve.evaluate_f(prob, sol.minimizer)

    return SciMLBase.build_solution(prob, alg, res, resid;
        retcode = sol.converged ? ReturnCode.Success : ReturnCode.Failure,
        stats = SciMLBase.NLStats(sol.maps, 0, 0, 0, sol.maps), original = sol)
end

end