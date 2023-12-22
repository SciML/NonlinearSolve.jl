module NonlinearSolveFixedPointAccelerationExt

using NonlinearSolve, FixedPointAcceleration, DiffEqBase, SciMLBase

function SciMLBase.__solve(prob::NonlinearProblem, alg::FixedPointAccelerationJL, args...;
        abstol = nothing, maxiters = 1000, alias_u0::Bool = false,
        show_trace::Val{PrintReports} = Val(false), termination_condition = nothing,
        kwargs...) where {PrintReports}
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "SpeedMappingJL does not support termination conditions!"

    u0 = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    u_size = size(u0)
    T = eltype(u0)
    iip = isinplace(prob)
    p = prob.p

    if !iip && prob.u0 isa Number
        # FixedPointAcceleration makes the scalar problem into a vector problem
        f = (u) -> [prob.f(u[1], p) .+ u[1]]
    elseif !iip && prob.u0 isa AbstractVector
        f = (u) -> (prob.f(u, p) .+ u)
    elseif !iip && prob.u0 isa AbstractArray
        f = (u) -> vec(prob.f(reshape(u, u_size), p) .+ u)
    elseif iip && prob.u0 isa AbstractVector
        du = similar(u0)
        f = (u) -> (prob.f(du, u, p); du .+ u)
    else
        du = similar(u0)
        f = (u) -> (prob.f(du, reshape(u, u_size), p); vec(du) .+ u)
    end

    tol = abstol === nothing ? real(oneunit(T)) * (eps(real(one(T))))^(4 // 5) : abstol

    sol = fixed_point(f, NonlinearSolve._vec(u0); Algorithm = alg.algorithm,
        ConvergenceMetricThreshold = tol, MaxIter = maxiters, MaxM = alg.m,
        ExtrapolationPeriod = alg.extrapolation_period, Dampening = alg.dampening,
        PrintReports, ReplaceInvalids = alg.replace_invalids,
        ConditionNumberThreshold = alg.condition_number_threshold, quiet_errors = true)

    res = prob.u0 isa Number ? first(sol.FixedPoint_) : sol.FixedPoint_
    if res === missing
        resid = NonlinearSolve.evaluate_f(prob, u0)
        res = u0
        converged = false
    else
        resid = NonlinearSolve.evaluate_f(prob, res)
        converged = maximum(abs, resid) ≤ tol
    end
    return SciMLBase.build_solution(prob, alg, res, resid;
        retcode = converged ? ReturnCode.Success : ReturnCode.Failure,
        stats = SciMLBase.NLStats(sol.Iterations_, 0, 0, 0, sol.Iterations_),
        original = sol)
end

end
