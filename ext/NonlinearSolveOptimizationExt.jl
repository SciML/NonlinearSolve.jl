module NonlinearSolveOptimizationExt

using FastClosures, LinearAlgebra, NonlinearSolve, Optimization

function SciMLBase.__solve(
        prob::NonlinearProblem, alg::OptimizationJL, args...; abstol = nothing,
        maxiters = 1000, termination_condition = nothing, kwargs...)
    NonlinearSolve.__test_termination_condition(termination_condition, :OptimizationJL)

    prob.u0 isa Number &&
        throw(ArgumentError("`OptimizationJL` doesn't support scalar `u0`"))

    _objective_function = if SciMLBase.isinplace(prob)
        @closure (u, p) -> begin
            du = similar(u)
            prob.f(du, u, p)
            return norm(du, 2)
        end
    else
        @closure (u, p) -> norm(prob.f(u, p), 2)
    end

    cons = if SciMLBase.isinplace(prob)
        prob.f
    else
        @closure (du, u, p) -> copyto!(du, prob.f(u, p))
    end

    if alg.autodiff === nothing || alg.autodiff isa SciMLBase.NoAD
        opt_func = OptimizationFunction(_objective_function; cons)
    else
        opt_func = OptimizationFunction(_objective_function, alg.autodiff; cons)
    end
    bounds = similar(prob.u0)
    fill!(bounds, 0)
    opt_prob = OptimizationProblem(
        opt_func, prob.u0, prob.p; lcons = bounds, ucons = bounds)
    sol = solve(opt_prob, alg.solver, args...; abstol, maxiters, kwargs...)

    fu = zero(prob.u0)
    cons(fu, sol.u, prob.p)

    stats = SciMLBase.NLStats(sol.stats.fevals, sol.stats.gevals, -1, -1, -1)

    return SciMLBase.build_solution(
        prob, alg, sol.u, fu; retcode = sol.retcode, original = sol, stats)
end

function SciMLBase.__solve(prob::NonlinearLeastSquaresProblem, alg::OptimizationJL, args...;
        abstol = nothing, maxiters = 1000, termination_condition = nothing, kwargs...)
    NonlinearSolve.__test_termination_condition(termination_condition, :OptimizationJL)

    _objective_function = if SciMLBase.isinplace(prob)
        @closure (θ, p) -> begin
            resid = prob.f.resid_prototype === nothing ? similar(θ) :
                    similar(prob.f.resid_prototype, eltype(θ))
            prob.f(resid, θ, p)
            return norm(resid, 2)
        end
    else
        @closure (θ, p) -> norm(prob.f(θ, p), 2)
    end

    if alg.autodiff === nothing || alg.autodiff isa SciMLBase.NoAD
        opt_func = OptimizationFunction(_objective_function)
    else
        opt_func = OptimizationFunction(_objective_function, alg.autodiff)
    end
    opt_prob = OptimizationProblem(opt_func, prob.u0, prob.p)
    sol = solve(opt_prob, alg.solver, args...; abstol, maxiters, kwargs...)

    if SciMLBase.isinplace(prob)
        resid = prob.f.resid_prototype === nothing ? similar(prob.u0) :
                prob.f.resid_prototype
        prob.f(resid, sol.u, prob.p)
    else
        resid = prob.f(sol.u, prob.p)
    end

    stats = SciMLBase.NLStats(sol.stats.fevals, sol.stats.gevals, -1, -1, -1)

    return SciMLBase.build_solution(
        prob, alg, sol.u, resid; retcode = sol.retcode, original = sol, stats)
end

end
