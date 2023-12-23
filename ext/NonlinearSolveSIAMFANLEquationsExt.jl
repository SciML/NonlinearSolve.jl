module NonlinearSolveSIAMFANLEquationsExt

using NonlinearSolve, SciMLBase
using SIAMFANLEquations
import ConcreteStructs: @concrete
import UnPack: @unpack
import FiniteDiff, ForwardDiff

function SciMLBase.__solve(prob::NonlinearProblem, alg::SIAMFANLEquationsJL, args...; abstol = 1e-8,
        reltol = 1e-8, alias_u0::Bool = false, maxiters = 1000, termination_condition = nothing, kwargs...)
    @assert (termination_condition === nothing) || (termination_condition isa AbsNormTerminationMode) "SIAMFANLEquationsJL does not support termination conditions!"

    @unpack method, autodiff, show_trace, delta, linsolve = alg

    iip = SciMLBase.isinplace(prob)
    if typeof(prob.u0) <: Number
        f! = if iip
            function (u)
                du = similar(u)
                prob.f(du, u, prob.p)
                return du
            end
        else
            u -> prob.f(u, prob.p)
        end

        if method == :newton
            sol = nsolsc(f!, prob.u0; maxit = maxiters, atol = abstol, rtol = reltol, printerr = show_trace)
        elseif method == :pseudotransient
            sol = ptcsolsc(f!, prob.u0; delta0 = delta, maxit = maxiters, atol = abstol, rtol=reltol, printerr = show_trace)
        elseif method == :secant
            sol = secant(f!, prob.u0; maxit = maxiters, atol = abstol, rtol = reltol, printerr = show_trace)
        end

        if sol.errcode == 0
            retcode = ReturnCode.Success
        elseif sol.errcode == 10
            retcode = ReturnCode.MaxIters
        elseif sol.errcode == 1
            retcode = ReturnCode.Failure
        elseif sol.errcode == -1
            retcode = ReturnCode.Default
        end
        stats = method == :pseudotransient ? nothing : (SciMLBase.NLStats(sum(sol.stats.ifun), sum(sol.stats.ijac), 0, 0, sum(sol.stats.iarm)))
        return SciMLBase.build_solution(prob, alg, sol.solution, sol.history; retcode, stats, original = sol)
    else
        u = NonlinearSolve.__maybe_unaliased(prob.u0, alias_u0)
    end

    fu = NonlinearSolve.evaluate_f(prob, u)

    if iip
        f! = function (du, u)
            prob.f(du, u, prob.p)
            return du
        end
    else
        f! = function (du, u)
            du .= prob.f(u, prob.p)
            return du
        end
    end

    # Allocate ahead for function and Jacobian
    N = length(u)
    FS = zeros(eltype(u), N)
    FPS = zeros(eltype(u), N, N)
    # Allocate ahead for Krylov basis

    # Jacobian free Newton Krylov
    if linsolve !== nothing
        JVS = linsolve == :gmres ? zeros(eltype(u), N, 3) : zeros(eltype(u), N)
        # `linsolve` as a Symbol to keep unified interface with other EXTs, SIAMFANLEquations directly use String to choose between different linear solvers
        linsolve_alg = String(linsolve)

        if method == :newton
            sol = nsoli(f!, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol = abstol, rtol = reltol, printerr = show_trace)
        elseif method == :pseudotransient
            sol = ptcsoli(f!, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol = abstol, rtol = reltol, printerr = show_trace)
        end
        
        if sol.errcode == 0
            retcode = ReturnCode.Success
        elseif sol.errcode == 10
            retcode = ReturnCode.MaxIters
        elseif sol.errcode == 1
            retcode = ReturnCode.Failure
        elseif sol.errcode == -1
            retcode = ReturnCode.Default
        end
        stats = method == :pseudotransient ? nothing : (SciMLBase.NLStats(sum(sol.stats.ifun), sum(sol.stats.ijac), 0, 0, sum(sol.stats.iarm)))
        return SciMLBase.build_solution(prob, alg, sol.solution, sol.history; retcode, stats, original = sol)
    end

    if prob.f.jac === nothing
        use_forward_diff = if alg.autodiff === nothing
            ForwardDiff.can_dual(eltype(u))
        else
            alg.autodiff isa AutoForwardDiff
        end
        uf = SciMLBase.JacobianWrapper{iip}(prob.f, prob.p)
        if use_forward_diff
            cache = iip ? ForwardDiff.JacobianConfig(uf, fu, u) :
                    ForwardDiff.JacobianConfig(uf, u)
        else
            cache = FiniteDiff.JacobianCache(u, fu)
        end
        J! = if iip
            if use_forward_diff
                fu_cache = similar(fu)
                function (J, x, p)
                    uf.p = p
                    ForwardDiff.jacobian!(J, uf, fu_cache, x, cache)
                    return J
                end
            else
                function (J, x, p)
                    uf.p = p
                    FiniteDiff.finite_difference_jacobian!(J, uf, x, cache)
                    return J
                end
            end
        else
            if use_forward_diff
                function (J, x, p)
                    uf.p = p
                    ForwardDiff.jacobian!(J, uf, x, cache)
                    return J
                end
            else
                function (J, x, p)
                    uf.p = p
                    J_ = FiniteDiff.finite_difference_jacobian(uf, x, cache)
                    copyto!(J, J_)
                    return J
                end
            end
        end
    else
        J! = prob.f.jac
    end

    AJ!(J, u, x) = J!(J, x, prob.p)    

    if method == :newton
        sol = nsol(f!, u, FS, FPS, AJ!;
                    sham=1, rtol = reltol, atol = abstol, maxit = maxiters,
                    printerr = show_trace)
    elseif method == :pseudotransient
        sol = ptcsol(f!, u, FS, FPS, AJ!;
                    rtol = reltol, atol = abstol, maxit = maxiters,
                    delta0 = delta, printerr = show_trace)
    end

    if sol.errcode == 0
        retcode = ReturnCode.Success
    elseif sol.errcode == 10
        retcode = ReturnCode.MaxIters
    elseif sol.errcode == 1
        retcode = ReturnCode.Failure
    elseif sol.errcode == -1
        retcode = ReturnCode.Default
    end

    # pseudo transient continuation has a fixed cost per iteration, iteration statistics are not interesting here.
    stats = method == :pseudotransient ? nothing : (SciMLBase.NLStats(sum(sol.stats.ifun), sum(sol.stats.ijac), 0, 0, sum(sol.stats.iarm)))
    println(sol.stats)
    return SciMLBase.build_solution(prob, alg, sol.solution, sol.history; retcode, stats, original = sol)
end

end