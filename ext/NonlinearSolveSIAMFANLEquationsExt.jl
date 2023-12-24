module NonlinearSolveSIAMFANLEquationsExt

using NonlinearSolve, SciMLBase
using SIAMFANLEquations
import ConcreteStructs: @concrete
import UnPack: @unpack
import FiniteDiff, ForwardDiff

function SciMLBase.__solve(prob::NonlinearProblem, alg::SIAMFANLEquationsJL, args...; abstol = nothing,
        reltol = nothing, alias_u0::Bool = false, maxiters = 1000, termination_condition = nothing, kwargs...)
    @assert (termination_condition === nothing) || (termination_condition isa AbsNormTerminationMode) "SIAMFANLEquationsJL does not support termination conditions!"

    @unpack method, autodiff, show_trace, delta, linsolve = alg

    iip = SciMLBase.isinplace(prob)
    T = eltype(prob.u0)

    atol = abstol === nothing ? real(oneunit(T)) * (eps(real(one(T))))^(4 // 5) : abstol
    rtol = reltol === nothing ? real(oneunit(T)) * (eps(real(one(T))))^(4 // 5) : reltol

    if prob.u0 isa Number
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
            sol = nsolsc(f!, prob.u0; maxit = maxiters, atol = atol, rtol = rtol, printerr = show_trace)
        elseif method == :pseudotransient
            sol = ptcsolsc(f!, prob.u0; delta0 = delta, maxit = maxiters, atol = atol, rtol=rtol, printerr = show_trace)
        elseif method == :secant
            sol = secant(f!, prob.u0; maxit = maxiters, atol = atol, rtol = rtol, printerr = show_trace)
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

    # Allocate ahead for function
    N = length(u)
    FS = zeros(T, N)

    # Jacobian free Newton Krylov
    if linsolve !== nothing
        # Allocate ahead for Krylov basis
        JVS = linsolve == :gmres ? zeros(T, N, 3) : zeros(T, N)
        # `linsolve` as a Symbol to keep unified interface with other EXTs, SIAMFANLEquations directly use String to choose between different linear solvers
        linsolve_alg = String(linsolve)

        if method == :newton
            sol = nsoli(f!, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol = atol, rtol = rtol, printerr = show_trace)
        elseif method == :pseudotransient
            sol = ptcsoli(f!, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol = atol, rtol = rtol, printerr = show_trace)
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

    # Allocate ahead for Jacobian
    FPS = zeros(T, N, N)
    if prob.f.jac === nothing
        # Use the built-in Jacobian machinery
        if method == :newton
            sol = nsol(f!, u, FS, FPS;
                        sham=1, atol = atol, rtol = rtol, maxit = maxiters,
                        printerr = show_trace)
        elseif method == :pseudotransient
            sol = ptcsol(f!, u, FS, FPS;
                        atol = atol, rtol = rtol, maxit = maxiters,
                        delta0 = delta, printerr = show_trace)
        end
    else
        AJ!(J, u, x) = prob.f.jac(J, x, prob.p)
        if method == :newton
            sol = nsol(f!, u, FS, FPS, AJ!;
                        sham=1, atol = atol, rtol = rtol, maxit = maxiters,
                        printerr = show_trace)
        elseif method == :pseudotransient
            sol = ptcsol(f!, u, FS, FPS, AJ!;
                        atol = atol, rtol = rtol, maxit = maxiters,
                        delta0 = delta, printerr = show_trace)
        end
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
    return SciMLBase.build_solution(prob, alg, sol.solution, sol.history; retcode, stats, original = sol)
end

end