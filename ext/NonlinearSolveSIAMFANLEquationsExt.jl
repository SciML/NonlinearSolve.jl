module NonlinearSolveSIAMFANLEquationsExt

using NonlinearSolve, SciMLBase
using SIAMFANLEquations

@inline function __siam_fanl_equations_retcode_mapping(sol)
    if sol.errcode == 0
        return ReturnCode.Success
    elseif sol.errcode == 10
        return ReturnCode.MaxIters
    elseif sol.errcode == 1
        return ReturnCode.Failure
    elseif sol.errcode == -1
        return ReturnCode.Default
    end
end

@inline function __zeros_like(x, args...)
    z = similar(x, args...)
    fill!(z, zero(eltype(x)))
    return z
end

# pseudo transient continuation has a fixed cost per iteration, iteration statistics are
# not interesting here.
@inline function __siam_fanl_equations_stats_mapping(method, sol)
    ((method === :pseudotransient) || (method === :anderson)) && return nothing
    return SciMLBase.NLStats(sum(sol.stats.ifun), sum(sol.stats.ijac), 0, 0,
        sum(sol.stats.iarm))
end

function SciMLBase.__solve(prob::NonlinearProblem, alg::SIAMFANLEquationsJL, args...;
        abstol = nothing, reltol = nothing, alias_u0::Bool = false, maxiters = 1000,
        termination_condition = nothing, show_trace::Val{ShT} = Val(false),
        kwargs...) where {ShT}
    @assert (termination_condition ===
             nothing)||(termination_condition isa AbsNormTerminationMode) "SIAMFANLEquationsJL does not support termination conditions!"

    (; method, delta, linsolve, m, beta) = alg

    T = eltype(prob.u0)
    atol = NonlinearSolve.DEFAULT_TOLERANCE(abstol, T)
    rtol = NonlinearSolve.DEFAULT_TOLERANCE(reltol, T)

    if prob.u0 isa Number
        f = method == :anderson ? (du, u) -> (du = prob.f(u, prob.p)) : ((u) -> prob.f(u, prob.p))

        if method == :newton
            sol = nsolsc(f, prob.u0; maxit = maxiters, atol, rtol, printerr = ShT)
        elseif method == :pseudotransient
            sol = ptcsolsc(f, prob.u0; delta0 = delta, maxit = maxiters, atol, rtol,
                printerr = ShT)
        elseif method == :secant
            sol = secant(f, prob.u0; maxit = maxiters, atol, rtol, printerr = ShT)
        elseif method == :anderson
            f, u = NonlinearSolve.__construct_f(prob; alias_u0,
                make_fixed_point = Val(true), can_handle_arbitrary_dims = Val(true))
            sol = aasol(f, [prob.u0], m, __zeros_like(u, 1, 2*m+4); maxit = maxiters,
                atol, rtol, beta = beta)
        end

        retcode = __siam_fanl_equations_retcode_mapping(sol)
        stats = __siam_fanl_equations_stats_mapping(method, sol)
        resid = NonlinearSolve.evaluate_f(prob, sol.solution[1])
        return SciMLBase.build_solution(prob, alg, sol.solution, resid; retcode,
            stats, original = sol)
    end

    f!, u = NonlinearSolve.__construct_f(prob; alias_u0,
        can_handle_arbitrary_dims = Val(true))

    # Allocate ahead for function
    N = length(u)
    FS = __zeros_like(u, N)

    # Jacobian free Newton Krylov
    if linsolve !== nothing
        # Allocate ahead for Krylov basis
        JVS = linsolve == :gmres ? __zeros_like(u, N, 3) : __zeros_like(u, N)
        # `linsolve` as a Symbol to keep unified interface with other EXTs,
        # SIAMFANLEquations directly use String to choose between different linear solvers
        linsolve_alg = String(linsolve)

        if method == :newton
            sol = nsoli(f!, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol,
                rtol, printerr = ShT)
        elseif method == :pseudotransient
            sol = ptcsoli(f!, u, FS, JVS; lsolver = linsolve_alg, maxit = maxiters, atol,
                rtol, printerr = ShT)
        end

        retcode = __siam_fanl_equations_retcode_mapping(sol)
        stats = __siam_fanl_equations_stats_mapping(method, sol)
        resid = NonlinearSolve.evaluate_f(prob, sol.solution)
        return SciMLBase.build_solution(prob, alg, sol.solution, resid; retcode,
            stats, original = sol)
    end

    # Allocate ahead for Jacobian
    FPS = __zeros_like(u, N, N)

    if prob.f.jac === nothing
        # Use the built-in Jacobian machinery
        if method == :newton
            sol = nsol(f!, u, FS, FPS; sham = 1, atol, rtol, maxit = maxiters,
                printerr = ShT)
        elseif method == :pseudotransient
            sol = ptcsol(f!, u, FS, FPS; atol, rtol, maxit = maxiters,
                delta0 = delta, printerr = ShT)
        elseif method == :anderson
            f!, u = NonlinearSolve.__construct_f(prob; alias_u0,
                can_handle_arbitrary_dims = Val(true), make_fixed_point = Val(true))
            sol = aasol(f!, u, m, zeros(T, N, 2*m+4), atol = atol, rtol = rtol,
                maxit = maxiters, beta = beta)
        end
    else
        AJ!(J, u, x) = prob.f.jac(J, x, prob.p)
        if method == :newton
            sol = nsol(f!, u, FS, FPS, AJ!; sham = 1, atol, rtol, maxit = maxiters,
                printerr = ShT)
        elseif method == :pseudotransient
            sol = ptcsol(f!, u, FS, FPS, AJ!; atol, rtol, maxit = maxiters,
                delta0 = delta, printerr = ShT)
        end
    end

    retcode = __siam_fanl_equations_retcode_mapping(sol)
    stats = __siam_fanl_equations_stats_mapping(method, sol)
    resid = NonlinearSolve.evaluate_f(prob, sol.solution)
    return SciMLBase.build_solution(prob, alg, sol.solution, resid; retcode, stats,
        original = sol)
end

end
